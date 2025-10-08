"""
Anatomical Constraints for Echocardiogram Views
Implements view-specific detection constraints to improve mAP and accuracy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class AnatomicalConstraints:
    """
    Anatomical constraints for echocardiogram views based on medical knowledge.
    Each view has specific anatomical limitations for valve regurgitation detection.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Define anatomical constraints for each view
        # Key: view_index, Value: list of allowed detection classes
        self.constraints = {
            0: [1, 3],  # A4C: MR, TR (Mitral, Tricuspid)
            1: [2, 3],  # PSAX: PR, TR (Pulmonary, Tricuspid) 
            2: [0, 1],  # PLAX: AR, MR (Aortic, Mitral)
        }
        
        # View names for debugging
        self.view_names = ['A4C', 'PSAX', 'PLAX']
        self.detection_names = ['AR', 'MR', 'PR', 'TR']
        
        # Constraint penalties (higher = more strict)
        self.constraint_penalty = 10.0
        
        # Soft constraint weights (for gradual learning)
        self.soft_weights = {
            0: {1: 1.0, 3: 1.0, 0: 0.1, 2: 0.1},  # A4C
            1: {2: 1.0, 3: 1.0, 0: 0.1, 1: 0.1},  # PSAX
            2: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.1},  # PLAX (PR impossible)
        }
    
    def get_constraint_mask(self, classification_labels: torch.Tensor) -> torch.Tensor:
        """
        Generate constraint mask based on classification labels.
        
        Args:
            classification_labels: [batch_size, num_classes] one-hot encoded
            
        Returns:
            constraint_mask: [batch_size, num_detection_classes] constraint weights
        """
        batch_size, num_classes = classification_labels.shape
        constraint_mask = torch.ones(batch_size, 4, device=self.device)
        
        # Get view indices (which view is active for each sample)
        view_indices = torch.argmax(classification_labels, dim=1)
        
        for i, view_idx in enumerate(view_indices):
            view_idx = view_idx.item()
            
            # Apply soft constraints based on view
            for det_class in range(4):
                if det_class in self.soft_weights[view_idx]:
                    constraint_mask[i, det_class] = self.soft_weights[view_idx][det_class]
                else:
                    constraint_mask[i, det_class] = 0.0
        
        return constraint_mask
    
    def apply_constraint_loss(self, 
                            detection_loss: torch.Tensor,
                            classification_labels: torch.Tensor,
                            detection_predictions: torch.Tensor) -> torch.Tensor:
        """
        Apply anatomical constraint loss to penalize impossible detections.
        
        Args:
            detection_loss: Original detection loss
            classification_labels: [batch_size, num_classes] one-hot encoded
            detection_predictions: Detection predictions
            
        Returns:
            Constraint-adjusted loss
        """
        constraint_mask = self.get_constraint_mask(classification_labels)
        
        # Calculate constraint penalty
        # Penalize detections that violate anatomical constraints
        view_indices = torch.argmax(classification_labels, dim=1)
        
        constraint_penalty = torch.zeros_like(detection_loss)
        
        for i, view_idx in enumerate(view_indices):
            view_idx = view_idx.item()
            allowed_classes = self.constraints[view_idx]
            
            # For each detection in the batch
            for det_class in range(4):
                if det_class not in allowed_classes:
                    # High penalty for impossible detections
                    class_mask = (detection_predictions[:, :, 4] == det_class)
                    if class_mask.any():
                        penalty = self.constraint_penalty * class_mask.float()
                        constraint_penalty += penalty.mean()
        
        return detection_loss + constraint_penalty
    
    def filter_predictions(self, 
                          predictions: torch.Tensor,
                          classification_labels: torch.Tensor,
                          confidence_threshold: float = 0.5) -> torch.Tensor:
        """
        Filter predictions based on anatomical constraints.
        
        Args:
            predictions: Detection predictions [batch_size, num_detections, 6]
            classification_labels: [batch_size, num_classes] one-hot encoded
            confidence_threshold: Minimum confidence for filtering
            
        Returns:
            Filtered predictions
        """
        batch_size = predictions.shape[0]
        filtered_predictions = []
        
        for i in range(batch_size):
            pred = predictions[i]
            view_idx = torch.argmax(classification_labels[i]).item()
            allowed_classes = self.constraints[view_idx]
            
            # Filter out impossible detections
            valid_mask = torch.zeros_like(pred[:, 0], dtype=torch.bool)
            
            for det_class in allowed_classes:
                class_mask = (pred[:, 4] == det_class) & (pred[:, 5] > confidence_threshold)
                valid_mask |= class_mask
            
            # Keep only valid predictions
            filtered_pred = pred[valid_mask]
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def get_view_statistics(self, dataset_path: str) -> Dict:
        """
        Analyze view-specific detection statistics from dataset.
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Statistics dictionary
        """
        import os
        import glob
        from collections import defaultdict
        
        stats = defaultdict(lambda: defaultdict(int))
        
        for split in ['train', 'valid', 'test']:
            labels_path = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(labels_path):
                continue
                
            label_files = glob.glob(os.path.join(labels_path, '*.txt'))
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) >= 2:
                        # Parse detection class
                        detection_parts = lines[0].strip().split()
                        if detection_parts:
                            detection_class = int(detection_parts[0])
                            detection_name = self.detection_names[detection_class]
                            
                            # Parse view class
                            cls_parts = lines[1].strip().split()
                            if len(cls_parts) == 3:
                                view_idx = cls_parts.index('1')
                                view_name = self.view_names[view_idx]
                                
                                stats[view_name][detection_name] += 1
                                
                except Exception:
                    continue
        
        return dict(stats)
    
    def print_constraints(self):
        """Print anatomical constraints for debugging."""
        print("Anatomical Constraints:")
        print("=" * 50)
        
        for view_idx, allowed_classes in self.constraints.items():
            view_name = self.view_names[view_idx]
            allowed_names = [self.detection_names[cls] for cls in allowed_classes]
            print(f"{view_name}: {', '.join(allowed_names)}")
        
        print("\nSoft Constraint Weights:")
        print("=" * 50)
        
        for view_idx, weights in self.soft_weights.items():
            view_name = self.view_names[view_idx]
            print(f"{view_name}:")
            for det_class, weight in weights.items():
                det_name = self.detection_names[det_class]
                print(f"  {det_name}: {weight}")


class ConstraintLoss(nn.Module):
    """
    Loss function that incorporates anatomical constraints.
    """
    
    def __init__(self, constraint_penalty: float = 10.0):
        super().__init__()
        self.constraints = AnatomicalConstraints()
        self.constraint_penalty = constraint_penalty
    
    def forward(self, 
                detection_loss: torch.Tensor,
                classification_labels: torch.Tensor,
                detection_predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate constraint-aware loss.
        """
        return self.constraints.apply_constraint_loss(
            detection_loss, classification_labels, detection_predictions
        )


def test_constraints():
    """Test function for anatomical constraints."""
    constraints = AnatomicalConstraints()
    constraints.print_constraints()
    
    # Test constraint mask generation
    batch_size = 3
    classification_labels = torch.tensor([
        [1, 0, 0],  # A4C
        [0, 1, 0],  # PSAX  
        [0, 0, 1],  # PLAX
    ])
    
    constraint_mask = constraints.get_constraint_mask(classification_labels)
    print(f"\nConstraint Mask:\n{constraint_mask}")


if __name__ == "__main__":
    test_constraints()
