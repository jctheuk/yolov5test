# YOLOv5 🚀 Classification Task Loss Implementation
# This script implements loss calculation using classification training approach
# Based on YOLOv5 classify/train.py methodology with cls_task support
# Does NOT calculate IOU, cls, obj losses - only classification task loss

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_utils import de_parallel


class ClassificationTaskLoss:
    """
    Classification Task Loss - Computes loss using classification training approach
    Similar to YOLOv5 classify/train.py but adapted for joint training with cls_task
    """
    
    def __init__(self, model, autobalance=False, 
                 enable_classification=True,
                 cls_task_weight=0.3,
                 label_smoothing=0.1,
                 classification_criterion=None,
                 class_weights=None):
        """
        Initialize Classification Task Loss
        
        Args:
            model: YOLOv5 model with classification head
            autobalance: Whether to use automatic loss balancing
            enable_classification: Whether to enable classification loss
            cls_task_weight: Weight for classification task loss
            label_smoothing: Label smoothing factor
            classification_criterion: Custom classification criterion
            class_weights: Class weights tensor for handling class imbalance
        """
        self.model = model
        self.autobalance = autobalance
        self.enable_classification = enable_classification
        self.cls_task_weight = cls_task_weight
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        
        # Set device from model
        self.device = next(model.parameters()).device
        
        # Convert class weights to tensor if provided
        if self.class_weights is not None:
            if isinstance(self.class_weights, (list, tuple)):
                self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
            print(f"Using class weights: {self.class_weights}")
        
        # Get model hyperparameters
        self.hyp = getattr(model, 'hyp', {})
        
        # Set up classification criterion - use manual implementation to avoid all PyTorch issues
        if classification_criterion is None:
            # Always use manual implementation to avoid any PyTorch version issues
            self.classification_criterion = None  # We'll use manual implementation
            self.use_builtin_smoothing = False
            if label_smoothing > 0:
                print(f"WARNING ⚠️ label smoothing {label_smoothing} will be ignored (using manual implementation)")
        else:
            self.classification_criterion = classification_criterion
            self.use_builtin_smoothing = False
            
        # Classification loss tracking
        self.classification_loss = 0.0
        self.total_loss = 0.0
        
        # Temperature for softmax sharpness (same as original loss.py)
        self.temperature = self.hyp.get('temperature', 1.0)
        
    def set_epoch(self, epoch):
        """Update epoch for dynamic weight adjustment"""
        self.epoch = epoch
        
    def get_classification_weight(self):
        """Get current classification loss weight"""
        if not self.enable_classification:
            return 0.0
        return self.cls_task_weight
    
    def manual_cross_entropy_loss(self, logits, targets):
        """
        Manual CrossEntropy loss implementation with class weights support
        Equivalent to nn.CrossEntropyLoss(weight=class_weights) but works on all PyTorch versions
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Target class indices [batch_size]
            
        Returns:
            CrossEntropy loss value
        """
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=1)
        
        # Gather the log probabilities for the target classes
        # targets should be long tensor with class indices
        batch_size = logits.shape[0]
        target_log_probs = log_probs[range(batch_size), targets]
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Get weights for each target class
            target_weights = self.class_weights[targets]
            # Weight the losses
            weighted_losses = -target_log_probs * target_weights
            return weighted_losses.mean()
        else:
            # Return negative log likelihood (CrossEntropy loss)
            return -target_log_probs.mean()
    
    def apply_label_smoothing(self, targets, num_classes, smoothing=0.1):
        """
        Apply label smoothing manually for compatibility across PyTorch versions
        This implementation matches PyTorch's built-in label smoothing exactly
        
        Args:
            targets: Target class indices [batch_size]
            num_classes: Number of classes
            smoothing: Label smoothing factor (0.0 = no smoothing)
            
        Returns:
            Smoothed target probabilities [batch_size, num_classes]
        """
        if smoothing == 0.0:
            # No smoothing, return one-hot encoding
            return F.one_hot(targets, num_classes=num_classes).float()
        
        # Apply label smoothing exactly like PyTorch's built-in implementation
        # Convert to one-hot first
        one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Apply smoothing: (1 - smoothing) * one_hot + smoothing / num_classes
        smoothed = (1.0 - smoothing) * one_hot + smoothing / num_classes
        
        return smoothed
    
    def standard_classification_loss(self, logits, targets):
        """
        Calculate standard CrossEntropy loss for classification task.
        Same as original loss.py method.
        
        Args:
            logits: Raw classification logits [batch_size, num_classes]
            targets: Target class indices [batch_size]
        
        Returns:
            CrossEntropy loss value
        """
        return self.classification_criterion(logits, targets)
    
    def classification_loss_fn(self, pred_classification, targets_classification):
        """
        Compute classification loss using CrossEntropyLoss
        Based on YOLOv5 classify/train.py approach
        
        Args:
            pred_classification: Model predictions for classification [batch_size, num_classes]
            targets_classification: Ground truth classification labels [batch_size]
            
        Returns:
            classification_loss: Computed classification loss
        """
        if not self.enable_classification or pred_classification is None:
            return torch.zeros(1, device=pred_classification.device if pred_classification is not None else 'cpu', requires_grad=True)
            
        # Ensure targets are long tensors for CrossEntropyLoss
        if targets_classification.dtype != torch.long:
            targets_classification = targets_classification.long()
            
        # Compute classification loss using CrossEntropyLoss
        classification_loss = self.classification_criterion(pred_classification, targets_classification)
        
        return classification_loss
    
    def analyze_loss_components(self, classification_output, cls_targets, loss_value, batch_idx=0):
        """
        Analyze loss components to diagnose potential issues
        
        Args:
            classification_output: Model predictions [batch_size, num_classes]
            cls_targets: Ground truth labels [batch_size]
            loss_value: Computed loss value
            batch_idx: Current batch index for logging
        """
        if classification_output is None or cls_targets is None:
            print(f"[DEBUG] Batch {batch_idx}: No data for loss analysis")
            return
            
        with torch.no_grad():
            # Ensure cls_targets are on the same device as classification_output
            cls_targets = cls_targets.to(classification_output.device)
            
            # Get predictions and probabilities
            pred_classes = torch.argmax(classification_output, dim=1)
            pred_probs = torch.softmax(classification_output, dim=1)
            
            # Labels should already be processed by train_classification_task.py
            # Just ensure they are in the correct format for CrossEntropyLoss
            if cls_targets.dim() > 1:
                # If still multi-dimensional, squeeze to 1D
                target_indices = cls_targets.squeeze()
            else:
                # Already 1D, keep as is
                target_indices = cls_targets
            
            # Always use long dtype for CrossEntropyLoss targets
            # CrossEntropyLoss requires integer class indices, not float
            target_indices = target_indices.long()
            
            print(f"\n[DEBUG] ===== BATCH {batch_idx} LOSS ANALYSIS =====")
            
            # 1. Check for extreme values in logits
            logits_min = classification_output.min().item()
            logits_max = classification_output.max().item()
            logits_mean = classification_output.mean().item()
            logits_std = classification_output.std().item()
            
            print(f"[DEBUG] Logits Statistics:")
            print(f"  Min: {logits_min:.4f}, Max: {logits_max:.4f}")
            print(f"  Mean: {logits_mean:.4f}, Std: {logits_std:.4f}")
            
            # Check for extreme logits (potential cause of loss issues)
            if abs(logits_max) > 10 or abs(logits_min) > 10:
                print(f"  ⚠️  WARNING: Extreme logit values detected!")
            if logits_std > 5:
                print(f"  ⚠️  WARNING: High logit variance detected!")
            
            # 2. Check probabilities
            prob_min = pred_probs.min().item()
            prob_max = pred_probs.max().item()
            prob_mean = pred_probs.mean().item()
            
            print(f"[DEBUG] Probability Statistics:")
            print(f"  Min: {prob_min:.4f}, Max: {prob_max:.4f}, Mean: {prob_mean:.4f}")
            
            # Check for extreme probabilities
            if prob_max > 0.99:
                print(f"  ⚠️  WARNING: Very high confidence predictions (potential overfitting)")
            if prob_min < 0.01:
                print(f"  ⚠️  WARNING: Very low confidence predictions")
            
            # 3. Analyze individual sample losses
            individual_losses = []
            for i in range(classification_output.shape[0]):
                # Calculate CrossEntropy loss for this sample
                sample_loss = F.cross_entropy(
                    classification_output[i:i+1], 
                    target_indices[i:i+1], 
                    reduction='none'
                )
                individual_losses.append(sample_loss)
            
            # Stack losses and ensure they're on the same device
            individual_losses = torch.cat(individual_losses, dim=0)
            loss_min = individual_losses.min().item()
            loss_max = individual_losses.max().item()
            loss_mean = individual_losses.mean().item()
            loss_std = individual_losses.std().item()
            
            print(f"[DEBUG] Individual Loss Statistics:")
            print(f"  Min: {loss_min:.4f}, Max: {loss_max:.4f}")
            print(f"  Mean: {loss_mean:.4f}, Std: {loss_std:.4f}")
            print(f"  Total batch loss: {loss_value:.4f}")
            
            # Check for loss anomalies
            if loss_max > 5.0:
                print(f"  ⚠️  WARNING: Very high individual losses detected!")
            if loss_std > 2.0:
                print(f"  ⚠️  WARNING: High loss variance (inconsistent predictions)")
            
            # 4. Analyze prediction vs target relationship
            correct_mask = (pred_classes == target_indices)
            correct_losses = individual_losses[correct_mask]
            incorrect_losses = individual_losses[~correct_mask]
            
            print(f"[DEBUG] Loss by Prediction Correctness:")
            if len(correct_losses) > 0:
                print(f"  Correct predictions: {len(correct_losses)} samples, avg loss: {correct_losses.mean():.4f}")
            if len(incorrect_losses) > 0:
                print(f"  Incorrect predictions: {len(incorrect_losses)} samples, avg loss: {incorrect_losses.mean():.4f}")
            
            # 5. Check for class imbalance in predictions
            unique_preds, pred_counts = torch.unique(pred_classes, return_counts=True)
            unique_targets, target_counts = torch.unique(target_indices, return_counts=True)
            
            print(f"[DEBUG] Class Distribution:")
            print(f"  Predicted classes: {unique_preds.tolist()}")
            print(f"  Predicted counts: {pred_counts.tolist()}")
            print(f"  Target classes: {unique_targets.tolist()}")
            print(f"  Target counts: {target_counts.tolist()}")
            
            # Check for prediction bias
            if len(unique_preds) == 1:
                print(f"  ⚠️  WARNING: Model predicting only one class (severe bias)")
            if len(unique_preds) < len(unique_targets):
                print(f"  ⚠️  WARNING: Model not predicting all target classes")
            
            # 6. Show detailed sample analysis for problematic cases
            high_loss_indices = torch.where(individual_losses > loss_mean + 2 * loss_std)[0]
            if len(high_loss_indices) > 0:
                print(f"[DEBUG] High Loss Samples (>{loss_mean + 2*loss_std:.4f}):")
                for idx in high_loss_indices[:3]:  # Show first 3
                    idx = idx.item()
                    pred_class = pred_classes[idx].item()
                    target_class = target_indices[idx].item()
                    confidence = pred_probs[idx, pred_class].item()
                    sample_loss = individual_losses[idx].item()
                    logits = classification_output[idx].cpu().numpy()
                    
                    print(f"  Sample {idx}: pred={pred_class}, target={target_class}, "
                          f"conf={confidence:.4f}, loss={sample_loss:.4f}")
                    print(f"    Logits: {logits}")
            
            print(f"[DEBUG] ===========================================\n")

    def print_predictions_and_labels(self, classification_output, cls_targets, batch_idx=0, max_samples=5, image_paths=None, class_names=None):
        """
        Print model predictions and ground truth labels for debugging
        
        Args:
            classification_output: Model predictions [batch_size, num_classes]
            cls_targets: Ground truth labels [batch_size]
            batch_idx: Current batch index for logging
            max_samples: Maximum number of samples to print
            image_paths: List of image paths for this batch
            class_names: List of class names (e.g., ['PSAX', 'PLAX', 'A4C'])
        """
        if classification_output is None or cls_targets is None:
            print(f"[DEBUG] Batch {batch_idx}: No classification data to print")
            return
            
        with torch.no_grad():
            # Ensure cls_targets are on the same device as classification_output
            cls_targets = cls_targets.to(classification_output.device)
            
            # Get predictions (argmax of logits)
            pred_classes = torch.argmax(classification_output, dim=1)
            pred_probs = torch.softmax(classification_output, dim=1)
            
            # Labels should already be processed by train_classification_task.py
            # Just ensure they are in the correct format for CrossEntropyLoss
            if cls_targets.dim() > 1:
                # If still multi-dimensional, squeeze to 1D
                target_indices = cls_targets.squeeze()
            else:
                # Already 1D, keep as is
                target_indices = cls_targets
            
            # Always use long dtype for CrossEntropyLoss targets
            # CrossEntropyLoss requires integer class indices, not float
            target_indices = target_indices.long()
            
            # Limit number of samples to print
            num_samples = min(max_samples, classification_output.shape[0])
            
            # Set default class names if not provided
            if class_names is None:
                class_names = ['Class_0', 'Class_1', 'Class_2']
            
            print(f"\n[DEBUG] Batch {batch_idx} - Model Predictions vs Ground Truth:")
            print(f"{'Sample':<8} {'Image Name (parent/name)':<44} {'Predicted':<15} {'Ground Truth':<15} {'Confidence':<12} {'Correct':<8}")
            print("-" * 112)

            def _tail2(p):
                p = Path(p)
                parent = p.parent.name
                name = p.name
                full = f"{parent}/{name}" if parent else name
                # middle-truncate to max 44 chars
                maxw = 44
                if len(full) <= maxw:
                    return full
                keep = maxw - 3
                left = keep // 2
                right = keep - left
                return full[:left] + '...' + full[-right:]
            
            for i in range(num_samples):
                pred_class = pred_classes[i].item()
                target_class = target_indices[i].item()
                confidence = pred_probs[i, pred_class].item()
                is_correct = "✓" if pred_class == target_class else "✗"
                
                # Get image name (filename only, no path)
                if image_paths is not None and i < len(image_paths):
                    image_name = _tail2(image_paths[i])
                else:
                    image_name = f"sample_{i}"
                
                # Get class names
                pred_class_name = class_names[pred_class] if pred_class < len(class_names) else f"Class_{pred_class}"
                target_class_name = class_names[target_class] if target_class < len(class_names) else f"Class_{target_class}"
                
                print(f"{i:<8} {image_name:<44} {pred_class_name:<15} {target_class_name:<15} {confidence:<12.4f} {is_correct:<8}")
            
            # Print batch statistics
            correct_predictions = (pred_classes == target_indices).sum().item()
            total_predictions = pred_classes.shape[0]
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            print(f"\n[DEBUG] Batch Statistics:")
            print(f"  Total samples: {total_predictions}")
            print(f"  Correct predictions: {correct_predictions}")
            print(f"  Batch accuracy: {accuracy:.4f}")

    def __call__(self, p, targets, cls_targets=None, image_paths=None, class_names=None):  # predictions, targets, classification_targets
        """
        Compute classification-only loss using train.py approach
        Pure classification loss - NO detection losses (box, obj, cls)
        """
        # Handle dual outputs: p can be either detection outputs only or (detection_outputs, classification_output)
        if isinstance(p, tuple) and len(p) == 2:
            detection_outputs, classification_output = p
        else:
            detection_outputs = p
            classification_output = None

        # Initialize loss components
        # Detection losses: keep constant zeros for logging (do not participate in gradients)
        lcls = torch.zeros(1, device=self.device)  # detection cls loss (ZERO)
        lbox = torch.zeros(1, device=self.device)  # detection box loss (ZERO)
        lobj = torch.zeros(1, device=self.device)  # detection obj loss (ZERO)
        # Classification loss will be computed below if available
        lcls_task = None
        
        # Calculate classification loss using train.py approach (simple CrossEntropyLoss)
        if classification_output is not None and cls_targets is not None:
            # Ensure cls_targets are on the same device as classification_output
            cls_targets = cls_targets.to(classification_output.device)
            
            # Labels should already be processed by train_classification_task.py
            # Handle both one-hot encoded and class indices formats
            if cls_targets.dim() > 1 and cls_targets.shape[-1] > 1:
                # One-hot encoded: [batch_size, num_classes] -> [batch_size]
                target_indices = cls_targets.argmax(dim=-1).long()
            elif cls_targets.dim() > 1:
                # Class indices with extra dim: [batch_size, 1] -> [batch_size]
                target_indices = cls_targets.squeeze().long()
            else:
                # Already 1D class indices
                target_indices = cls_targets.long()
            num_classes = classification_output.shape[-1]
            
            # Ensure targets are within valid range
            if target_indices.max() >= num_classes:
                target_indices = torch.clamp(target_indices, 0, num_classes - 1)
            
            # Use manual CrossEntropy implementation to avoid all PyTorch version issues
            if self.classification_criterion is not None:
                # Use provided criterion if available
                lcls_task = self.classification_criterion(classification_output, target_indices)
            else:
                # Use manual implementation (always works)
                lcls_task = self.manual_cross_entropy_loss(classification_output, target_indices)
                
            # Print predictions and labels for debugging (only for first few batches)
            # This will help debug the model(images) predictions vs ground truth labels
            if hasattr(self, '_debug_batch_count'):
                self._debug_batch_count += 1
            else:
                self._debug_batch_count = 0
            
            # Print predictions for first batch only
            if self._debug_batch_count == 1:
                self.print_predictions_and_labels(
                    classification_output,
                    cls_targets,
                    batch_idx=self._debug_batch_count,
                    max_samples=5,
                    image_paths=image_paths,
                    class_names=class_names,
                )
            
            # Print predictions for final batch only
            is_final_batch = getattr(self, '_is_final_batch', False)
            if is_final_batch:
                print(f"\n[DEBUG] ===== FINAL BATCH PREDICTIONS AND GROUND TRUTH =====")
                self.print_predictions_and_labels(
                    classification_output,
                    cls_targets,
                    batch_idx=f"final_{self._debug_batch_count}",
                    max_samples=8,
                    image_paths=image_paths,
                    class_names=class_names,
                )
                # Reset the flag
                self._is_final_batch = False

        # Total loss - match classify/ behavior: mean CE (no batch-size scaling)
        total_loss = lcls_task  # may be None if no classification outputs in this batch
        
        # Check for NaN/Inf in total loss (when available)
        if total_loss is not None and (torch.isnan(total_loss) or torch.isinf(total_loss)):
            print(f"[DEBUG] WARNING: NaN/Inf detected in total_loss!")
            print(f"[DEBUG]   lcls_task: {lcls_task.item():.6f}")
        
        # Ensure all loss components are properly shaped tensors (same as original loss.py format)
        def ensure_tensor_shape(tensor):
            if tensor.numel() == 0:
                return torch.zeros(1, device=self.device, requires_grad=True)
            elif tensor.dim() == 0:
                return tensor.unsqueeze(0)
            else:
                return tensor
        
        # Return in same format as original loss.py: [lbox, lobj, lcls, lcls_task]
        # lbox, lobj, lcls are constant ZERO (detection disabled); lcls_task may be None if unavailable
        lbox_final = ensure_tensor_shape(lbox.detach()).view(1)
        lobj_final = ensure_tensor_shape(lobj.detach()).view(1)
        lcls_final = ensure_tensor_shape(lcls.detach()).view(1)
        if lcls_task is None:
            lcls_task_final = torch.zeros(1, device=self.device).view(1)
        else:
            lcls_task_final = ensure_tensor_shape(lcls_task.detach()).view(1)
        
        return total_loss, [lbox_final, lobj_final, lcls_final, lcls_task_final]
    
    def update(self, epoch):
        """Update loss function for new epoch"""
        self.set_epoch(epoch)
        
    def get_loss_info(self):
        """Get current loss information"""
        return {
            'classification_loss': self.classification_loss,
            'total_loss': self.total_loss,
            'classification_weight': self.get_classification_weight(),
            'enable_classification': self.enable_classification
        }
    
    def set_final_batch_flag(self, is_final=True):
        """Set flag to indicate this is a final batch for debugging"""
        self._is_final_batch = is_final


class SmartCrossEntropyLoss(nn.Module):
    """
    Smart Cross Entropy Loss with label smoothing
    Based on YOLOv5 classify/train.py smartCrossEntropyLoss
    """
    
    def __init__(self, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, pred, targets):
        """
        Forward pass with label smoothing
        
        Args:
            pred: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            loss: Computed loss with label smoothing
        """
        if self.label_smoothing > 0:
            # Apply label smoothing
            num_classes = pred.size(-1)
            log_pred = F.log_softmax(pred, dim=-1)
            
            # Create smooth targets
            smooth_targets = torch.zeros_like(log_pred)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing)
            
            # Compute loss
            loss = (-smooth_targets * log_pred).sum(dim=-1)
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            # Standard CrossEntropyLoss
            return F.cross_entropy(pred, targets, reduction=self.reduction)


def create_classification_task_loss(model, **kwargs):
    """
    Factory function to create ClassificationTaskLoss
    
    Args:
        model: YOLOv5 model
        **kwargs: Additional arguments for ClassificationTaskLoss
        
    Returns:
        ClassificationTaskLoss instance
    """
    return ClassificationTaskLoss(model, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the classification task loss
    import torch
    
    # Create mock model
    class MockModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.hyp = {'cls_task': 0.3, 'label_smoothing': 0.1}
            self.classifier = nn.Linear(512, num_classes)
            
        def forward(self, x):
            return self.classifier(x)
    
    # Test parameters
    batch_size = 8
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and loss
    model = MockModel(num_classes).to(device)
    loss_fn = ClassificationTaskLoss(
        model, 
        enable_classification=True,
        cls_task_weight=0.3,
        label_smoothing=0.1
    )
    
    # Create mock data
    pred_classification = torch.randn(batch_size, num_classes, device=device)
    targets_classification = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Test loss computation
    preds = (None, pred_classification)  # (detection, classification)
    targets = (None, targets_classification)  # (detection, classification)
    
    total_loss, loss_items = loss_fn(preds, targets)
    
    print(f"✅ Classification Task Loss Test Results:")
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Classification Loss: {loss_items[0].item():.4f}")
    print(f"   Classification Weight: {loss_fn.get_classification_weight():.3f}")
    print(f"   Loss Info: {loss_fn.get_loss_info()}")
    
    # Test with different epochs
    print(f"\n📊 Testing different epochs:")
    for epoch in [0, 5, 10, 15, 20]:
        loss_fn.set_epoch(epoch)
        weight = loss_fn.get_classification_weight()
        print(f"   Epoch {epoch:2d}: Classification weight = {weight:.3f}")
    
    print(f"\n🎯 Classification Task Loss implementation completed successfully!")
