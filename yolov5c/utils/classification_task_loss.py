# YOLOv5 🚀 Classification Task Loss Implementation
# This script implements loss calculation using classification training approach
# Based on YOLOv5 classify/train.py methodology with cls_task support
# Does NOT calculate IOU, cls, obj losses - only classification task loss

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
                 classification_criterion=None):
        """
        Initialize Classification Task Loss
        
        Args:
            model: YOLOv5 model with classification head
            autobalance: Whether to use automatic loss balancing
            enable_classification: Whether to enable classification loss
            cls_task_weight: Weight for classification task loss
            label_smoothing: Label smoothing factor
            classification_criterion: Custom classification criterion
        """
        self.model = model
        self.autobalance = autobalance
        self.enable_classification = enable_classification
        self.cls_task_weight = cls_task_weight
        self.label_smoothing = label_smoothing
        
        # Get model hyperparameters
        self.hyp = getattr(model, 'hyp', {})
        
        # Set up classification criterion
        if classification_criterion is None:
            # Use CrossEntropyLoss with label smoothing (like classify/train.py)
            self.classification_criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )
        else:
            self.classification_criterion = classification_criterion
            
        # Classification loss tracking
        self.classification_loss = 0.0
        self.total_loss = 0.0
        
    def set_epoch(self, epoch):
        """Update epoch for dynamic weight adjustment"""
        self.epoch = epoch
        
    def get_classification_weight(self):
        """Get current classification loss weight"""
        if not self.enable_classification:
            return 0.0
        return self.cls_task_weight
    
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
            return torch.tensor(0.0, device=pred_classification.device if pred_classification is not None else 'cpu')
            
        # Ensure targets are long tensors for CrossEntropyLoss
        if targets_classification.dtype != torch.long:
            targets_classification = targets_classification.long()
            
        # Compute classification loss using CrossEntropyLoss
        classification_loss = self.classification_criterion(pred_classification, targets_classification)
        
        return classification_loss
    
    def __call__(self, preds, targets):
        """
        Compute total loss using classification task approach
        
        Args:
            preds: Model predictions (tuple of detection and classification predictions)
            targets: Ground truth targets
            
        Returns:
            total_loss: Total computed loss
            loss_items: Tuple of individual loss components
        """
        # Get device from predictions (handle None values)
        if isinstance(preds, (list, tuple)):
            device = None
            for pred in preds:
                if pred is not None:
                    device = pred.device
                    break
            if device is None:
                device = torch.device('cpu')
        else:
            device = preds.device
        
        # Initialize loss components
        classification_loss = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        
        # Extract classification predictions and targets
        if isinstance(preds, (list, tuple)) and len(preds) >= 2:
            # Joint model: preds[0] = detection, preds[1] = classification
            pred_classification = preds[1] if preds[1] is not None else None
        else:
            # Pure classification model
            pred_classification = preds
            
        # Extract classification targets
        if isinstance(targets, (list, tuple)) and len(targets) >= 2:
            targets_classification = targets[1] if targets[1] is not None else None
        else:
            targets_classification = targets
            
        # Compute classification loss if enabled and data available
        if (self.enable_classification and 
            pred_classification is not None and 
            targets_classification is not None):
            
            classification_loss = self.classification_loss_fn(
                pred_classification, targets_classification
            )
            
            # Apply classification weight
            classification_loss = classification_loss * self.get_classification_weight()
            
        # Total loss is only classification loss (no IOU, cls, obj)
        total_loss = classification_loss
        
        # Store for tracking
        self.classification_loss = classification_loss.item() if hasattr(classification_loss, 'item') else 0.0
        self.total_loss = total_loss.item() if hasattr(total_loss, 'item') else 0.0
        
        # Return loss items in format: (classification_loss, total_loss, 0.0)
        # Third element is 0.0 to maintain compatibility with standard loss format
        loss_items = (classification_loss, total_loss, torch.tensor(0.0, device=device))
        
        return total_loss, loss_items
    
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
