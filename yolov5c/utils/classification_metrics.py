import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from utils.general import LOGGER


def compute_classification_metrics(predictions, targets, class_names=None):
    """
    Compute classification metrics for multi-class classification.
    
    Args:
        predictions: Tensor of shape (N, num_classes) with logits or probabilities
        targets: Tensor of shape (N,) with true class labels
        class_names: List of class names for reporting
        
    Returns:
        dict: Dictionary containing various classification metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Apply standardized one-hot encoding handling to targets
    if isinstance(targets, np.ndarray):
        if targets.ndim > 1:
            if targets.shape[-1] > 1:
                # One-hot encoded: [batch_size, num_classes] -> [batch_size]
                targets = np.argmax(targets, axis=-1)
            elif targets.shape[-1] == 1:
                # Class indices with extra dim: [batch_size, 1] -> [batch_size]
                targets = targets.squeeze(-1)
        else:
            # 1D array: check if it's one-hot or class indices
            if targets.shape[0] > 1 and np.sum(targets) == 1:
                # One-hot encoding in 1D: [num_classes] -> scalar, but expand to match batch size
                class_idx = np.argmax(targets)
                targets = np.full(predictions.shape[0], class_idx)
    
    # Convert logits to probabilities if needed - STANDARDIZED LOGIC
    if predictions.shape[1] > 1:
        # Multi-class classification: Apply softmax to get probabilities
        exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        pred_labels = np.argmax(probabilities, axis=1)
    elif predictions.shape[1] == 1:
        # Binary classification: Apply sigmoid
        probabilities = 1 / (1 + np.exp(-predictions))
        pred_labels = (probabilities > 0.5).astype(int)
    else:
        # Single value per sample: assume already probabilities
        probabilities = predictions
        pred_labels = (probabilities > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(targets, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, pred_labels, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        targets, pred_labels, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(targets, pred_labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'predictions': pred_labels,
        'probabilities': probabilities,
        'targets': targets
    }
    
    if class_names:
        metrics['class_names'] = class_names
    
    return metrics


def plot_classification_results(metrics, save_dir, prefix='classification'):
    """
    Plot classification results including confusion matrix and metrics.
    
    Args:
        metrics: Dictionary returned by compute_classification_metrics
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class names, with safety check for empty metrics
    precision_per_class = metrics.get('precision_per_class', [])
    class_names = metrics.get('class_names', [f'Class_{i}' for i in range(len(precision_per_class))])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ensure metrics and class names have matching lengths
    num_classes = len(class_names)
    # precision_per_class already extracted above for safety
    recall_per_class = metrics.get('recall_per_class', [])
    f1_per_class = metrics.get('f1_per_class', [])
    
    # Pad metrics to match number of classes if needed
    if len(precision_per_class) < num_classes:
        precision_per_class = list(precision_per_class) + [0.0] * (num_classes - len(precision_per_class))
    if len(recall_per_class) < num_classes:
        recall_per_class = list(recall_per_class) + [0.0] * (num_classes - len(recall_per_class))
    if len(f1_per_class) < num_classes:
        f1_per_class = list(f1_per_class) + [0.0] * (num_classes - len(f1_per_class))
    
    # Truncate if we have more metrics than classes (shouldn't happen but safety check)
    precision_per_class = precision_per_class[:num_classes]
    recall_per_class = recall_per_class[:num_classes]
    f1_per_class = f1_per_class[:num_classes]
    
    # Precision
    axes[0].bar(class_names, precision_per_class)
    axes[0].set_title('Precision per Class')
    axes[0].set_ylabel('Precision')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall
    axes[1].bar(class_names, recall_per_class)
    axes[1].set_title('Recall per Class')
    axes[1].set_ylabel('Recall')
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1 Score
    axes[2].bar(class_names, f1_per_class)
    axes[2].set_title('F1 Score per Class')
    axes[2].set_ylabel('F1 Score')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}_per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics summary
    with open(save_dir / f'{prefix}_metrics.txt', 'w') as f:
        f.write("Classification Metrics Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['precision']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['recall']:.4f}\n")
        f.write(f"Weighted F1 Score: {metrics['f1']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-" * 20 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            # Use the padded arrays that match class_names length
            prec = precision_per_class[i] if i < len(precision_per_class) else 0.0
            rec = recall_per_class[i] if i < len(recall_per_class) else 0.0
            f1 = f1_per_class[i] if i < len(f1_per_class) else 0.0
            f.write(f"  Precision: {prec:.4f}\n")
            f.write(f"  Recall: {rec:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n\n")


def validate_classification_outputs(classification_output, targets=None, class_names=None, save_dir=None, verbose=True):
    """
    Validate classification outputs and compute metrics.
    
    Args:
        classification_output: Model classification output tensor
        targets: Ground truth classification labels (optional)
        class_names: List of class names
        save_dir: Directory to save results
        verbose: Whether to print results
        
    Returns:
        dict: Classification metrics if targets provided, otherwise just predictions
    """
    if classification_output is None:
        if verbose:
            LOGGER.info("No classification output provided")
        return None
    
    # Get predictions
    cls_preds = torch.softmax(classification_output, dim=1)
    cls_pred_labels = torch.argmax(cls_preds, dim=1)
    
    if verbose:
        LOGGER.info(f"Classification outputs shape: {classification_output.shape}")
        LOGGER.info(f"Classification predictions: {cls_pred_labels}")
        LOGGER.info(f"Classification probabilities: {cls_preds.max(dim=1)[0]}")
    
    # If targets are provided, compute metrics
    if targets is not None:
        metrics = compute_classification_metrics(classification_output, targets, class_names)
        
        if verbose:
            LOGGER.info(f"Classification Accuracy: {metrics['accuracy']:.4f}")
            LOGGER.info(f"Classification F1 Score: {metrics['f1']:.4f}")
        
        # Save plots if save_dir is provided
        if save_dir is not None:
            plot_classification_results(metrics, save_dir)
        
        return metrics
    
    # Return just predictions if no targets
    return {
        'predictions': cls_pred_labels.cpu().numpy(),
        'probabilities': cls_preds.cpu().numpy()
    } 