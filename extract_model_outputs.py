#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to extract model outputs and labels equivalent to yolov5original model(images), labels
for manual cross-entropy calculation
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add yolov5c to path
sys.path.append('yolov5c')

def extract_model_outputs_and_labels(model, dataloader, device='cuda', max_batches=None):
    """
    Extract model outputs and labels equivalent to yolov5original model(images), labels
    
    Args:
        model: YOLOv5 model (trained or untrained)
        dataloader: DataLoader with images and labels
        device: Device to run inference on
        max_batches: Maximum number of batches to process (None for all)
    
    Returns:
        list: List of tuples (model_outputs, labels) for each batch
              - model_outputs: Model predictions (detection + classification)
              - labels: Ground truth labels (detection + classification)
    """
    model.eval()
    results = []
    
    print(f"🔍 Extracting model outputs and labels...")
    print(f"   Device: {device}")
    print(f"   Max batches: {max_batches if max_batches else 'All'}")
    
    with torch.no_grad():
        for batch_idx, (images, targets, paths, shapes, classification_labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move to device
            images = images.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            
            # Process classification labels
            if classification_labels is not None:
                classification_labels = classification_labels.to(device)
                # Handle different label formats
                if classification_labels.dim() > 1:
                    if classification_labels.shape[-1] > 1:
                        # One-hot encoded: [batch_size, num_classes] -> [batch_size]
                        classification_labels = classification_labels.argmax(dim=-1)
                    elif classification_labels.shape[-1] == 1:
                        # Class indices with extra dim: [batch_size, 1] -> [batch_size]
                        classification_labels = classification_labels.squeeze(-1)
                
                # Ensure labels are long tensors
                if classification_labels.dtype != torch.long:
                    classification_labels = classification_labels.long()
            else:
                # Create default classification labels if none provided
                classification_labels = torch.zeros(images.shape[0], dtype=torch.long, device=device)
            
            # Forward pass - equivalent to yolov5original model(images)
            model_output = model(images)
            
            # Store results
            batch_result = {
                'batch_idx': batch_idx,
                'images': images,
                'model_output': model_output,
                'detection_targets': targets,
                'classification_labels': classification_labels,
                'paths': paths,
                'shapes': shapes
            }
            
            results.append(batch_result)
            
            if batch_idx % 10 == 0:
                print(f"   Processed batch {batch_idx}")
    
    print(f"✅ Extracted {len(results)} batches")
    return results

def parse_model_output(model_output):
    """
    Parse model output to separate detection and classification outputs
    
    Args:
        model_output: Raw model output (tuple or tensor)
    
    Returns:
        tuple: (detection_outputs, classification_output)
    """
    if isinstance(model_output, tuple) and len(model_output) == 2:
        detection_outputs, classification_output = model_output
        return detection_outputs, classification_output
    elif isinstance(model_output, tuple) and len(model_output) == 1:
        # Only detection output
        return model_output[0], None
    else:
        # Single tensor output
        return model_output, None

def manual_cross_entropy_calculation(classification_output, labels):
    """
    Manual cross-entropy calculation equivalent to PyTorch's CrossEntropyLoss
    
    Args:
        classification_output: Model classification logits [batch_size, num_classes]
        labels: Ground truth class indices [batch_size]
    
    Returns:
        torch.Tensor: Cross-entropy loss value
    """
    if classification_output is None:
        return torch.tensor(0.0, device=labels.device)
    
    # Ensure labels are on the same device
    labels = labels.to(classification_output.device)
    
    # Compute log softmax
    log_probs = F.log_softmax(classification_output, dim=1)
    
    # Gather the log probabilities for the target classes
    batch_size = classification_output.shape[0]
    target_log_probs = log_probs[range(batch_size), labels]
    
    # Return negative log likelihood (CrossEntropy loss)
    return -target_log_probs.mean()

def analyze_batch_outputs(batch_result, class_names=None):
    """
    Analyze a single batch's model outputs and labels
    
    Args:
        batch_result: Result from extract_model_outputs_and_labels
        class_names: List of class names for display
    
    Returns:
        dict: Analysis results
    """
    batch_idx = batch_result['batch_idx']
    model_output = batch_result['model_output']
    classification_labels = batch_result['classification_labels']
    
    # Parse model output
    detection_outputs, classification_output = parse_model_output(model_output)
    
    analysis = {
        'batch_idx': batch_idx,
        'num_samples': classification_labels.shape[0],
        'detection_output_available': detection_outputs is not None,
        'classification_output_available': classification_output is not None,
    }
    
    if classification_output is not None:
        # Get predictions
        pred_classes = torch.argmax(classification_output, dim=1)
        pred_probs = torch.softmax(classification_output, dim=1)
        
        # Calculate manual cross-entropy
        manual_ce_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
        
        # Calculate accuracy
        correct = (pred_classes == classification_labels).sum().item()
        accuracy = correct / classification_labels.shape[0]
        
        analysis.update({
            'classification_shape': classification_output.shape,
            'predicted_classes': pred_classes.cpu().numpy(),
            'ground_truth_classes': classification_labels.cpu().numpy(),
            'prediction_probabilities': pred_probs.max(dim=1)[0].cpu().numpy(),
            'manual_cross_entropy_loss': manual_ce_loss.item(),
            'accuracy': accuracy,
            'correct_predictions': correct,
        })
        
        # Class-wise analysis
        if class_names:
            unique_preds, pred_counts = torch.unique(pred_classes, return_counts=True)
            unique_targets, target_counts = torch.unique(classification_labels, return_counts=True)
            
            pred_class_names = [class_names[i] if i < len(class_names) else f'Class_{i}' 
                               for i in unique_preds.cpu().numpy()]
            target_class_names = [class_names[i] if i < len(class_names) else f'Class_{i}' 
                                 for i in unique_targets.cpu().numpy()]
            
            analysis.update({
                'predicted_class_names': pred_class_names,
                'predicted_counts': pred_counts.cpu().numpy().tolist(),
                'target_class_names': target_class_names,
                'target_counts': target_counts.cpu().numpy().tolist(),
            })
    
    return analysis

def print_batch_analysis(analysis, class_names=None):
    """
    Print detailed analysis of a batch
    
    Args:
        analysis: Analysis result from analyze_batch_outputs
        class_names: List of class names for display
    """
    batch_idx = analysis['batch_idx']
    print(f"\n📊 Batch {batch_idx} Analysis:")
    print(f"   Samples: {analysis['num_samples']}")
    print(f"   Detection output available: {analysis['detection_output_available']}")
    print(f"   Classification output available: {analysis['classification_output_available']}")
    
    if analysis['classification_output_available']:
        print(f"   Classification shape: {analysis['classification_shape']}")
        print(f"   Manual Cross-Entropy Loss: {analysis['manual_cross_entropy_loss']:.4f}")
        print(f"   Accuracy: {analysis['accuracy']:.4f}")
        print(f"   Correct predictions: {analysis['correct_predictions']}/{analysis['num_samples']}")
        
        # Show first few predictions
        print(f"   First 5 predictions:")
        for i in range(min(5, analysis['num_samples'])):
            pred_class = analysis['predicted_classes'][i]
            target_class = analysis['ground_truth_classes'][i]
            confidence = analysis['prediction_probabilities'][i]
            
            pred_name = class_names[pred_class] if class_names and pred_class < len(class_names) else f'Class_{pred_class}'
            target_name = class_names[target_class] if class_names and target_class < len(class_names) else f'Class_{target_class}'
            
            status = "✓" if pred_class == target_class else "✗"
            print(f"     Sample {i}: {pred_name} vs {target_name} (conf: {confidence:.3f}) {status}")
        
        # Class distribution
        if 'predicted_class_names' in analysis:
            print(f"   Predicted classes: {analysis['predicted_class_names']}")
            print(f"   Predicted counts: {analysis['predicted_counts']}")
            print(f"   Target classes: {analysis['target_class_names']}")
            print(f"   Target counts: {analysis['target_counts']}")

def extract_and_analyze(model, dataloader, device='cuda', max_batches=5, class_names=None):
    """
    Complete extraction and analysis pipeline
    
    Args:
        model: YOLOv5 model
        dataloader: DataLoader
        device: Device to run on
        max_batches: Maximum batches to analyze
        class_names: List of class names
    
    Returns:
        tuple: (extracted_results, analysis_results)
    """
    print("🚀 Starting model output extraction and analysis...")
    
    # Extract model outputs
    extracted_results = extract_model_outputs_and_labels(model, dataloader, device, max_batches)
    
    # Analyze each batch
    analysis_results = []
    for batch_result in extracted_results:
        analysis = analyze_batch_outputs(batch_result, class_names)
        analysis_results.append(analysis)
        print_batch_analysis(analysis, class_names)
    
    # Summary statistics
    if analysis_results:
        total_samples = sum(a['num_samples'] for a in analysis_results)
        total_correct = sum(a.get('correct_predictions', 0) for a in analysis_results)
        avg_loss = np.mean([a.get('manual_cross_entropy_loss', 0) for a in analysis_results])
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Total batches analyzed: {len(analysis_results)}")
        print(f"   Total samples: {total_samples}")
        print(f"   Overall accuracy: {total_correct/total_samples:.4f}")
        print(f"   Average cross-entropy loss: {avg_loss:.4f}")
    
    return extracted_results, analysis_results

def save_extracted_data(extracted_results, save_path="extracted_model_outputs.pt"):
    """
    Save extracted data to file for later analysis
    
    Args:
        extracted_results: Results from extract_model_outputs_and_labels
        save_path: Path to save the data
    """
    # Convert to CPU tensors for saving
    save_data = []
    for batch_result in extracted_results:
        save_batch = {
            'batch_idx': batch_result['batch_idx'],
            'images': batch_result['images'].cpu(),
            'model_output': batch_result['model_output'],
            'detection_targets': batch_result['detection_targets'].cpu(),
            'classification_labels': batch_result['classification_labels'].cpu(),
            'paths': batch_result['paths'],
            'shapes': batch_result['shapes']
        }
        save_data.append(save_batch)
    
    torch.save(save_data, save_path)
    print(f"💾 Saved extracted data to {save_path}")

def load_extracted_data(load_path="extracted_model_outputs.pt"):
    """
    Load previously extracted data
    
    Args:
        load_path: Path to load the data from
    
    Returns:
        list: Loaded extracted results
    """
    data = torch.load(load_path, map_location='cpu')
    print(f"📂 Loaded extracted data from {load_path}")
    return data

# Example usage function
def example_usage():
    """
    Example of how to use the extraction functions
    """
    print("📖 Example Usage:")
    print("""
    # Load your model and dataloader
    from yolov5c.models.experimental import attempt_load
    from yolov5c.utils.dataloaders import create_dataloader
    
    # Load model
    model = attempt_load('path/to/your/model.pt', device='cuda')
    
    # Create dataloader
    train_loader, dataset = create_dataloader(
        train_path, imgsz, batch_size, gs, single_cls, hyp=hyp, 
        augment=True, cache=None, rect=opt.rect, rank=-1, workers=8
    )
    
    # Extract and analyze
    class_names = ['A4C', 'PSAX', 'PLAX']  # Your class names
    extracted_results, analysis_results = extract_and_analyze(
        model, train_loader, device='cuda', max_batches=5, class_names=class_names
    )
    
    # Save for later analysis
    save_extracted_data(extracted_results, "my_model_outputs.pt")
    
    # Load later
    loaded_data = load_extracted_data("my_model_outputs.pt")
    """)

if __name__ == "__main__":
    example_usage()

