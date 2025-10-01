#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to extract model outputs from your trained YOLOv5WithClassification model
for manual cross-entropy calculation
"""

import torch
import sys
from pathlib import Path

# Add yolov5c to path
sys.path.append('yolov5c')

from extract_model_outputs import extract_and_analyze, save_extracted_data, parse_model_output, manual_cross_entropy_calculation

def extract_from_trained_model():
    """
    Extract model outputs from your trained YOLOv5WithClassification model
    """
    print("🚀 Extracting outputs from trained YOLOv5WithClassification model...")
    
    # Configuration - UPDATE THESE PATHS FOR YOUR SETUP
    model_path = "yolov5c/runs/classifybackbone13/weights/last.pt"  # Your actual trained model
    data_yaml = "regurgitationV1/data.yaml"  # Your actual dataset
    class_names = ['A4C', 'PSAX', 'PLAX']  # From your data.yaml
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Import required modules
        from yolov5c.models.experimental import attempt_load
        from yolov5c.utils.dataloaders import create_dataloader
        from yolov5c.utils.general import check_dataset
        
        print(f"📂 Loading model from: {model_path}")
        print(f"📂 Loading dataset from: {data_yaml}")
        print(f"🖥️  Using device: {device}")
        
        # 1. Load your trained model
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            print("   Available model files:")
            runs_dir = Path("runs/train")
            if runs_dir.exists():
                for exp_dir in runs_dir.iterdir():
                    weights_dir = exp_dir / "weights"
                    if weights_dir.exists():
                        for weight_file in weights_dir.glob("*.pt"):
                            print(f"     {weight_file}")
            return None
        
        model = attempt_load(model_path, device=device)
        print(f"✅ Model loaded successfully")
        
        # 2. Load your dataset
        if not Path(data_yaml).exists():
            print(f"❌ Dataset file not found: {data_yaml}")
            print("   Available dataset files:")
            for yaml_file in Path(".").glob("**/*.yaml"):
                print(f"     {yaml_file}")
            return None
        
        data_dict = check_dataset(data_yaml)
        train_path = data_dict['train']
        print(f"✅ Dataset loaded: {train_path}")
        
        # 3. Create dataloader
        print("📊 Creating dataloader...")
        train_loader, dataset = create_dataloader(
            train_path, 
            imgsz=640, 
            batch_size=8,  # Smaller batch for analysis
            gs=32, 
            single_cls=False, 
            hyp={'cls_task': 0.3, 'label_smoothing': 0.1}, 
            augment=False,  # Disable for consistent results
            cache=None, 
            rect=False, 
            rank=-1, 
            workers=4
        )
        print(f"✅ Dataloader created with {len(train_loader)} batches")
        
        # 4. Extract and analyze (first 5 batches for quick test)
        print("🔍 Extracting model outputs and labels...")
        extracted_results, analysis_results = extract_and_analyze(
            model, 
            train_loader, 
            device=device, 
            max_batches=5,  # Analyze first 5 batches
            class_names=class_names
        )
        
        # 5. Save for later analysis
        save_path = "trained_model_analysis.pt"
        save_extracted_data(extracted_results, save_path)
        print(f"💾 Results saved to: {save_path}")
        
        # 6. Show detailed results
        print("\n📊 Detailed Analysis Results:")
        print("=" * 50)
        
        total_manual_loss = 0.0
        total_samples = 0
        
        for i, (batch_result, analysis) in enumerate(zip(extracted_results, analysis_results)):
            print(f"\n🔍 Batch {i}:")
            
            # Get model outputs
            model_output = batch_result['model_output']
            detection_outputs, classification_output = parse_model_output(model_output)
            
            # Get labels
            classification_labels = batch_result['classification_labels']
            
            if classification_output is not None:
                # Calculate manual cross-entropy
                manual_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
                
                print(f"   📊 Classification shape: {classification_output.shape}")
                print(f"   📊 Labels shape: {classification_labels.shape}")
                print(f"   📊 Manual Cross-Entropy Loss: {manual_loss.item():.6f}")
                print(f"   📊 Accuracy: {analysis['accuracy']:.4f}")
                
                # Show sample predictions
                pred_classes = torch.argmax(classification_output, dim=1)
                pred_probs = torch.softmax(classification_output, dim=1)
                
                print(f"   📋 Sample predictions:")
                for j in range(min(3, classification_output.shape[0])):
                    pred_class = pred_classes[j].item()
                    target_class = classification_labels[j].item()
                    confidence = pred_probs[j, pred_class].item()
                    
                    pred_name = class_names[pred_class] if pred_class < len(class_names) else f'Class_{pred_class}'
                    target_name = class_names[target_class] if target_class < len(class_names) else f'Class_{target_class}'
                    
                    status = "✅" if pred_class == target_class else "❌"
                    print(f"     Sample {j}: {pred_name} vs {target_name} (conf: {confidence:.3f}) {status}")
                
                total_manual_loss += manual_loss.item() * classification_output.shape[0]
                total_samples += classification_output.shape[0]
        
        # Overall statistics
        if total_samples > 0:
            avg_manual_loss = total_manual_loss / total_samples
            print(f"\n📈 Overall Statistics:")
            print(f"   Total samples analyzed: {total_samples}")
            print(f"   Average manual cross-entropy loss: {avg_manual_loss:.6f}")
        
        return extracted_results, analysis_results
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_pytorch_loss():
    """
    Compare manual cross-entropy with PyTorch's implementation
    """
    print("\n🧪 Comparing manual vs PyTorch cross-entropy...")
    
    # Load saved results
    save_path = "trained_model_analysis.pt"
    if not Path(save_path).exists():
        print(f"❌ No saved results found at {save_path}")
        print("   Run extract_from_trained_model() first")
        return
    
    try:
        extracted_data = torch.load(save_path, map_location='cpu')
        
        total_manual_loss = 0.0
        total_pytorch_loss = 0.0
        total_samples = 0
        
        for i, batch_data in enumerate(extracted_data):
            # Parse model output
            model_output = batch_data['model_output']
            if isinstance(model_output, tuple) and len(model_output) == 2:
                _, classification_output = model_output
            else:
                continue
            
            if classification_output is None:
                continue
            
            classification_labels = batch_data['classification_labels']
            
            # Manual calculation
            manual_loss = manual_cross_entropy_calculation(classification_output, classification_labels)
            
            # PyTorch calculation
            pytorch_loss = torch.nn.functional.cross_entropy(classification_output, classification_labels)
            
            batch_size = classification_output.shape[0]
            total_manual_loss += manual_loss.item() * batch_size
            total_pytorch_loss += pytorch_loss.item() * batch_size
            total_samples += batch_size
            
            print(f"   Batch {i}: Manual={manual_loss.item():.6f}, PyTorch={pytorch_loss.item():.6f}, Diff={abs(manual_loss.item() - pytorch_loss.item()):.8f}")
        
        if total_samples > 0:
            avg_manual = total_manual_loss / total_samples
            avg_pytorch = total_pytorch_loss / total_samples
            print(f"\n📊 Average Comparison:")
            print(f"   Manual Cross-Entropy: {avg_manual:.6f}")
            print(f"   PyTorch Cross-Entropy: {avg_pytorch:.6f}")
            print(f"   Difference: {abs(avg_manual - avg_pytorch):.8f}")
            print(f"   ✅ {'Perfect match!' if abs(avg_manual - avg_pytorch) < 1e-6 else 'Small difference detected'}")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")

def main():
    """
    Main function to run the extraction
    """
    print("🎯 YOLOv5WithClassification Model Output Extraction")
    print("=" * 60)
    
    # Extract from trained model
    results = extract_from_trained_model()
    
    if results is not None:
        # Compare with PyTorch implementation
        compare_with_pytorch_loss()
        
        print("\n✅ Extraction completed successfully!")
        print("\n📋 What you can do next:")
        print("   1. Check the saved file: trained_model_analysis.pt")
        print("   2. Analyze the manual cross-entropy calculations")
        print("   3. Compare different batches and epochs")
        print("   4. Use the extracted data for further analysis")
    else:
        print("\n❌ Extraction failed. Please check the paths and try again.")

if __name__ == "__main__":
    main()
