"""
Analyze confusion matrix to see where PSAX (class 1) samples are being mispredicted

From validation output:
- PSAX has 33 samples in validation set
- Only 3 are correctly predicted (9% recall)
- Where are the other 30 PSAX samples being predicted?
"""

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_debug_output():
    """Parse the debug output from validation"""
    
    # From your debug output:
    pred_classes = [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 2, 0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 2, 2, 1, 1, 1, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2]
    
    true_classes = [0, 0, 1, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 2, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2]
    
    return np.array(pred_classes), np.array(true_classes)

def analyze_confusion_matrix(pred, true):
    """Analyze confusion matrix for PSAX"""
    print("=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    # Create confusion matrix
    cm = confusion_matrix(true, pred)
    
    class_names = ['A4C', 'PSAX', 'PLAX']
    
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              A4C  PSAX  PLAX")
    for i, name in enumerate(class_names):
        print(f"True {name:4s}:  {cm[i, 0]:3d}  {cm[i, 1]:3d}  {cm[i, 2]:3d}")
    
    # Analyze PSAX (class 1) specifically
    print("\n" + "=" * 60)
    print("PSAX (CLASS 1) DETAILED ANALYSIS")
    print("=" * 60)
    
    psax_total = cm[1, :].sum()
    psax_correct = cm[1, 1]
    psax_as_a4c = cm[1, 0]
    psax_as_plax = cm[1, 2]
    
    print(f"\nTotal PSAX samples: {psax_total}")
    print(f"Correctly predicted as PSAX: {psax_correct} ({psax_correct/psax_total*100:.1f}%)")
    print(f"Mispredicted as A4C: {psax_as_a4c} ({psax_as_a4c/psax_total*100:.1f}%)")
    print(f"Mispredicted as PLAX: {psax_as_plax} ({psax_as_plax/psax_total*100:.1f}%)")
    
    if psax_as_a4c > psax_correct:
        print(f"\nCRITICAL: PSAX is being predicted as A4C {psax_as_a4c/psax_correct:.1f}x more often than correctly!")
        print("  This suggests a systematic bias toward A4C for PSAX samples")
    
    if psax_as_plax > psax_correct:
        print(f"\nCRITICAL: PSAX is being predicted as PLAX {psax_as_plax/psax_correct:.1f}x more often than correctly!")
        print("  This suggests a systematic bias toward PLAX for PSAX samples")
    
    # Check if there's a pattern
    print("\n" + "=" * 60)
    print("PATTERN ANALYSIS")
    print("=" * 60)
    
    total_predictions = len(pred)
    a4c_predictions = np.sum(pred == 0)
    psax_predictions = np.sum(pred == 1)
    plax_predictions = np.sum(pred == 2)
    
    print(f"\nOverall prediction distribution:")
    print(f"  Predicted A4C:  {a4c_predictions} ({a4c_predictions/total_predictions*100:.1f}%)")
    print(f"  Predicted PSAX: {psax_predictions} ({psax_predictions/total_predictions*100:.1f}%)")
    print(f"  Predicted PLAX: {plax_predictions} ({plax_predictions/total_predictions*100:.1f}%)")
    
    print(f"\nExpected distribution (from true labels):")
    a4c_true = np.sum(true == 0)
    psax_true = np.sum(true == 1)
    plax_true = np.sum(true == 2)
    print(f"  True A4C:  {a4c_true} ({a4c_true/total_predictions*100:.1f}%)")
    print(f"  True PSAX: {psax_true} ({psax_true/total_predictions*100:.1f}%)")
    print(f"  True PLAX: {plax_true} ({plax_true/total_predictions*100:.1f}%)")
    
    if psax_predictions < psax_true * 0.5:
        print(f"\nCRITICAL BUG: Model predicts PSAX {psax_predictions} times but should predict ~{psax_true} times")
        print("  The model is SEVERELY underpredictiing PSAX!")
        print("  This suggests:")
        print("    - Model initialization bias against class 1")
        print("    - Or loss function not properly updating class 1 weights")
        print("    - Or gradient flow issue for class 1")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - PSAX Bug Analysis')
    plt.tight_layout()
    plt.savefig('psax_confusion_matrix.png', dpi=150)
    print(f"\nConfusion matrix saved to: psax_confusion_matrix.png")

if __name__ == "__main__":
    pred, true = parse_debug_output()
    analyze_confusion_matrix(pred, true)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Examine confusion matrix image")
    print("2. Check where PSAX samples are being mispredicted")
    print("3. Investigate code bug in that specific prediction path")

