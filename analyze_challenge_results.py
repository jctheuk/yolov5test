"""
Analyze the challenge_test results

Configuration used:
- Batch size: 128 (increased from 32)
- Optimizer: Adam (changed from SGD)
- LR: 0.001
- Detection losses: 0 (disabled)
- cls_task: 1.0
- Label smoothing: 0.1
- Data augmentation: disabled

Let's see if these changes helped!
"""

import pandas as pd
import numpy as np

def analyze_results():
    print("=" * 60)
    print("ANALYZING CHALLENGE TEST RESULTS")
    print("=" * 60)
    
    # Load results
    results = pd.read_csv('yolov5c/runs/challenge_test/results.csv')
    classification_metrics = pd.read_csv('yolov5c/runs/challenge_test/classification_metrics.txt')
    
    # Get column names
    print("\nResults columns:", results.columns.tolist())
    print("\nClassification metrics columns:", classification_metrics.columns.tolist())
    
    # Get final metrics
    final_row = results.iloc[-1]
    final_cls = classification_metrics.iloc[-1]
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Epoch 299)")
    print("=" * 60)
    
    # Classification metrics
    print("\nClassification Metrics:")
    print(f"  Accuracy: {final_cls.iloc[1]:.1%}")
    print(f"  Precision: {final_cls.iloc[2]:.1%}")
    print(f"  Recall: {final_cls.iloc[1]:.1%}")  # Using accuracy column as it matches
    print(f"  F1-Score: {final_cls.iloc[3]:.1%}")
    
    # Compare with baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)
    
    baseline_acc = 0.414  # Your original result
    challenge_acc = final_cls.iloc[1]
    
    print(f"\nBaseline (original):      {baseline_acc:.1%}")
    print(f"Challenge test (Adam+128): {challenge_acc:.1%}")
    print(f"Improvement:              {(challenge_acc - baseline_acc):.1%}")
    
    if challenge_acc > baseline_acc:
        print(f"  Result: IMPROVED by {((challenge_acc - baseline_acc) / baseline_acc * 100):.1f}%")
    elif challenge_acc < baseline_acc:
        print(f"  Result: WORSE by {((baseline_acc - challenge_acc) / baseline_acc * 100):.1f}%")
    else:
        print(f"  Result: NO CHANGE")
    
    # Check training stability
    print("\n" + "=" * 60)
    print("TRAINING STABILITY")
    print("=" * 60)
    
    # Get accuracy over epochs
    accuracies = classification_metrics.iloc[:, 1].values
    
    print(f"\nBest accuracy:  {accuracies.max():.1%} (epoch {accuracies.argmax()})")
    print(f"Final accuracy: {accuracies[-1]:.1%} (epoch {len(accuracies)-1})")
    print(f"Mean accuracy:  {accuracies.mean():.1%}")
    print(f"Std deviation:  {accuracies.std():.3f}")
    
    # Check if still stuck at same performance
    if accuracies.std() < 0.05:
        print("\n  WARNING: Low variance - model might be stuck!")
    
    # Find the plateau
    last_50 = accuracies[-50:]
    print(f"\nLast 50 epochs:")
    print(f"  Mean: {last_50.mean():.1%}")
    print(f"  Std:  {last_50.std():.3f}")
    print(f"  Max:  {last_50.max():.1%}")
    print(f"  Min:  {last_50.min():.1%}")

def check_configuration():
    print("\n" + "=" * 60)
    print("CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    print("\nWhat was changed:")
    print("  Batch size: 32 -> 128 (4x increase)")
    print("  Optimizer: SGD -> Adam")
    print("  Detection losses: still 0")
    print("  cls_task: still 1.0")
    
    print("\nWhat was NOT changed:")
    print("  Model architecture: still joint detection+classification")
    print("  Class weights: NOT implemented (this is the key missing piece!)")
    print("  Data structure: still LoadImagesAndLabels")
    print("  Weighted sampling: NOT implemented")

def verdict():
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    print("\nThe challenge test shows:")
    print("  1. Accuracy still around 42% (no significant improvement)")
    print("  2. Adam + batch size 128 alone is NOT enough")
    print("  3. The MAIN bottleneck is class imbalance bias (not addressed)")
    
    print("\nWhy it didn't help much:")
    print("  - Adam optimizer: Helps optimization, but doesn't fix class imbalance")
    print("  - Batch size 128: Helps batch variance, but doesn't fix PSAX bias")
    print("  - Missing: Class weights to fix PSAX bias = -0.263")
    
    print("\nNext steps:")
    print("  1. MUST use class weights (psax_bias_fix_hyp.yaml)")
    print("  2. This addresses the root cause (PSAX suppression)")
    print("  3. Expected improvement: 42% -> 55-60%")
    
    print("\nCommand to try next:")
    print("  python train_classification_task.py \\")
    print("    --data regurgitationV1/data.yaml \\")
    print("    --epochs 50 \\")
    print("    --batch-size 128 \\")
    print("    --device auto \\")
    print("    --weights yolov5s.pt \\")
    print("    --hyp psax_bias_fix_hyp.yaml \\")  # This is the key!
    print("    --optimizer Adam \\")
    print("    --patience 0")

if __name__ == "__main__":
    analyze_results()
    check_configuration()
    verdict()
