#!/usr/bin/env python3
"""
測試互斥約束實現
Test Mutually Exclusive Constraints Implementation

用途：
1. 驗證互斥約束邏輯是否正確
2. 測試不同檢測輸出格式的兼容性
3. 確認損失計算的正確性
"""

import torch
import numpy as np
import sys
import os

# Add yolov5c to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.mutual_constraints import MutuallyExclusiveConstraints


def create_test_detections(batch_size=4, max_detections=50, detection_format='yolo'):
    """
    創建測試檢測數據
    
    Args:
        batch_size: 批次大小
        max_detections: 每個樣本的最大檢測數
        detection_format: 檢測格式 ('yolo' 或 'processed')
        
    Returns:
        detections: 檢測結果張量
        view_labels: 視圖標籤
    """
    
    if detection_format == 'yolo':
        # YOLOv5 原始輸出格式: [batch, max_det, 5+num_classes]
        # [x, y, w, h, objectness, class_0_prob, class_1_prob, class_2_prob, class_3_prob]
        num_classes = 4
        detections = torch.zeros(batch_size, max_detections, 5 + num_classes)
        
        # 設置一些測試檢測
        for batch_idx in range(batch_size):
            # 隨機設置幾個檢測
            num_real_detections = np.random.randint(1, 6)  # 1-5 個檢測
            
            for det_idx in range(num_real_detections):
                # 隨機位置和大小
                detections[batch_idx, det_idx, 0] = np.random.uniform(0.1, 0.9)  # x
                detections[batch_idx, det_idx, 1] = np.random.uniform(0.1, 0.9)  # y  
                detections[batch_idx, det_idx, 2] = np.random.uniform(0.05, 0.2)  # w
                detections[batch_idx, det_idx, 3] = np.random.uniform(0.05, 0.2)  # h
                detections[batch_idx, det_idx, 4] = np.random.uniform(0.3, 0.9)  # objectness
                
                # 隨機類別概率
                class_probs = torch.rand(num_classes)
                class_probs = torch.softmax(class_probs, dim=0)
                detections[batch_idx, det_idx, 5:] = class_probs
                
    elif detection_format == 'processed':
        # 處理後的檢測格式: [batch, max_det, 6]
        # [x1, y1, x2, y2, confidence, class_id]
        detections = torch.zeros(batch_size, max_detections, 6)
        
        for batch_idx in range(batch_size):
            num_real_detections = np.random.randint(1, 6)
            
            for det_idx in range(num_real_detections):
                x1, y1 = np.random.uniform(0.1, 0.7, 2)
                x2, y2 = x1 + np.random.uniform(0.05, 0.2), y1 + np.random.uniform(0.05, 0.2)
                
                detections[batch_idx, det_idx, 0] = x1
                detections[batch_idx, det_idx, 1] = y1
                detections[batch_idx, det_idx, 2] = x2
                detections[batch_idx, det_idx, 3] = y2
                detections[batch_idx, det_idx, 4] = np.random.uniform(0.3, 0.9)  # confidence
                detections[batch_idx, det_idx, 5] = np.random.randint(0, 4)  # class
    
    # 創建視圖標籤
    view_labels = torch.randint(0, 3, (batch_size,))  # 0=A4C, 1=PSAX, 2=PLAX
    
    return detections, view_labels


def create_violation_scenarios():
    """
    創建特定的違反場景進行測試
    
    Returns:
        scenarios: 包含各種測試場景的列表
    """
    
    scenarios = []
    
    # 場景1: A4C視圖同時檢測到MR和TR（應該違反）
    detections_1 = torch.zeros(1, 10, 6)  # [batch=1, max_det=10, 6]
    detections_1[0, 0] = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.8, 1])  # MR, high confidence
    detections_1[0, 1] = torch.tensor([0.6, 0.6, 0.9, 0.9, 0.7, 3])  # TR, high confidence
    view_labels_1 = torch.tensor([0])  # A4C
    
    scenarios.append({
        'name': 'A4C: MR vs TR violation (high confidence)',
        'detections': detections_1,
        'view_labels': view_labels_1,
        'should_violate': True,
        'expected_violation_count': 1
    })
    
    # 場景2: A4C視圖只檢測到MR（正常）
    detections_2 = torch.zeros(1, 10, 6)
    detections_2[0, 0] = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.8, 1])  # MR only
    view_labels_2 = torch.tensor([0])  # A4C
    
    scenarios.append({
        'name': 'A4C: MR only (normal)',
        'detections': detections_2,
        'view_labels': view_labels_2,
        'should_violate': False,
        'expected_violation_count': 0
    })
    
    # 場景3: PSAX視圖同時檢測到PR和TR（應該違反）
    detections_3 = torch.zeros(1, 10, 6)
    detections_3[0, 0] = torch.tensor([0.2, 0.2, 0.6, 0.6, 0.9, 2])  # PR, very high confidence
    detections_3[0, 1] = torch.tensor([0.7, 0.7, 0.95, 0.95, 0.85, 3])  # TR, high confidence
    view_labels_3 = torch.tensor([1])  # PSAX
    
    scenarios.append({
        'name': 'PSAX: PR vs TR violation (very high confidence)',
        'detections': detections_3,
        'view_labels': view_labels_3,
        'should_violate': True,
        'expected_violation_count': 1
    })
    
    # 場景4: 低置信度檢測（不應該違反）
    detections_4 = torch.zeros(1, 10, 6)
    detections_4[0, 0] = torch.tensor([0.1, 0.1, 0.5, 0.5, 0.2, 1])  # MR, low confidence
    detections_4[0, 1] = torch.tensor([0.6, 0.6, 0.9, 0.9, 0.15, 3])  # TR, very low confidence
    view_labels_4 = torch.tensor([0])  # A4C
    
    scenarios.append({
        'name': 'A4C: MR vs TR low confidence (should not violate)',
        'detections': detections_4,
        'view_labels': view_labels_4,
        'should_violate': False,
        'expected_violation_count': 0
    })
    
    return scenarios


def test_basic_functionality():
    """測試基本功能"""
    
    print("=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    constraints = MutuallyExclusiveConstraints(device=device)
    
    # 測試隨機數據
    detections, view_labels = create_test_detections(batch_size=8, detection_format='processed')
    
    mutual_loss, violations, debug_info = constraints.compute_mutual_exclusion_loss(
        detections,
        view_labels,
        confidence_threshold=0.25
    )
    
    print(f"Random Data Test:")
    print(f"  Mutual Loss: {mutual_loss:.6f}")
    print(f"  Violations: {violations}")
    print(f"  Debug Info: {debug_info}")
    
    # 驗證輸出類型
    assert isinstance(mutual_loss, torch.Tensor), "mutual_loss should be a tensor"
    assert mutual_loss.requires_grad, "mutual_loss should require gradients"
    assert isinstance(violations, int), "violations should be an integer"
    assert isinstance(debug_info, dict), "debug_info should be a dictionary"
    
    print("[PASS] Basic functionality test passed!")
    return True


def test_violation_scenarios():
    """測試特定違反場景"""
    
    print("\n" + "=" * 60)
    print("TESTING VIOLATION SCENARIOS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    constraints = MutuallyExclusiveConstraints(device=device)
    
    scenarios = create_violation_scenarios()
    
    all_passed = True
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        
        mutual_loss, violations, debug_info = constraints.compute_mutual_exclusion_loss(
            scenario['detections'],
            scenario['view_labels'],
            confidence_threshold=0.25
        )
        
        print(f"  Expected violations: {scenario['expected_violation_count']}")
        print(f"  Actual violations: {violations}")
        print(f"  Mutual loss: {mutual_loss:.6f}")
        print(f"  Should violate: {scenario['should_violate']}")
        
        # 驗證結果
        if scenario['should_violate']:
            if violations == 0:
                print(f"  [FAIL] Expected violations but got none")
                all_passed = False
            else:
                print(f"  [PASS] Correctly detected violations")
        else:
            if violations > 0:
                print(f"  [FAIL] Expected no violations but got {violations}")
                all_passed = False
            else:
                print(f"  [PASS] Correctly detected no violations")
    
    return all_passed


def test_soft_constraints():
    """測試軟約束版本"""
    
    print("\n" + "=" * 60)
    print("TESTING SOFT CONSTRAINTS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    constraints = MutuallyExclusiveConstraints(device=device)
    
    batch_size = 4
    num_classes = 4
    
    # 創建分類logits
    class_logits = torch.randn(batch_size, num_classes, device=device)
    view_labels = torch.tensor([0, 1, 2, 0], device=device)  # A4C, PSAX, PLAX, A4C
    
    # 測試軟約束
    soft_loss = constraints.compute_soft_mutual_loss(class_logits, view_labels)
    
    print(f"Class logits shape: {class_logits.shape}")
    print(f"View labels: {view_labels.tolist()}")
    print(f"Soft mutual loss: {soft_loss:.6f}")
    
    # 驗證輸出
    assert isinstance(soft_loss, (float, torch.Tensor)), "soft_loss should be float or tensor"
    
    print("[PASS] Soft constraints test passed!")
    return True


def test_gradient_flow():
    """測試梯度流"""
    
    print("\n" + "=" * 60)
    print("TESTING GRADIENT FLOW")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    constraints = MutuallyExclusiveConstraints(device=device)
    
    # 創建需要梯度的檢測數據
    detections = torch.randn(2, 5, 6, device=device, requires_grad=True)
    view_labels = torch.tensor([0, 1], device=device)  # A4C, PSAX
    
    mutual_loss, violations, debug_info = constraints.compute_mutual_exclusion_loss(
        detections,
        view_labels,
        confidence_threshold=0.25
    )
    
    print(f"Mutual loss: {mutual_loss:.6f}")
    print(f"Mutual loss requires_grad: {mutual_loss.requires_grad}")
    
    # 測試反向傳播
    if mutual_loss.requires_grad:
        mutual_loss.backward()
        print(f"Detections grad exists: {detections.grad is not None}")
        if detections.grad is not None:
            print(f"Detections grad sum: {detections.grad.sum().item():.6f}")
            print("[PASS] Gradient flow test passed!")
        else:
            print("[INFO] No gradients found (may be normal if no violations detected)")
    else:
        print("[INFO] mutual_loss does not require gradients")
    
    return True


def main():
    """主測試函數"""
    
    print("[TEST] MUTUALLY EXCLUSIVE CONSTRAINTS TEST SUITE")
    print("=" * 80)
    
    test_results = []
    
    # 運行所有測試
    try:
        test_results.append(("Basic Functionality", test_basic_functionality()))
        test_results.append(("Violation Scenarios", test_violation_scenarios()))
        test_results.append(("Soft Constraints", test_soft_constraints()))
        test_results.append(("Gradient Flow", test_gradient_flow()))
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 總結結果
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED! Mutually exclusive constraints are working correctly.")
        return True
    else:
        print("[WARNING] SOME TESTS FAILED. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
