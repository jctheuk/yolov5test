"""
互斥事件約束實現
Mutually Exclusive Constraints for Medical Image Detection

基於醫學解剖學規則，避免在同一視圖中同時檢測到互斥的反流類型
"""

import torch
import torch.nn as nn


class MutuallyExclusiveConstraints:
    """
    互斥事件約束 - 惩罰同時檢測到互斥的反流類型
    
    醫學背景：
    - A4C 視圖：MR (二尖瓣) vs TR (三尖瓣) 互斥
    - PSAX 視圖：PR (肺動脈瓣) vs TR (三尖瓣) 互斥  
    - PLAX 視圖：AR (主動脈瓣) vs MR (二尖瓣) 互斥
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 定義互斥關係：每個視圖中哪些檢測類別是互斥的
        # 格式：{view_index: [(class_a, class_b), ...]}
        self.mutually_exclusive_pairs = {
            0: [(1, 3)],  # A4C: MR (class 1) vs TR (class 3) 互斥
            1: [(2, 3)],  # PSAX: PR (class 2) vs TR (class 3) 互斥
            2: [(0, 1)],  # PLAX: AR (class 0) vs MR (class 1) 互斥
        }
        
        # 類別和視圖名稱（用於調試）
        self.class_names = ['AR', 'MR', 'PR', 'TR']
        self.view_names = ['A4C', 'PSAX', 'PLAX']
        
        print(f"[INFO] Mutually Exclusive Constraints initialized")
        print(f"[INFO] Exclusive pairs: {self.format_pairs()}")
    
    def format_pairs(self):
        """格式化互斥對為可讀字符串"""
        formatted = {}
        for view_idx, pairs in self.mutually_exclusive_pairs.items():
            view_name = self.view_names[view_idx]
            pair_names = []
            for class_a, class_b in pairs:
                pair_names.append(f"{self.class_names[class_a]} vs {self.class_names[class_b]}")
            formatted[view_name] = pair_names
        return formatted
    
    def compute_mutual_exclusion_loss(
        self, 
        detection_predictions,
        view_labels,
        confidence_threshold=0.25
    ):
        """
        計算互斥約束損失
        
        Args:
            detection_predictions: 檢測預測
                - 可以是 YOLOv5 輸出 [batch_size, num_anchors, 5+num_classes]
                - 或處理後的檢測結果 [batch_size, max_det, 6] (x1,y1,x2,y2,conf,cls)
            view_labels: 視圖標籤 [batch_size] (0=A4C, 1=PSAX, 2=PLAX)
            confidence_threshold: 置信度閾值
            
        Returns:
            mutual_loss: 互斥約束損失 (標量 tensor)
            violation_count: 違反次數 (int)
            debug_info: 調試信息 (dict)
        """
        
        total_penalty = 0.0
        violation_count = 0
        debug_info = {
            'violations_per_view': {view: 0 for view in self.view_names},
            'max_violation': 0.0,
            'total_samples': view_labels.shape[0]
        }
        
        batch_size = view_labels.shape[0]
        
        for batch_idx in range(batch_size):
            view_idx = view_labels[batch_idx].item()
            
            # 跳過未定義互斥關係的視圖
            if view_idx not in self.mutually_exclusive_pairs:
                continue
            
            exclusive_pairs = self.mutually_exclusive_pairs[view_idx]
            view_name = self.view_names[view_idx]
            
            # 對每個互斥對計算惩罰
            for class_a, class_b in exclusive_pairs:
                # 獲取每個類別的最大置信度
                conf_a = self._get_max_confidence_for_class(
                    detection_predictions[batch_idx], 
                    class_a, 
                    confidence_threshold
                )
                conf_b = self._get_max_confidence_for_class(
                    detection_predictions[batch_idx], 
                    class_b, 
                    confidence_threshold
                )
                
                # 互斥惩罰：E_mutual = P(A) × P(B)
                # 理想情況：一個為 1，另一個為 0，乘積為 0
                # 違反情況：兩者都 > 0，乘積 > 0
                mutual_penalty = conf_a * conf_b
                
                # 只有當違反程度足夠大時才计算损失
                if mutual_penalty > 0.01:
                    total_penalty += mutual_penalty
                    violation_count += 1
                    debug_info['violations_per_view'][view_name] += 1
                    debug_info['max_violation'] = max(debug_info['max_violation'], mutual_penalty)
                    
                    # 顯著違反時輸出調試信息
                    if mutual_penalty > 0.1:
                        print(f"[CONSTRAINT] Mutual violation in {view_name}: "
                              f"{self.class_names[class_a]}={conf_a:.3f} × "
                              f"{self.class_names[class_b]}={conf_b:.3f} = {mutual_penalty:.3f}")
        
        # 歸一化到 batch size
        if batch_size > 0:
            mutual_loss = total_penalty / batch_size
        else:
            mutual_loss = 0.0
        
        return torch.tensor(mutual_loss, device=self.device, requires_grad=True), violation_count, debug_info
    
    def _get_max_confidence_for_class(self, detections, target_class, threshold):
        """
        獲取指定類別的最大置信度
        
        Args:
            detections: 單個樣本的檢測結果
            target_class: 目標類別索引 (0-3)
            threshold: 置信度閾值
            
        Returns:
            max_conf: 該類別的最大置信度
        """
        
        if detections is None or len(detections) == 0:
            return 0.0
        
        try:
            # 處理不同的檢測輸出格式
            if len(detections.shape) == 2:
                # 格式1: [num_det, 6] - (x1, y1, x2, y2, conf, cls)
                if detections.shape[1] == 6:
                    confidences = detections[:, 4]  # 總置信度
                    classes = detections[:, 5]      # 預測類別
                    
                    # 找到目標類別的檢測
                    target_mask = (classes == target_class) & (confidences > threshold)
                    
                    if target_mask.any():
                        return confidences[target_mask].max().item()
                    else:
                        return 0.0
                
                # 格式2: [num_det, 5+num_classes] - (x, y, w, h, obj, cls0, cls1, ...)
                elif detections.shape[1] >= 9:  # 5 + 4 classes
                    objectness = detections[:, 4]
                    class_probs = detections[:, 5 + target_class]
                    
                    # 最終置信度 = objectness × class_prob
                    confidences = objectness * class_probs
                    
                    # 過濾低置信度
                    valid_confidences = confidences[confidences > threshold]
                    
                    if len(valid_confidences) > 0:
                        return valid_confidences.max().item()
                    else:
                        return 0.0
            
            # 其他格式處理
            return 0.0
            
        except Exception as e:
            print(f"[DEBUG] Error processing detections for class {target_class}: {e}")
            return 0.0
    
    def compute_soft_mutual_loss(self, class_logits, view_labels):
        """
        基於分類 logits 計算軟互斥損失
        適用於直接的分類預測輸出
        
        Args:
            class_logits: 分類 logits [batch_size, num_classes]
            view_labels: 視圖標籤 [batch_size]
            
        Returns:
            mutual_loss: 軟互斥損失
        """
        
        # 轉換為概率（多標籤場景）
        class_probs = torch.sigmoid(class_logits)
        
        total_penalty = 0.0
        batch_size = view_labels.shape[0]
        
        for batch_idx in range(batch_size):
            view_idx = view_labels[batch_idx].item()
            
            if view_idx not in self.mutually_exclusive_pairs:
                continue
            
            # 對每個互斥對計算惩罰
            for class_a, class_b in self.mutually_exclusive_pairs[view_idx]:
                prob_a = class_probs[batch_idx, class_a]
                prob_b = class_probs[batch_idx, class_b]
                
                # 軟互斥惩罰
                penalty = prob_a * prob_b
                total_penalty += penalty
        
        # 歸一化
        mutual_loss = total_penalty / batch_size if batch_size > 0 else 0.0
        
        return mutual_loss


def test_mutual_constraints():
    """測試互斥約束功能"""
    
    print("=" * 60)
    print("TESTING MUTUALLY EXCLUSIVE CONSTRAINTS")
    print("=" * 60)
    
    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    constraints = MutuallyExclusiveConstraints(device=device)
    
    # 測試數據
    batch_size = 4
    max_detections = 10
    num_classes = 4
    
    # 模擬檢測結果 [batch, max_det, 6] (x1, y1, x2, y2, conf, cls)
    detections = torch.zeros(batch_size, max_detections, 6)
    
    # 設置一些檢測結果
    # Batch 0: A4C 視圖，同時檢測到 MR (class 1) 和 TR (class 3) - 應該違反
    detections[0, 0] = torch.tensor([10, 10, 50, 50, 0.8, 1])  # MR, high confidence
    detections[0, 1] = torch.tensor([60, 60, 100, 100, 0.7, 3])  # TR, high confidence
    
    # Batch 1: A4C 視圖，只檢測到 MR - 正常
    detections[1, 0] = torch.tensor([10, 10, 50, 50, 0.9, 1])  # MR only
    
    # 視圖標籤
    view_labels = torch.tensor([0, 0, 1, 2])  # A4C, A4C, PSAX, PLAX
    
    # 計算互斥損失
    mutual_loss, violations, debug_info = constraints.compute_mutual_exclusion_loss(
        detections, 
        view_labels, 
        confidence_threshold=0.25
    )
    
    print(f"\nResults:")
    print(f"Mutual Loss: {mutual_loss:.6f}")
    print(f"Violations: {violations}")
    print(f"Debug Info: {debug_info}")
    
    # 測試軟約束版本
    print(f"\n" + "=" * 40)
    print("TESTING SOFT CONSTRAINTS")
    print("=" * 40)
    
    class_logits = torch.randn(batch_size, num_classes)
    soft_loss = constraints.compute_soft_mutual_loss(class_logits, view_labels)
    
    print(f"Soft Mutual Loss: {soft_loss:.6f}")
    
    return mutual_loss, violations


if __name__ == "__main__":
    test_mutual_constraints()