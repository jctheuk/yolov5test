"""
Mutually Exclusive Constraints for Medical Image Detection
互斥事件约束实现

医学背景：
在特定超声视图中，某些反流类型在空间上是互斥的。
同时检测到互斥的反流通常表示模型误检。
"""

import torch
import torch.nn as nn


class MutuallyExclusiveConstraints:
    """
    互斥事件约束 - 惩罚同时检测到互斥的反流类型
    
    Example:
        在 A4C 视图中，如果同时检测到 AR 和 MR：
        E_a4c = E_ar × E_mr
        理想情况：E_a4c = 0（一个为1，另一个为0）
        违反情况：E_a4c > 0（两者都大于0）
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 定义互斥关系
        # 基于 anatomical_constraints.py 的视图定义：
        # - A4C (0) allows: MR (1), TR (3)
        # - PSAX (1) allows: PR (2), TR (3)
        # - PLAX (2) allows: AR (0), MR (1)
        # 格式：{view_index: [(class_a, class_b), ...]}
        # 类别索引：0=AR, 1=MR, 2=PR, 3=TR
        self.mutually_exclusive_pairs = {
            0: [(1, 3)],  # A4C: MR (1) vs TR (3) 互斥
            1: [(2, 3)],  # PSAX: PR (2) vs TR (3) 互斥  
            2: [(0, 1)],  # PLAX: AR (0) vs MR (1) 互斥
        }
        
        # 类别和视图名称（用于调试）
        self.class_names = ['AR', 'MR', 'PR', 'TR']
        self.view_names = ['A4C', 'PSAX', 'PLAX']
        
        print(f"[INFO] Mutually Exclusive Constraints initialized")
        print(f"[INFO] Exclusive pairs:")
        for view_idx, pairs in self.mutually_exclusive_pairs.items():
            for class_a, class_b in pairs:
                print(f"       {self.view_names[view_idx]}: "
                      f"{self.class_names[class_a]} <-> {self.class_names[class_b]}")
    
    def compute_mutual_exclusion_loss(
        self, 
        detection_predictions,
        view_labels,
        confidence_threshold=0.25
    ):
        """
        计算互斥约束损失
        
        Args:
            detection_predictions: 检测预测
                如果是 list: YOLOv5 的三层输出 [P3, P4, P5]
                如果是 tensor: 后处理后的检测 [batch, num_det, 5+nc]
            view_labels: 视图标签 [batch_size]
            confidence_threshold: 置信度阈值
            
        Returns:
            mutual_loss: 互斥约束损失 (标量 tensor)
            violation_count: 违反次数 (int)
        """
        
        total_penalty = 0.0
        violation_count = 0
        
        # 获取 batch size
        if isinstance(view_labels, torch.Tensor):
            batch_size = view_labels.shape[0]
        else:
            batch_size = len(view_labels)
        
        for batch_idx in range(batch_size):
            view_idx = view_labels[batch_idx].item()
            
            # 跳过没有定义互斥对的视图
            if view_idx not in self.mutually_exclusive_pairs:
                continue
            
            exclusive_pairs = self.mutually_exclusive_pairs[view_idx]
            
            # 对每个互斥对计算惩罚
            for class_a, class_b in exclusive_pairs:
                # 获取每个类别的最大置信度
                conf_a = self._get_max_confidence(
                    detection_predictions, 
                    batch_idx,
                    class_a, 
                    confidence_threshold
                )
                conf_b = self._get_max_confidence(
                    detection_predictions,
                    batch_idx,
                    class_b, 
                    confidence_threshold
                )
                
                # 互斥惩罚：E_mutual = P(A) × P(B)
                # 理想情况：一个为 1，另一个为 0，乘积为 0
                # 违反情况：两者都 > 0，乘积 > 0
                mutual_penalty = conf_a * conf_b
                
                # 只惩罚显著的违反（避免噪声）
                if mutual_penalty > 0.01:
                    total_penalty += mutual_penalty
                    violation_count += 1
                    
                    # 打印显著违反
                    if mutual_penalty > 0.1:
                        print(f"[MUTUAL] {self.view_names[view_idx]} violation: "
                              f"{self.class_names[class_a]}={conf_a:.3f} × "
                              f"{self.class_names[class_b]}={conf_b:.3f} = "
                              f"{mutual_penalty:.3f}")
        
        # 归一化到 batch（返回平均惩罚）
        if batch_size > 0:
            mutual_loss = total_penalty / batch_size
        else:
            mutual_loss = 0.0
        
        return torch.tensor(mutual_loss, device=self.device), violation_count
    
    def _get_max_confidence(self, detections, batch_idx, target_class, threshold):
        """
        获取指定类别的最大置信度
        
        Args:
            detections: 检测输出（list 或 tensor）
            batch_idx: batch 索引
            target_class: 目标类别索引
            threshold: 置信度阈值
            
        Returns:
            max_conf: 该类别的最大置信度（0 如果没有检测到）
        """
        
        # 处理不同的检测输出格式
        if isinstance(detections, list):
            # YOLOv5 原始输出：list of [P3, P4, P5]
            # 这种情况比较复杂，需要从原始输出中提取
            # 简化处理：返回 0（在实际使用中需要后处理）
            return 0.0
        
        elif isinstance(detections, torch.Tensor):
            # 后处理后的检测：[batch, num_det, 5+nc]
            # [x, y, w, h, objectness, class_0, ..., class_n]
            
            if detections.dim() == 2:
                # 单个样本 [num_det, 5+nc]
                batch_detections = detections
            elif detections.dim() == 3:
                # 多个样本 [batch, num_det, 5+nc]
                batch_detections = detections[batch_idx]
            else:
                return 0.0
            
            if len(batch_detections) == 0:
                return 0.0
            
            try:
                # 提取置信度
                objectness = batch_detections[:, 4]  # [num_det]
                class_probs = batch_detections[:, 5 + target_class]  # [num_det]
                
                # 最终置信度 = objectness × class_prob
                confidences = objectness * class_probs
                
                # 过滤低置信度
                valid_confidences = confidences[confidences > threshold]
                
                if len(valid_confidences) > 0:
                    return valid_confidences.max().item()
                else:
                    return 0.0
            
            except Exception as e:
                print(f"[DEBUG] Error extracting confidence: {e}")
                return 0.0
        
        else:
            return 0.0
    
    def add_exclusive_pair(self, view_idx, class_a, class_b):
        """
        动态添加互斥对
        
        Args:
            view_idx: 视图索引
            class_a: 类别 A 索引
            class_b: 类别 B 索引
        """
        if view_idx not in self.mutually_exclusive_pairs:
            self.mutually_exclusive_pairs[view_idx] = []
        
        self.mutually_exclusive_pairs[view_idx].append((class_a, class_b))
        
        print(f"[INFO] Added exclusive pair: "
              f"{self.view_names[view_idx]} - "
              f"{self.class_names[class_a]} <-> {self.class_names[class_b]}")
    
    def get_statistics(self):
        """
        获取约束统计信息
        
        Returns:
            stats: 字典，包含约束的统计信息
        """
        stats = {
            'num_views': len(self.mutually_exclusive_pairs),
            'total_pairs': sum(len(pairs) for pairs in self.mutually_exclusive_pairs.values()),
            'pairs_by_view': {
                self.view_names[view_idx]: [
                    f"{self.class_names[a]} <-> {self.class_names[b]}"
                    for a, b in pairs
                ]
                for view_idx, pairs in self.mutually_exclusive_pairs.items()
            }
        }
        return stats


# 测试代码
if __name__ == '__main__':
    """测试互斥约束"""
    
    print("="*60)
    print("Testing Mutually Exclusive Constraints")
    print("="*60)
    
    # 初始化约束
    constraints = MutuallyExclusiveConstraints(device='cpu')
    
    # 打印统计信息
    stats = constraints.get_statistics()
    print(f"\nConstraint Statistics:")
    print(f"  Number of views with constraints: {stats['num_views']}")
    print(f"  Total exclusive pairs: {stats['total_pairs']}")
    print(f"  Pairs by view:")
    for view, pairs in stats['pairs_by_view'].items():
        print(f"    {view}: {pairs}")
    
    # 模拟检测输出
    batch_size = 4
    num_detections = 10
    num_classes = 4
    
    # 创建模拟检测 [batch, num_det, 5+nc]
    detections = torch.rand(batch_size, num_detections, 5 + num_classes)
    
    # 模拟视图标签
    view_labels = torch.tensor([0, 1, 2, 0])  # A4C, PSAX, PLAX, A4C
    
    # 计算互斥损失
    mutual_loss, violations = constraints.compute_mutual_exclusion_loss(
        detections,
        view_labels,
        confidence_threshold=0.25
    )
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Mutual exclusion loss: {mutual_loss:.4f}")
    print(f"  Violations detected: {violations}")
    print(f"{'='*60}")

