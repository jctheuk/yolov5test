#!/usr/bin/env python3
"""
YOLOv5 Performance Analysis Tool
Analyzes training results and provides insights for improvement
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
import seaborn as sns

class PerformanceAnalyzer:
    def __init__(self, exp_dir):
        self.exp_dir = Path(exp_dir)
        self.results_file = self.exp_dir / "results.csv"
        self.confusion_matrix_file = self.exp_dir / "confusion_matrix.png"
        
    def analyze_training_curves(self):
        """分析訓練曲線"""
        if not self.results_file.exists():
            print(f"❌ 找不到結果文件: {self.results_file}")
            return None
            
        df = pd.read_csv(self.results_file)
        
        # 創建訓練曲線圖
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YOLOv5 訓練性能分析', fontsize=16, fontweight='bold')
        
        # 1. 損失曲線
        ax1 = axes[0, 0]
        ax1.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
        ax1.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
        ax1.set_title('Box Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 分類損失
        ax2 = axes[0, 1]
        ax2.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='blue')
        ax2.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='red')
        ax2.set_title('Classification Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 分類任務損失
        ax3 = axes[0, 2]
        ax3.plot(df['epoch'], df['train/cls_task_loss'], label='Train Cls Task Loss', color='blue')
        ax3.plot(df['epoch'], df['val/cls_task_loss'], label='Val Cls Task Loss', color='red')
        ax3.set_title('Classification Task Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. mAP 曲線
        ax4 = axes[1, 0]
        ax4.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5', color='green')
        ax4.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='orange')
        ax4.set_title('mAP Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('mAP')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 精確率和召回率
        ax5 = axes[1, 1]
        ax5.plot(df['epoch'], df['metrics/precision'], label='Precision', color='purple')
        ax5.plot(df['epoch'], df['metrics/recall'], label='Recall', color='brown')
        ax5.set_title('Precision & Recall')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 學習率
        ax6 = axes[1, 2]
        ax6.plot(df['epoch'], df['x/lr0'], label='Learning Rate', color='red')
        ax6.set_title('Learning Rate Schedule')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Learning Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
        
    def analyze_final_metrics(self, df):
        """分析最終指標"""
        if df is None:
            return
            
        print("\n" + "="*60)
        print("最終訓練指標分析")
        print("="*60)
        
        # 獲取最後幾個 epoch 的平均值
        last_epochs = df.tail(5)
        
        print(f"📊 檢測性能 (最後5個epoch平均):")
        print(f"   mAP@0.5: {last_epochs['metrics/mAP_0.5'].mean():.4f}")
        print(f"   mAP@0.5:0.95: {last_epochs['metrics/mAP_0.5:0.95'].mean():.4f}")
        print(f"   精確率: {last_epochs['metrics/precision'].mean():.4f}")
        print(f"   召回率: {last_epochs['metrics/recall'].mean():.4f}")
        
        print(f"\n📈 損失分析:")
        print(f"   最終 Box Loss: {last_epochs['val/box_loss'].mean():.4f}")
        print(f"   最終 Cls Loss: {last_epochs['val/cls_loss'].mean():.4f}")
        print(f"   最終 Cls Task Loss: {last_epochs['val/cls_task_loss'].mean():.4f}")
        
        # 分析收斂性
        print(f"\n🔄 收斂性分析:")
        early_epochs = df.head(5)
        late_epochs = df.tail(5)
        
        box_loss_improvement = (early_epochs['val/box_loss'].mean() - late_epochs['val/box_loss'].mean()) / early_epochs['val/box_loss'].mean() * 100
        map_improvement = (late_epochs['metrics/mAP_0.5'].mean() - early_epochs['metrics/mAP_0.5'].mean()) / early_epochs['metrics/mAP_0.5'].mean() * 100
        
        print(f"   Box Loss 改善: {box_loss_improvement:.1f}%")
        print(f"   mAP 改善: {map_improvement:.1f}%")
        
        # 提供改進建議
        self.provide_improvement_suggestions(df)
        
    def provide_improvement_suggestions(self, df):
        """提供改進建議"""
        print(f"\n💡 改進建議:")
        
        last_epochs = df.tail(5)
        avg_map = last_epochs['metrics/mAP_0.5'].mean()
        avg_precision = last_epochs['metrics/precision'].mean()
        avg_recall = last_epochs['metrics/recall'].mean()
        
        if avg_map < 0.3:
            print("   🔴 mAP 過低 - 建議:")
            print("      - 檢查數據集標註質量")
            print("      - 增加訓練輪數")
            print("      - 調整 IoU 閾值")
            print("      - 檢查類別不平衡問題")
            
        if avg_precision < 0.5:
            print("   🟡 精確率偏低 - 建議:")
            print("      - 提高檢測閾值")
            print("      - 增加負樣本訓練")
            print("      - 檢查假陽性案例")
            
        if avg_recall < 0.5:
            print("   🟡 召回率偏低 - 建議:")
            print("      - 降低檢測閾值")
            print("      - 增加正樣本訓練")
            print("      - 檢查假陰性案例")
            
        # 檢查過擬合
        train_box_loss = df['train/box_loss'].tail(5).mean()
        val_box_loss = df['val/box_loss'].tail(5).mean()
        
        if val_box_loss > train_box_loss * 1.2:
            print("   🟠 可能過擬合 - 建議:")
            print("      - 增加數據擴增")
            print("      - 減少模型複雜度")
            print("      - 提前停止訓練")
            
    def analyze_dataset_distribution(self, data_yaml_path):
        """分析數據集分布"""
        print(f"\n📁 數據集分析:")
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        train_path = Path(data_config['train'])
        val_path = Path(data_config['val'])
        
        # 統計圖像數量
        train_images = len(list(train_path.glob('*.png')))
        val_images = len(list(val_path.glob('*.png')))
        
        print(f"   訓練圖像: {train_images}")
        print(f"   驗證圖像: {val_images}")
        print(f"   檢測類別: {data_config['names']}")
        print(f"   分類類別: {data_config['cls_names']}")
        
        # 分析標註分布
        self.analyze_label_distribution(train_path, "訓練集")
        self.analyze_label_distribution(val_path, "驗證集")
        
    def analyze_label_distribution(self, images_path, dataset_name):
        """分析標註分布"""
        labels_path = images_path.parent / 'labels'
        
        if not labels_path.exists():
            print(f"   ⚠️  找不到 {dataset_name} 標註文件")
            return
            
        detection_counts = {'AR': 0, 'MR': 0, 'PR': 0, 'TR': 0}
        classification_counts = {'PSAX': 0, 'PLAX': 0, 'A4C': 0}
        
        for label_file in labels_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # 檢測標註
                    class_id = int(parts[0])
                    if class_id < len(detection_counts):
                        class_name = list(detection_counts.keys())[class_id]
                        detection_counts[class_name] += 1
                        
                if len(parts) == 1:
                    # 分類標註
                    class_id = int(parts[0])
                    if class_id < len(classification_counts):
                        class_name = list(classification_counts.keys())[class_id]
                        classification_counts[class_name] += 1
                        
        print(f"\n   📊 {dataset_name} 標註分布:")
        print(f"      檢測標註: {detection_counts}")
        print(f"      分類標註: {classification_counts}")

def main():
    """主函數"""
    print("🔍 YOLOv5 性能分析工具")
    print("="*50)
    
    # 找到最新的實驗目錄
    runs_dir = Path("runs/train")
    if not runs_dir.exists():
        print("❌ 找不到 runs/train 目錄")
        return
        
    exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp')]
    if not exp_dirs:
        print("❌ 找不到實驗目錄")
        return
        
    # 使用最新的實驗目錄
    latest_exp = max(exp_dirs, key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else 0)
    print(f"📁 分析實驗目錄: {latest_exp}")
    
    analyzer = PerformanceAnalyzer(latest_exp)
    
    # 分析訓練曲線
    df = analyzer.analyze_training_curves()
    
    # 分析最終指標
    analyzer.analyze_final_metrics(df)
    
    # 分析數據集分布
    data_yaml_path = "../Regurgitation-YOLODataset-Detection/data.yaml"
    if Path(data_yaml_path).exists():
        analyzer.analyze_dataset_distribution(data_yaml_path)
    else:
        print(f"⚠️  找不到數據配置文件: {data_yaml_path}")
        
    print(f"\n✅ 分析完成！詳細圖表已保存到: {latest_exp}/training_analysis.png")

if __name__ == "__main__":
    main()
