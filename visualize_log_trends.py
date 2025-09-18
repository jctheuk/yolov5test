#!/usr/bin/env python3
"""
可視化 YOLOv5WithClassification 日誌輸出趨勢
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def parse_log_file():
    """解析日誌文件，提取關鍵信息"""
    log_file = "files/job_262554_1_1757659951.log"
    
    if not Path(log_file).exists():
        print(f"日誌文件不存在: {log_file}")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取過擬合警告
    overfitting_warnings = re.findall(r'\[DEBUG\] WARNING: Model is predicting only class (\d+) \(overfitting\)', content)
    
    # 提取分類結果
    classification_results = re.findall(r'Classification Results - Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)', content)
    
    # 提取批次信息
    batch_info = re.findall(r'\[DEBUG\] Batch (\d+) information:', content)
    
    return {
        'overfitting_warnings': overfitting_warnings,
        'classification_results': classification_results,
        'batch_info': batch_info
    }

def analyze_results_csv():
    """分析 results.csv 文件"""
    csv_file = "files/testingbalanced/results.csv"
    
    if not Path(csv_file).exists():
        print(f"結果文件不存在: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"讀取 CSV 文件時出錯: {e}")
        return None

def create_overfitting_analysis():
    """創建過擬合分析圖表"""
    log_data = parse_log_file()
    if not log_data:
        return
    
    # 統計過擬合警告
    overfitting_count = len(log_data['overfitting_warnings'])
    unique_classes = set(log_data['overfitting_warnings'])
    
    print(f"過擬合分析:")
    print(f"  總警告次數: {overfitting_count}")
    print(f"  預測的類別: {list(unique_classes)}")
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 過擬合警告分布
    class_counts = {}
    for class_id in log_data['overfitting_warnings']:
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    ax1.bar(class_counts.keys(), class_counts.values(), color='red', alpha=0.7)
    ax1.set_title('過擬合警告分布\n(模型預測的類別)')
    ax1.set_xlabel('預測類別')
    ax1.set_ylabel('警告次數')
    ax1.grid(True, alpha=0.3)
    
    # 分類性能趨勢
    if log_data['classification_results']:
        results = np.array(log_data['classification_results'], dtype=float)
        epochs = range(len(results))
        
        ax2.plot(epochs, results[:, 0], label='準確率', marker='o', markersize=3)
        ax2.plot(epochs, results[:, 1], label='精確率', marker='s', markersize=3)
        ax2.plot(epochs, results[:, 2], label='召回率', marker='^', markersize=3)
        ax2.plot(epochs, results[:, 3], label='F1分數', marker='d', markersize=3)
        
        ax2.set_title('分類性能趨勢')
        ax2.set_xlabel('訓練輪數')
        ax2.set_ylabel('性能指標')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_training_metrics_analysis():
    """創建訓練指標分析圖表"""
    df = analyze_results_csv()
    if df is None:
        return
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 損失趨勢
    axes[0, 0].plot(df['               epoch'], df['      train/box_loss'], label='訓練 Box Loss', alpha=0.8)
    axes[0, 0].plot(df['               epoch'], df['        val/box_loss'], label='驗證 Box Loss', alpha=0.8)
    axes[0, 0].set_title('Box Loss 趨勢')
    axes[0, 0].set_xlabel('訓練輪數')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 分類任務損失
    axes[0, 1].plot(df['               epoch'], df[' train/cls_task_loss'], label='訓練分類損失', alpha=0.8)
    axes[0, 1].plot(df['               epoch'], df['   val/cls_task_loss'], label='驗證分類損失', alpha=0.8)
    axes[0, 1].set_title('分類任務損失趨勢')
    axes[0, 1].set_xlabel('訓練輪數')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # mAP 趨勢
    axes[1, 0].plot(df['               epoch'], df['     metrics/mAP_0.5'], label='mAP@0.5', alpha=0.8)
    axes[1, 0].plot(df['               epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', alpha=0.8)
    axes[1, 0].set_title('mAP 趨勢')
    axes[1, 0].set_xlabel('訓練輪數')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 學習率趨勢
    axes[1, 1].plot(df['               epoch'], df['               x/lr0'], label='LR0', alpha=0.8)
    axes[1, 1].plot(df['               epoch'], df['               x/lr1'], label='LR1', alpha=0.8)
    axes[1, 1].plot(df['               epoch'], df['               x/lr2'], label='LR2', alpha=0.8)
    axes[1, 1].set_title('學習率趨勢')
    axes[1, 1].set_xlabel('訓練輪數')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_metrics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report():
    """創建摘要報告"""
    log_data = parse_log_file()
    df = analyze_results_csv()
    
    print("\n" + "="*60)
    print("YOLOv5WithClassification 日誌輸出摘要報告")
    print("="*60)
    
    if log_data:
        print(f"\n日誌文件分析:")
        print(f"  過擬合警告次數: {len(log_data['overfitting_warnings'])}")
        print(f"  分類結果記錄次數: {len(log_data['classification_results'])}")
        print(f"  批次信息記錄次數: {len(log_data['batch_info'])}")
        
        if log_data['classification_results']:
            latest_result = log_data['classification_results'][-1]
            print(f"\n最新分類性能:")
            print(f"  準確率: {float(latest_result[0]):.4f}")
            print(f"  精確率: {float(latest_result[1]):.4f}")
            print(f"  召回率: {float(latest_result[2]):.4f}")
            print(f"  F1分數: {float(latest_result[3]):.4f}")
    
    if df is not None:
        print(f"\n訓練結果分析:")
        print(f"  總訓練輪數: {len(df)}")
        print(f"  最終 mAP@0.5: {df['     metrics/mAP_0.5'].iloc[-1]:.6f}")
        print(f"  最終 mAP@0.5:0.95: {df['metrics/mAP_0.5:0.95'].iloc[-1]:.6f}")
        print(f"  最終分類損失: {df['   val/cls_task_loss'].iloc[-1]:.6f}")
        
        # 檢查是否有改善
        initial_map = df['     metrics/mAP_0.5'].iloc[0]
        final_map = df['     metrics/mAP_0.5'].iloc[-1]
        improvement = final_map - initial_map
        
        print(f"\n性能改善:")
        print(f"  mAP@0.5 改善: {improvement:.6f} ({improvement/initial_map*100:.2f}%)")
    
    print(f"\n關鍵問題:")
    if log_data and len(log_data['overfitting_warnings']) > 0:
        print(f"  ⚠️  嚴重過擬合: 模型持續預測單一類別")
    if log_data and log_data['classification_results']:
        latest_acc = float(log_data['classification_results'][-1][0])
        if latest_acc < 0.6:
            print(f"  ⚠️  分類準確率偏低: {latest_acc:.4f}")
    
    print(f"\n建議:")
    print(f"  1. 增加正則化措施防止過擬合")
    print(f"  2. 檢查數據集類別平衡")
    print(f"  3. 調整學習率和損失函數參數")
    print(f"  4. 考慮使用早停機制")

def main():
    """主函數"""
    print("YOLOv5WithClassification 日誌輸出可視化分析")
    print("="*60)
    
    # 創建過擬合分析
    print("創建過擬合分析圖表...")
    create_overfitting_analysis()
    
    # 創建訓練指標分析
    print("創建訓練指標分析圖表...")
    create_training_metrics_analysis()
    
    # 創建摘要報告
    create_summary_report()
    
    print(f"\n分析完成！生成的圖表:")
    print(f"  - overfitting_analysis.png")
    print(f"  - training_metrics_analysis.png")

if __name__ == "__main__":
    main()
