#!/usr/bin/env python3
"""
分析 YOLOv5WithClassification 的日誌輸出
檢查 loss.py, metrics.py, val.py 的輸出內容
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_loss_debug_outputs():
    """分析 loss.py 的 DEBUG 輸出"""
    print("=" * 60)
    print("LOSS.PY DEBUG 輸出分析")
    print("=" * 60)
    
    # 從 loss.py 中提取的 DEBUG 輸出類型
    debug_outputs = {
        "初始化信息": [
            "[DEBUG] Classification loss weight: {weight}",
            "[DEBUG] Classification focal gamma: {gamma}"
        ],
        "NaN/Inf 檢測": [
            "[DEBUG] WARNING: NaN values found in classification output!",
            "[DEBUG] WARNING: Inf values found in classification output!",
            "[DEBUG] WARNING: NaN/Inf detected in total_loss!"
        ],
        "過擬合檢測": [
            "[DEBUG] WARNING: Model is predicting only class {class_id} (overfitting)",
            "[DEBUG] WARNING: Only {num_classes} classes in targets"
        ],
        "錯誤處理": [
            "[DEBUG] ERROR in classification loss calculation: {error}"
        ]
    }
    
    for category, outputs in debug_outputs.items():
        print(f"\n{category}:")
        for output in outputs:
            print(f"  - {output}")
    
    return debug_outputs

def analyze_metrics_outputs():
    """分析 metrics.py 的輸出"""
    print("\n" + "=" * 60)
    print("METRICS.PY 輸出分析")
    print("=" * 60)
    
    metrics_outputs = {
        "混淆矩陣日誌": [
            "LOGGER.info: Confusion matrix plotting: {true_count} true labels, {pred_count} pred labels",
            "LOGGER.info: Classification confusion matrix generated successfully",
            "LOGGER.warning: No classification data available for confusion matrix"
        ],
        "混淆矩陣打印": [
            "print: Detection Confusion Matrix:",
            "print: Normalized Detection Confusion Matrix:",
            "print: Classification Confusion Matrix:",
            "print: Normalized Classification Confusion Matrix:"
        ],
        "文件保存": [
            "print: Classification confusion matrix saved to {path}"
        ]
    }
    
    for category, outputs in metrics_outputs.items():
        print(f"\n{category}:")
        for output in outputs:
            print(f"  - {output}")
    
    return metrics_outputs

def analyze_val_outputs():
    """分析 val.py 的輸出"""
    print("\n" + "=" * 60)
    print("VAL.PY 輸出分析")
    print("=" * 60)
    
    val_outputs = {
        "模型信息": [
            "LOGGER.info: Forcing --batch-size 1 square inference",
            "LOGGER.info: Collecting classification data: batch {batch}, targets shape {shape}"
        ],
        "結果打印": [
            "LOGGER.info: {header}",
            "LOGGER.info: {results_table}",
            "LOGGER.info: Speed: {time}ms pre-process, {time}ms inference, {time}ms NMS per image"
        ],
        "分類結果": [
            "print: Classification Confusion Matrix:",
            "LOGGER.info: Classification Results:",
            "LOGGER.info: {class_results_table}"
        ],
        "警告信息": [
            "LOGGER.warning: WARNING ⚠️ no labels found in {task} set",
            "LOGGER.info: WARNING ⚠️ confidence threshold {threshold} > 0.001 produces invalid results"
        ]
    }
    
    for category, outputs in val_outputs.items():
        print(f"\n{category}:")
        for output in outputs:
            print(f"  - {output}")
    
    return val_outputs

def analyze_actual_log_file():
    """分析實際的日誌文件"""
    print("\n" + "=" * 60)
    print("實際日誌文件分析")
    print("=" * 60)
    
    log_file = "files/job_262554_1_1757659951.log"
    if not os.path.exists(log_file):
        print(f"日誌文件不存在: {log_file}")
        return
    
    # 讀取日誌文件
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分析 DEBUG 輸出
    debug_patterns = {
        "分類損失權重": r"\[DEBUG\] Classification loss weight: ([\d.]+)",
        "Focal Loss Gamma": r"\[DEBUG\] Classification focal gamma: ([\d.]+)",
        "過擬合警告": r"\[DEBUG\] WARNING: Model is predicting only class (\d+) \(overfitting\)",
        "NaN/Inf 警告": r"\[DEBUG\] WARNING: (NaN|Inf) values found in classification output!",
        "批次信息": r"\[DEBUG\] Batch \d+ information:",
        "分類結果": r"Classification Results - Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-Score: ([\d.]+)"
    }
    
    print("\n實際日誌中的 DEBUG 輸出統計:")
    for pattern_name, pattern in debug_patterns.items():
        matches = re.findall(pattern, content)
        print(f"  {pattern_name}: {len(matches)} 次")
        if matches and len(matches) <= 5:  # 只顯示前5個匹配
            print(f"    範例: {matches[:3]}")
        elif matches:
            print(f"    範例: {matches[:3]} ... (共{len(matches)}個)")

def analyze_results_csv():
    """分析 results.csv 文件"""
    print("\n" + "=" * 60)
    print("RESULTS.CSV 分析")
    print("=" * 60)
    
    csv_file = "files/testingbalanced/results.csv"
    if not os.path.exists(csv_file):
        print(f"結果文件不存在: {csv_file}")
        return
    
    try:
        df = pd.read_csv(csv_file)
        print(f"訓練輪數: {len(df)}")
        print(f"列名: {list(df.columns)}")
        
        # 分析關鍵指標
        key_metrics = ['train/cls_task_loss', 'val/cls_task_loss', 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5']
        available_metrics = [col for col in key_metrics if col in df.columns]
        
        if available_metrics:
            print(f"\n關鍵指標趨勢:")
            for metric in available_metrics:
                if metric in df.columns:
                    initial = df[metric].iloc[0]
                    final = df[metric].iloc[-1]
                    print(f"  {metric}: {initial:.6f} → {final:.6f}")
        
        # 檢查是否有異常值
        print(f"\n數據完整性檢查:")
        for col in df.columns:
            if df[col].isna().any():
                print(f"  {col}: 有 {df[col].isna().sum()} 個 NaN 值")
            if (df[col] == float('inf')).any():
                print(f"  {col}: 有 {sum(df[col] == float('inf'))} 個 Inf 值")
        
    except Exception as e:
        print(f"讀取 CSV 文件時出錯: {e}")

def create_output_summary():
    """創建輸出摘要"""
    print("\n" + "=" * 60)
    print("輸出摘要")
    print("=" * 60)
    
    summary = {
        "loss.py 主要輸出": [
            "分類損失權重和 Focal Loss 參數初始化信息",
            "NaN/Inf 值檢測和警告",
            "過擬合檢測（模型只預測單一類別）",
            "分類損失計算錯誤處理"
        ],
        "metrics.py 主要輸出": [
            "混淆矩陣生成和保存信息",
            "檢測和分類混淆矩陣的打印輸出",
            "文件保存確認信息"
        ],
        "val.py 主要輸出": [
            "模型推理速度信息",
            "檢測和分類結果表格",
            "分類準確率、精確率、召回率、F1分數",
            "各種警告和錯誤信息"
        ]
    }
    
    for category, items in summary.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")

def main():
    """主函數"""
    print("YOLOv5WithClassification 日誌輸出分析")
    print("=" * 60)
    
    # 分析各文件的輸出
    analyze_loss_debug_outputs()
    analyze_metrics_outputs()
    analyze_val_outputs()
    
    # 分析實際日誌文件
    analyze_actual_log_file()
    
    # 分析結果文件
    analyze_results_csv()
    
    # 創建摘要
    create_output_summary()
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
