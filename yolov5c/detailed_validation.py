#!/usr/bin/env python3
"""
Detailed YOLOv5 Validation and Analysis
Provides comprehensive performance analysis with confusion matrices and error analysis
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import subprocess

def run_detailed_validation(weights_path, data_yaml, conf_threshold=0.001, iou_threshold=0.6):
    """運行詳細驗證"""
    
    print("🔍 運行詳細驗證分析...")
    print(f"   權重文件: {weights_path}")
    print(f"   數據配置: {data_yaml}")
    print(f"   置信度閾值: {conf_threshold}")
    print(f"   IoU 閾值: {iou_threshold}")
    
    # 運行驗證命令
    cmd = [
        "python", "val.py",
        "--data", data_yaml,
        "--weights", weights_path,
        "--conf", str(conf_threshold),
        "--iou", str(iou_threshold),
        "--save-txt",
        "--save-conf",
        "--save-json",
        "--verbose"
    ]
    
    print(f"\n🚀 執行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 驗證完成")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  警告信息:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 驗證失敗: {e}")
        print(f"錯誤輸出: {e.stderr}")
        return False
        
    return True

def analyze_confusion_matrix(exp_dir):
    """分析混淆矩陣"""
    exp_path = Path(exp_dir)
    confusion_matrix_file = exp_path / "confusion_matrix.png"
    
    if confusion_matrix_file.exists():
        print(f"\n📊 混淆矩陣已生成: {confusion_matrix_file}")
        print("   請查看混淆矩陣圖像來分析分類錯誤模式")
    else:
        print(f"⚠️  找不到混淆矩陣文件: {confusion_matrix_file}")

def analyze_predictions(exp_dir, data_yaml):
    """分析預測結果"""
    exp_path = Path(exp_dir)
    predictions_dir = exp_path / "predictions"
    
    if not predictions_dir.exists():
        print(f"⚠️  找不到預測結果目錄: {predictions_dir}")
        return
        
    print(f"\n📈 預測結果分析:")
    
    # 讀取數據配置
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        
    detection_classes = data_config['names']
    classification_classes = data_config['cls_names']
    
    print(f"   檢測類別: {detection_classes}")
    print(f"   分類類別: {classification_classes}")
    
    # 分析預測文件
    txt_files = list(predictions_dir.glob("*.txt"))
    if txt_files:
        print(f"   預測文件數量: {len(txt_files)}")
        
        # 統計檢測結果
        detection_counts = {cls: 0 for cls in detection_classes}
        confidence_scores = []
        
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 6:  # class_id, x, y, w, h, conf
                    class_id = int(parts[0])
                    confidence = float(parts[5])
                    
                    if class_id < len(detection_classes):
                        detection_counts[detection_classes[class_id]] += 1
                        confidence_scores.append(confidence)
                        
        print(f"\n   📊 檢測結果統計:")
        for cls, count in detection_counts.items():
            print(f"      {cls}: {count}")
            
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            print(f"   平均置信度: {avg_confidence:.3f}")
            print(f"   最高置信度: {max(confidence_scores):.3f}")
            print(f"   最低置信度: {min(confidence_scores):.3f}")

def generate_error_analysis_report(exp_dir, data_yaml):
    """生成錯誤分析報告"""
    exp_path = Path(exp_dir)
    
    print(f"\n📋 生成錯誤分析報告...")
    
    # 創建報告文件
    report_file = exp_path / "error_analysis_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# YOLOv5 錯誤分析報告\n\n")
        
        f.write("## 驗證配置\n")
        f.write(f"- 實驗目錄: {exp_path}\n")
        f.write(f"- 數據配置: {data_yaml}\n\n")
        
        f.write("## 性能指標\n")
        f.write("請查看以下文件獲取詳細指標:\n")
        f.write("- `results.csv`: 詳細的訓練和驗證指標\n")
        f.write("- `confusion_matrix.png`: 混淆矩陣\n")
        f.write("- `PR_curve.png`: 精確率-召回率曲線\n")
        f.write("- `F1_curve.png`: F1 分數曲線\n\n")
        
        f.write("## 改進建議\n\n")
        f.write("### 如果 mAP 過低:\n")
        f.write("1. 檢查數據集標註質量\n")
        f.write("2. 增加訓練輪數\n")
        f.write("3. 調整 IoU 閾值\n")
        f.write("4. 檢查類別不平衡問題\n\n")
        
        f.write("### 如果精確率偏低:\n")
        f.write("1. 提高檢測閾值\n")
        f.write("2. 增加負樣本訓練\n")
        f.write("3. 檢查假陽性案例\n\n")
        
        f.write("### 如果召回率偏低:\n")
        f.write("1. 降低檢測閾值\n")
        f.write("2. 增加正樣本訓練\n")
        f.write("3. 檢查假陰性案例\n\n")
        
        f.write("## 下一步行動\n")
        f.write("1. 運行 `python analyze_performance.py` 進行詳細分析\n")
        f.write("2. 檢查混淆矩陣中的錯誤模式\n")
        f.write("3. 分析假陽性和假陰性案例\n")
        f.write("4. 根據分析結果調整超參數\n")
        
    print(f"✅ 錯誤分析報告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="詳細 YOLOv5 驗證和分析")
    parser.add_argument("--weights", type=str, required=True, help="權重文件路徑")
    parser.add_argument("--data", type=str, default="../Regurgitation-YOLODataset-Detection/data.yaml", help="數據配置文件")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度閾值")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU 閾值")
    parser.add_argument("--exp-dir", type=str, help="實驗目錄 (自動檢測最新)")
    
    args = parser.parse_args()
    
    # 檢查文件是否存在
    if not Path(args.weights).exists():
        print(f"❌ 權重文件不存在: {args.weights}")
        return
        
    if not Path(args.data).exists():
        print(f"❌ 數據配置文件不存在: {args.data}")
        return
        
    # 確定實驗目錄
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        # 自動找到最新的實驗目錄
        runs_dir = Path("runs/train")
        if runs_dir.exists():
            exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('exp')]
            if exp_dirs:
                exp_dir = str(max(exp_dirs, key=lambda x: int(x.name[3:]) if x.name[3:].isdigit() else 0))
            else:
                exp_dir = "runs/train/exp"
        else:
            exp_dir = "runs/train/exp"
            
    print(f"📁 實驗目錄: {exp_dir}")
    
    # 運行詳細驗證
    success = run_detailed_validation(args.weights, args.data, args.conf, args.iou)
    
    if success:
        # 分析結果
        analyze_confusion_matrix(exp_dir)
        analyze_predictions(exp_dir, args.data)
        generate_error_analysis_report(exp_dir, args.data)
        
        print(f"\n🎉 詳細驗證和分析完成！")
        print(f"📊 請查看以下文件:")
        print(f"   - {exp_dir}/confusion_matrix.png")
        print(f"   - {exp_dir}/PR_curve.png")
        print(f"   - {exp_dir}/F1_curve.png")
        print(f"   - {exp_dir}/error_analysis_report.md")
        print(f"\n🔍 運行性能分析: python analyze_performance.py")

if __name__ == "__main__":
    main()
