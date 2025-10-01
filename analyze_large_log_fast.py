#!/usr/bin/env python3
"""
快速分析大型訓練日誌文件
優化版本：使用採樣和增量處理
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

def quick_summary(log_file, sample_rate=100):
    """
    快速摘要：只採樣部分行來快速了解訓練狀態
    
    Args:
        log_file: 日誌文件路徑
        sample_rate: 採樣率（每 N 行處理一行）
    """
    print(f"🚀 快速分析模式（採樣率：每 {sample_rate} 行）")
    print(f"📖 讀取日誌文件: {log_file}")
    
    log_file = Path(log_file)
    
    # 數據存儲
    data = {
        'cls_task_losses': [],
        'accuracies': [],
        'epochs': [],
        'warnings': 0,
        'errors': 0,
        'total_lines': 0
    }
    
    # 逐行讀取並採樣
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            data['total_lines'] = i + 1
            
            # 採樣
            if i % sample_rate != 0 and i < 1000:  # 前 1000 行全部處理
                continue
            
            # 提取損失值
            loss_match = re.search(r'(\d+\.\d+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if loss_match:
                cls_task_loss = float(loss_match.group(5))
                data['cls_task_losses'].append(cls_task_loss)
            
            # 提取準確率
            acc_match = re.search(r'Batch accuracy:\s+([\d.]+)', line)
            if acc_match:
                acc = float(acc_match.group(1))
                data['accuracies'].append(acc)
            
            # 提取 Epoch
            epoch_match = re.search(r'(\d+)/(\d+)', line)
            if epoch_match and 'Epoch' in line:
                epoch = int(epoch_match.group(1))
                data['epochs'].append(epoch)
            
            # 計數警告和錯誤
            if 'WARNING' in line or '⚠️' in line:
                data['warnings'] += 1
            if 'ERROR' in line or '❌' in line or 'Traceback' in line:
                data['errors'] += 1
            
            # 進度顯示
            if i % 100000 == 0:
                print(f"  處理進度：{i:,} 行...")
    
    print(f"✅ 處理完成：{data['total_lines']:,} 行")
    
    # 生成摘要
    print("\n" + "="*80)
    print("📊 快速摘要報告")
    print("="*80)
    
    print(f"\n### 基本統計")
    print(f"  總行數: {data['total_lines']:,}")
    print(f"  總 Epochs: {len(set(data['epochs']))}")
    print(f"  採樣的訓練步數: {len(data['cls_task_losses'])}")
    print(f"  採樣的準確率數據: {len(data['accuracies'])}")
    
    if data['cls_task_losses']:
        print(f"\n### 損失統計（分類任務）")
        print(f"  最小值: {min(data['cls_task_losses']):.4f}")
        print(f"  最大值: {max(data['cls_task_losses']):.4f}")
        print(f"  平均值: {np.mean(data['cls_task_losses']):.4f}")
        print(f"  最終值: {data['cls_task_losses'][-1]:.4f}")
        
        # 趨勢分析
        if len(data['cls_task_losses']) > 10:
            first_10 = np.mean(data['cls_task_losses'][:10])
            last_10 = np.mean(data['cls_task_losses'][-10:])
            improvement = (first_10 - last_10) / first_10 * 100 if first_10 > 0 else 0
            print(f"\n  趨勢分析:")
            print(f"    前 10 步平均: {first_10:.4f}")
            print(f"    後 10 步平均: {last_10:.4f}")
            print(f"    改善幅度: {improvement:.2f}%")
    
    if data['accuracies']:
        print(f"\n### 準確率統計")
        print(f"  最小值: {min(data['accuracies']):.4f}")
        print(f"  最大值: {max(data['accuracies']):.4f}")
        print(f"  平均值: {np.mean(data['accuracies']):.4f}")
        if len(data['accuracies']) > 0:
            print(f"  最終值: {data['accuracies'][-1]:.4f}")
    
    print(f"\n### 問題統計")
    print(f"  警告數量: {data['warnings']:,}")
    print(f"  錯誤數量: {data['errors']:,}")
    
    print("\n" + "="*80)
    
    return data

def extract_key_sections(log_file, output_file=None):
    """
    提取關鍵部分：第一個 epoch、最後一個 epoch、錯誤信息
    """
    print(f"\n📋 提取關鍵部分...")
    
    log_file = Path(log_file)
    if output_file is None:
        output_file = log_file.parent / f'{log_file.stem}_key_sections.txt'
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            in_first_epoch = False
            in_last_epoch = False
            first_epoch_lines = []
            last_epoch_lines = []
            error_lines = []
            
            current_epoch = None
            max_epoch = 0
            
            for i, line in enumerate(f_in):
                # 檢測 epoch
                epoch_match = re.search(r'(\d+)/(\d+)', line)
                if epoch_match and 'Epoch' in line:
                    current_epoch = int(epoch_match.group(1))
                    max_epoch = max(max_epoch, current_epoch)
                    
                    if current_epoch == 0:
                        in_first_epoch = True
                    elif current_epoch == max_epoch and current_epoch > 10:
                        in_last_epoch = True
                
                # 收集第一個 epoch 的行
                if in_first_epoch and len(first_epoch_lines) < 500:
                    first_epoch_lines.append(line)
                    if current_epoch > 0:
                        in_first_epoch = False
                
                # 收集最後一個 epoch 的行
                if in_last_epoch and len(last_epoch_lines) < 500:
                    last_epoch_lines.append(line)
                
                # 收集錯誤行
                if 'ERROR' in line or 'Traceback' in line or '❌' in line:
                    error_lines.append(f"Line {i}: {line}")
                
                # 進度
                if i % 500000 == 0:
                    print(f"  處理：{i:,} 行...")
            
            # 寫入關鍵部分
            f_out.write("="*80 + "\n")
            f_out.write("第一個 Epoch（前 500 行）\n")
            f_out.write("="*80 + "\n")
            f_out.writelines(first_epoch_lines)
            
            f_out.write("\n" + "="*80 + "\n")
            f_out.write(f"最後一個 Epoch（Epoch {max_epoch}，前 500 行）\n")
            f_out.write("="*80 + "\n")
            f_out.writelines(last_epoch_lines)
            
            if error_lines:
                f_out.write("\n" + "="*80 + "\n")
                f_out.write(f"錯誤信息（{len(error_lines)} 個）\n")
                f_out.write("="*80 + "\n")
                f_out.writelines(error_lines)
    
    print(f"✅ 關鍵部分已保存: {output_file}")
    print(f"  第一個 Epoch: {len(first_epoch_lines)} 行")
    print(f"  最後一個 Epoch: {len(last_epoch_lines)} 行")
    print(f"  錯誤信息: {len(error_lines)} 個")

def plot_quick_results(data, output_dir):
    """快速繪製結果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 生成快速分析圖表...")
    
    # 損失曲線
    if data['cls_task_losses']:
        plt.figure(figsize=(12, 6))
        plt.plot(data['cls_task_losses'], 'b-', alpha=0.7, linewidth=1)
        plt.title('分類任務損失曲線（採樣數據）', fontsize=14)
        plt.xlabel('採樣步數', fontsize=12)
        plt.ylabel('損失值', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = output_dir / 'quick_loss_curve.png'
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 損失曲線: {loss_path}")
    
    # 準確率曲線
    if data['accuracies']:
        plt.figure(figsize=(12, 6))
        plt.plot(data['accuracies'], 'r-', alpha=0.7, linewidth=1)
        plt.title('批次準確率曲線（採樣數據）', fontsize=14)
        plt.xlabel('採樣步數', fontsize=12)
        plt.ylabel('準確率', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        acc_path = output_dir / 'quick_accuracy_curve.png'
        plt.savefig(acc_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 準確率曲線: {acc_path}")

def main():
    parser = argparse.ArgumentParser(description='快速分析大型訓練日誌文件')
    parser.add_argument('log_file', type=str, help='日誌文件路徑')
    parser.add_argument('--output-dir', type=str, default=None, help='輸出目錄')
    parser.add_argument('--sample-rate', type=int, default=100, help='採樣率（默認：每 100 行）')
    parser.add_argument('--extract-only', action='store_true', help='只提取關鍵部分，不進行分析')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or Path(args.log_file).parent
    
    if args.extract_only:
        # 只提取關鍵部分
        extract_key_sections(args.log_file)
    else:
        # 快速分析
        data = quick_summary(args.log_file, args.sample_rate)
        
        # 繪製圖表
        plot_quick_results(data, output_dir)
        
        # 提取關鍵部分
        extract_key_sections(args.log_file)
        
        print(f"\n✅ 分析完成！結果保存在: {output_dir}")

if __name__ == '__main__':
    main()


