#!/usr/bin/env python3
"""
使用互斥約束的 YOLOv5WithClassification 訓練示例
Training Example with Mutually Exclusive Constraints

這個腳本展示如何：
1. 啟用互斥約束
2. 監控約束違反情況
3. 調整約束權重
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5c root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from train import main as train_main, parse_opt


def create_mutual_constraints_training_args():
    """創建包含互斥約束的訓練參數"""
    
    # 基本訓練參數
    args = [
        '--data', '../regurgitationV4/data.yaml',  # 使用 V4 數據集（已知穩定）
        '--cfg', 'models/yolov5sc_p3.yaml',       # 使用 Small 模型（穩定）
        '--epochs', '50',                         # 較短的訓練用於測試
        '--batch-size', '16',                     # 較小批次大小用於測試
        '--imgsz', '416',
        '--name', 'mutual_constraints_test',      # 實驗名稱
        '--cache', 'ram',
        '--patience', '10',                       # 允許早停用於測試
        '--hyp', 'data/hyps/hyp.with_mutual_constraints.yaml',  # 使用互斥約束配置
        '--device', '0'                           # 使用 GPU
    ]
    
    return args


def monitor_mutual_constraints_training():
    """監控互斥約束訓練過程"""
    
    print("[TRAIN] STARTING TRAINING WITH MUTUALLY EXCLUSIVE CONSTRAINTS")
    print("=" * 80)
    
    # 設置參數
    training_args = create_mutual_constraints_training_args()
    
    print("Training Configuration:")
    print("[MODEL] YOLOv5sc P3 (Small Classification)")
    print("[DATA] Dataset: regurgitationV4 (stable version)")
    print("[CONFIG] Hyperparameters: hyp.with_mutual_constraints.yaml")
    print("[CONSTRAINT] Mutual Constraints: ENABLED")
    print("[BENEFIT] Expected benefits:")
    print("   - Reduced false positives in mutual exclusive detections")
    print("   - Better medical consistency")
    print("   - More reliable predictions")
    
    # 顯示約束配置
    print("\n[RULES] Mutual Exclusion Rules:")
    print("   A4C view:  MR (Mitral) <-> TR (Tricuspid)")
    print("   PSAX view: PR (Pulmonary) <-> TR (Tricuspid)")  
    print("   PLAX view: AR (Aortic) <-> MR (Mitral)")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80)
    
    # 解析參數並開始訓練
    opt = parse_opt(training_args)
    
    # 開始訓練
    try:
        train_main(opt)
        print("\n[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
        
        # 檢查結果
        results_dir = Path(opt.project) / opt.name
        if results_dir.exists():
            print(f"\n[RESULTS] Results saved to: {results_dir}")
            print("[FILES] Check the following files:")
            print(f"   - results.csv: Training metrics")
            print(f"   - classification_metrics.txt: Classification performance")
            print(f"   - results.png: Training curves")
            
            # 檢查是否有約束違反日誌
            print("\n[MONITOR] Look for constraint violation messages in the training output:")
            print("   - '[CONSTRAINT] X mutual exclusion violations detected'")
            print("   - These should decrease as training progresses")
            
        return True
        
    except Exception as e:
        print(f"\n[ERROR] TRAINING FAILED: {e}")
        return False


def dry_run_test():
    """乾跑測試 - 不實際訓練，只測試配置"""
    
    print("[DRY-RUN] CONFIGURATION VALIDATION")
    print("=" * 60)
    
    try:
        # 導入必要的模塊來測試配置
        import torch
        from models.yolo import Model
        from utils.loss import ComputeLoss
        import yaml
        
        # 載入模型配置
        model_cfg = 'models/yolov5sc_p3.yaml'
        hyp_cfg = 'data/hyps/hyp.with_mutual_constraints.yaml'
        
        print(f"[CONFIG] Testing model config: {model_cfg}")
        print(f"[CONFIG] Testing hyperparameter config: {hyp_cfg}")
        
        # 載入超參數
        with open(hyp_cfg, 'r') as f:
            hyp = yaml.safe_load(f)
        
        print(f"\n[SETTINGS] Mutual constraint settings:")
        print(f"   use_mutual_constraints: {hyp.get('use_mutual_constraints', False)}")
        print(f"   mutual_constraint_weight: {hyp.get('mutual_constraint_weight', 0.15)}")
        print(f"   mutual_confidence_threshold: {hyp.get('mutual_confidence_threshold', 0.25)}")
        
        # 創建模型（小批次測試）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = Model(model_cfg, ch=3, nc=4, num_cls=3).to(device)
        model.hyp = hyp
        
        # 創建損失函數
        compute_loss = ComputeLoss(model)
        
        # 檢查互斥約束是否正確初始化
        if hasattr(compute_loss, 'mutual_constraints') and compute_loss.mutual_constraints is not None:
            print("[PASS] Mutual constraints correctly initialized in ComputeLoss")
        else:
            print("[FAIL] Mutual constraints NOT initialized")
            return False
        
        print("[PASS] Dry run test passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Dry run test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函數"""
    
    parser = argparse.ArgumentParser(description='Test Mutually Exclusive Constraints')
    parser.add_argument('--mode', choices=['test', 'dry-run', 'train'], default='test',
                       help='Test mode: test=unit tests, dry-run=config validation, train=full training')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # 運行單元測試
        return test_basic_functionality() and test_violation_scenarios()
    
    elif args.mode == 'dry-run':
        # 乾跑測試
        return dry_run_test()
    
    elif args.mode == 'train':
        # 完整訓練
        return monitor_mutual_constraints_training()
    
    return False


if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n[NEXT] Next steps:")
        print("   1. Run: python test_mutual_constraints.py --mode=dry-run")
        print("   2. Run: python test_mutual_constraints.py --mode=train")
        print("   3. Check training logs for constraint violations")
    else:
        print("[FAIL] TESTS FAILED - Please check the implementation")
    
    exit(0 if success else 1)
