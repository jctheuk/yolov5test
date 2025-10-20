# 快速開始：重新訓練指南

## 🎯 目標
使用正確的預訓練權重重新訓練 M 和 L 模型，以獲得更好的性能。

---

## ⚡ 快速執行（TWCC）

### Step 1: 進入工作目錄並下載權重
```bash
cd /work/jonchang3909/yolov5test/yolov5c/

# 檢查是否已有權重文件
ls -lh yolov5*.pt

# 如果缺少，下載所需權重
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
```

### Step 2: 清理資料集快取（重要！）
```bash
# 清理 V1-V5 所有資料集的快取
for DATASET in regurgitationV{1..5}; do
    for SPLIT in train valid test; do
        find "../${DATASET}/${SPLIT}/labels/" -name "*.cache*" -delete 2>/dev/null
    done
    echo "✓ Cleaned $DATASET"
done
```

### Step 3: 重新訓練優先模型

#### 選項 A：只訓練 backbone 配置（推薦，最快）
```bash
# 訓練 Medium model (約 10-15 小時)
bash yolov5mcbackbone.sh

# 訓練 Medium-Large model (約 15-20 小時，使用 Large 權重)
bash yolov5mlcbackbone.sh
```

#### 選項 B：訓練所有配置（完整實驗）
```bash
# Medium model (所有配置)
bash yolov5mcbackbone.sh
bash yolov5mc_p3.sh
bash yolov5mc_p4.sh
bash yolov5mc_p5.sh

# Medium-Large model (所有配置)
bash yolov5mlcbackbone.sh
bash yolov5mlc_p3.sh
bash yolov5mlc_p4.sh
bash yolov5mlc_p5.sh
```

---

## 📊 監控訓練進度

### 查看即時日誌
```bash
# 查看最新的訓練日誌
tail -f runs/yolov5mc_backbone_v1/train.log
```

### 檢查 GPU 使用率
```bash
nvidia-smi -l 1
```

### 預期訓練時間（參考）
| 模型 | 配置 | 每個 Fold | 5 Folds 總計 |
|------|------|----------|-------------|
| yolov5mc | backbone | 2-3 小時 | 10-15 小時 |
| yolov5mc | p3/p4/p5 | 2-3 小時 | 10-15 小時 |
| yolov5mlc | backbone | 3-4 小時 | 15-20 小時 |
| yolov5mlc | p3/p4/p5 | 3-4 小時 | 15-20 小時 |

---

## 🔍 驗證修復

### 確認腳本已正確更新
```bash
# 檢查 yolov5mcbackbone.sh 是否包含 --weights
grep -n "weights yolov5m.pt" yolov5mcbackbone.sh

# 應該看到多行輸出，例如：
# 8:... --weights yolov5m.pt --epochs 300 ...
# 10:... --weights yolov5m.pt --epochs 300 ...
# ...
```

### 驗證所有腳本
```bash
# 檢查所有腳本
echo "=== Small Model Scripts ==="
grep -l "weights yolov5s.pt" yolov5sc*.sh

echo "=== Medium Model Scripts ==="
grep -l "weights yolov5m.pt" yolov5mc*.sh yolov5mlc*.sh

echo "=== Large Model Scripts ==="
grep -l "weights yolov5l.pt" yolov5lc*.sh
```

---

## 📈 提取並比較結果

### 訓練完成後，提取新結果
```bash
cd /work/jonchang3909/yolov5test/

# 創建結果比較腳本
python3 << 'EOF'
import pandas as pd
import glob

# 提取新訓練的結果
results = []
for model in ['yolov5mc_backbone', 'yolov5mlc_backbone']:
    for v in range(1, 6):
        csv_file = f"yolov5c/runs/{model}_v{v}/results.csv"
        try:
            df = pd.read_csv(csv_file)
            last = df.iloc[-1]
            results.append({
                'Model': model,
                'Version': f'v{v}',
                'mAP@0.5': last['metrics/mAP_0.5'],
                'mAP@0.5:0.95': last['metrics/mAP_0.5:0.95'],
                'Precision': last['metrics/precision'],
                'Recall': last['metrics/recall']
            })
        except:
            pass

df_results = pd.DataFrame(results)
print("\n=== New Training Results (With Correct Weights) ===")
print(df_results.to_string(index=False))

# 計算平均值
print("\n=== Average Performance ===")
print(df_results.groupby('Model').mean())
EOF
```

### 與舊結果比較
```bash
python3 << 'EOF'
# 舊結果（沒有正確權重）
old_results = {
    'yolov5sc_backbone': {'mAP@0.5': 0.7945, 'mAP@0.5:0.95': 0.3494},
    'yolov5mc_backbone': {'mAP@0.5': 0.7488, 'mAP@0.5:0.95': 0.2982},
    'yolov5mlc_backbone': {'mAP@0.5': 0.7487, 'mAP@0.5:0.95': 0.2962},
}

# TODO: 填入新結果
new_results = {
    'yolov5mc_backbone': {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0},  # 從訓練獲取
    'yolov5mlc_backbone': {'mAP@0.5': 0.0, 'mAP@0.5:0.95': 0.0},  # 從訓練獲取
}

print("=== Performance Comparison ===")
print(f"{'Model':<25} {'Old mAP@0.5':<12} {'New mAP@0.5':<12} {'Improvement':<12}")
print("-" * 65)
for model in ['yolov5mc_backbone', 'yolov5mlc_backbone']:
    old = old_results[model]['mAP@0.5']
    new = new_results[model]['mAP@0.5']
    improvement = ((new - old) / old * 100) if new > 0 else 0
    print(f"{model:<25} {old:<12.4f} {new:<12.4f} {improvement:>+11.2f}%")
EOF
```

---

## 🎨 生成視覺化報告

### 訓練曲線
```bash
# 使用 Python 生成訓練曲線對比
python3 << 'EOF'
import matplotlib.pyplot as plt
import pandas as pd

models = ['yolov5mc_backbone_v1', 'yolov5mlc_backbone_v1']
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for idx, model in enumerate(models):
    df = pd.read_csv(f'yolov5c/runs/{model}/results.csv')
    axes[idx].plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
    axes[idx].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Epoch')
    axes[idx].set_ylabel('mAP')
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.savefig('new_training_curves.png', dpi=150)
print("✓ Saved: new_training_curves.png")
EOF
```

---

## ❓ 常見問題

### Q1: 如何確認權重文件正確下載？
```bash
# 檢查文件大小
ls -lh yolov5*.pt

# 應該看到：
# yolov5s.pt: ~14M
# yolov5m.pt: ~41M
# yolov5l.pt: ~90M
```

### Q2: 訓練中斷怎麼辦？
```bash
# YOLOv5 會自動從最後一個 checkpoint 恢復
# 只需重新運行相同的腳本即可
bash yolov5mcbackbone.sh
```

### Q3: 記憶體不足 (OOM) 怎麼辦？
修改腳本中的 batch size：
```bash
# 編輯腳本，將 --batch-size 128 改為
--batch-size 64  # 或 32
```

### Q4: 如何只重新訓練某個 fold？
```bash
# 例如只訓練 V1
cd /work/jonchang3909/yolov5test/yolov5c/
python train.py \
    --data ../regurgitationV1/data.yaml \
    --cfg models/yolov5mc_classify_backbone.yaml \
    --weights yolov5m.pt \
    --epochs 300 \
    --batch-size 128 \
    --imgsz 416 \
    --name yolov5mc_backbone_v1 \
    --cache --nosave --patience 0 \
    --hyp data/hyps/hyp.default.yaml
```

---

## 📋 檢查清單

### 訓練前
- [ ] 已下載所有需要的權重文件
- [ ] 已清理資料集快取
- [ ] 已確認腳本包含 `--weights` 參數
- [ ] GPU 可用且記憶體充足

### 訓練中
- [ ] 監控 GPU 使用率
- [ ] 查看訓練日誌，確認無錯誤
- [ ] 記錄訓練開始時間

### 訓練後
- [ ] 提取並比較新舊結果
- [ ] 生成視覺化報告
- [ ] 更新架構比較文檔
- [ ] 保存重要發現

---

## 🚀 一鍵執行（完整流程）

```bash
#!/bin/bash
# 完整重新訓練流程

cd /work/jonchang3909/yolov5test/yolov5c/

echo "=== Step 1: Download Weights ==="
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt

echo "=== Step 2: Clean Cache ==="
for DATASET in regurgitationV{1..5}; do
    for SPLIT in train valid test; do
        find "../${DATASET}/${SPLIT}/labels/" -name "*.cache*" -delete 2>/dev/null
    done
done

echo "=== Step 3: Start Training ==="
bash yolov5mcbackbone.sh

echo "=== Done! ==="
echo "Check results in: runs/yolov5mc_backbone_v{1..5}/"
```

---

**參考文檔**: `WEIGHTS_FIX_SUMMARY.md`  
**更新日期**: 2025-10-20

