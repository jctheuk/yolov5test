# YOLOv5 Original Classification Shell Scripts

## 📋 概述

本文檔說明如何使用 shell 腳本在服務器（TWCC）上批量訓練 YOLOv5 純分類模型。

**重要**：所有模型統一使用 **batch-size 128** 以確保公平對比。

---

## 📁 腳本文件列表

| 腳本文件 | 模型 | 預訓練 | Batch Size | 說明 |
|---------|------|--------|-----------|------|
| `yolov5original_classify_s.sh` | YOLOv5S | ✅ yolov5s-cls.pt | **128** | Small 模型 |
| `yolov5original_classify_m.sh` | YOLOv5M | ✅ yolov5m-cls.pt | **128** | Medium 模型 |
| `yolov5original_classify_l.sh` | YOLOv5L | ✅ yolov5l-cls.pt | **128** | Large 模型 |

---

## 🚀 使用方法

### 1. 上傳文件到服務器

將以下文件上傳到 TWCC 服務器：
```bash
/work/jonchang3909/yolov5test/
├── yolov5original/
│   └── classify/train.py
├── regurgitationV1_classify/
├── regurgitationV2_classify/
├── regurgitationV3_classify/
├── regurgitationV4_classify/
├── regurgitationV5_classify/
├── yolov5original_classify_s.sh
├── yolov5original_classify_m.sh
└── yolov5original_classify_l.sh
```

### 2. 給予執行權限

```bash
chmod +x yolov5original_classify_s.sh
chmod +x yolov5original_classify_m.sh
chmod +x yolov5original_classify_l.sh
```

### 3. 執行訓練

#### 選項 A: 單獨執行每個命令

```bash
# 訓練 Small 模型的 V1 數據集
cd /work/jonchang3909/yolov5test/yolov5original/ && \
sudo apt-get update && sudo apt-get install libgl1 -y && \
sudo pip install pandas && sudo pip install seaborn && \
python classify/train.py \
    --data ../regurgitationV1_classify \
    --model yolov5s-cls.pt \
    --epochs 300 \
    --batch-size 128 \
    --img 416 \
    --name classifys_v1 \
    --cache \
    --nosave
```

#### 選項 B: 使用 tmux 批量執行

```bash
# 開啟 tmux session
tmux new -s yolov5_classify_s

# 在 tmux 中執行（可以斷開連接）
bash yolov5original_classify_s.sh

# 斷開 tmux (不中斷訓練)
Ctrl+B, then D

# 重新連接
tmux attach -t yolov5_classify_s
```

#### 選項 C: 使用 nohup 後台執行

```bash
# 後台執行，輸出到日誌文件
nohup bash yolov5original_classify_s.sh > classify_s.log 2>&1 &
nohup bash yolov5original_classify_m.sh > classify_m.log 2>&1 &
nohup bash yolov5original_classify_l.sh > classify_l.log 2>&1 &

# 查看進程
ps aux | grep train.py

# 查看日誌
tail -f classify_s.log
```

---

## 📊 訓練配置詳情

### 統一參數（所有模型相同）

| 參數 | 值 | 說明 |
|------|-----|------|
| `--epochs` | 300 | 訓練 300 輪 |
| `--batch-size` | **128** | **統一使用 128** |
| `--img` | 416 | 圖像尺寸 416x416 |
| `--cache` | ✅ | 緩存圖像到記憶體 |
| `--nosave` | ✅ | 只保存最後和最佳權重 |

### 模型特定參數

#### YOLOv5S (Small)
```bash
--model yolov5s-cls.pt
--batch-size 128
```
- **預訓練**: ImageNet ✅
- **參數量**: ~5M
- **訓練時間**: ~5 小時/300 epochs
- **GPU 記憶體**: ~6-8GB

#### YOLOv5M (Medium)
```bash
--model yolov5m-cls.pt
--batch-size 128
```
- **預訓練**: ImageNet ✅
- **參數量**: ~12M
- **訓練時間**: ~10 小時/300 epochs
- **GPU 記憶體**: ~10-12GB ⚠️

#### YOLOv5L (Large)
```bash
--model yolov5l-cls.pt
--batch-size 128
```
- **預訓練**: ImageNet ✅
- **參數量**: ~25M
- **訓練時間**: ~20 小時/300 epochs
- **GPU 記憶體**: ~16-20GB ⚠️⚠️

---

## ⚠️ 統一 Batch Size 的優缺點

### ✅ 優點

1. **公平對比** - 所有模型使用相同的 batch size
2. **一致的訓練動態** - 相同的梯度累積行為
3. **結果可比性** - 消除 batch size 對結果的影響
4. **簡化實驗** - 無需為每個模型調整參數

### ⚠️ 注意事項

1. **GPU 記憶體需求高**
   - Medium 模型需要 ~10-12GB GPU 記憶體
   - Large 模型需要 ~16-20GB GPU 記憶體

2. **如果出現 OOM (記憶體溢出)**
   ```bash
   # 減少 batch size
   --batch-size 64   # Medium 模型降級方案
   --batch-size 32   # Large 模型降級方案
   ```

3. **建議的 GPU 配置**
   - Small: 8GB+ GPU (GTX 1080, RTX 2070)
   - Medium: 12GB+ GPU (RTX 3060, RTX 2080 Ti)
   - Large: 16GB+ GPU (RTX 3090, V100, A100)

---

## 🗂️ 輸出結果位置

訓練結果保存在：
```
yolov5original/runs/train-cls/
├── classifys_v1/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.csv
│   ├── train_images.jpg
│   └── test_images.jpg
├── classifys_v2/
├── classifym_v1/
└── classifyl_v1/
```

---

## 📈 監控訓練進度

### 方法 1: 查看 results.csv
```bash
cd yolov5original/runs/train-cls/classifys_v1
tail -f results.csv
```

### 方法 2: 查看 GPU 使用率
```bash
watch -n 1 nvidia-smi
```

### 方法 3: 檢查記憶體使用
```bash
# 監控 GPU 記憶體，確保不會 OOM
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

---

## 🔄 與 yolov5c 對比

### 數據集格式差異

| 特性 | yolov5original (純分類) | yolov5c (聯合訓練) |
|------|----------------------|-------------------|
| **數據格式** | 文件夾分類格式 | YOLO 格式（bbox + class）|
| **數據集** | regurgitationV*_classify | regurgitationV* |
| **標註需求** | 僅需類別 | 需要 bbox + 類別 |
| **Batch Size** | 128（統一） | 視模型大小而定 |

---

## 💡 實驗建議

### 階段 1: 快速驗證（Small 模型）
```bash
# 只訓練 V1，驗證流程和 GPU 記憶體
bash yolov5original_classify_s.sh  # 執行第一個命令
```

### 階段 2: 測試 GPU 記憶體（Medium 模型）
```bash
# 測試 M 模型是否能用 batch-size 128
# 如果 OOM，需要降低 batch size
bash yolov5original_classify_m.sh
```

### 階段 3: 完整實驗（如果 GPU 記憶體足夠）
```bash
# 使用 tmux 同時運行
tmux new -s classify_s
bash yolov5original_classify_s.sh
# Ctrl+B, D

tmux new -s classify_m
bash yolov5original_classify_m.sh
# Ctrl+B, D

tmux new -s classify_l
bash yolov5original_classify_l.sh
# Ctrl+B, D
```

---

## 🔧 記憶體不足時的解決方案

### 選項 1: 降低 Batch Size（推薦）

編輯對應的 .sh 文件：
```bash
# Medium 模型
--batch-size 128  →  --batch-size 64

# Large 模型
--batch-size 128  →  --batch-size 32
```

### 選項 2: 使用梯度累積（保持等效 batch size）

```bash
# 實現等效 batch-size 128
python classify/train.py \
    --batch-size 64 \
    --accumulate 2  # 64 * 2 = 128
```

### 選項 3: 使用更小的圖像尺寸

```bash
--img 416  →  --img 320  # 減少記憶體使用
```

---

## 📊 模型對比總結

| 模型 | 預訓練 | Batch Size | 參數量 | 訓練時間 | GPU 記憶體 | 備註 |
|------|--------|-----------|--------|----------|-----------|------|
| **S** | ✅ | 128 | ~5M | ~5h | ~6-8GB | ✅ 安全 |
| **M** | ✅ | 128 | ~12M | ~10h | ~10-12GB | ⚠️ 需 12GB+ |
| **L** | ✅ | 128 | ~25M | ~20h | ~16-20GB | ⚠️⚠️ 需 16GB+ |

---

## 📧 問題排查

### 問題 1: OOM (Out of Memory)
```bash
# 降低 batch size
--batch-size 64  # 或 32, 16
```

### 問題 2: 訓練太慢
```bash
# 確保使用 --cache 緩存圖像
--cache  # 已經在命令中
```

### 問題 3: 模型下載失敗
```bash
# 手動下載
cd yolov5original
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-cls.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-cls.pt
```

---

## ✅ 完成清單

- [ ] 上傳腳本文件到服務器
- [ ] 給予執行權限 (`chmod +x`)
- [ ] 確認數據集路徑正確
- [ ] **檢查 GPU 記憶體大小**（重要！）
- [ ] 測試 Small 模型（驗證流程）
- [ ] 測試 Medium 模型（確認記憶體足夠）
- [ ] 執行完整訓練
- [ ] 監控訓練進度
- [ ] 收集訓練結果

---

**結論**：統一使用 batch-size 128 可以確保公平對比，但需要足夠的 GPU 記憶體（建議 16GB+）。如果記憶體不足，請降低 Medium 和 Large 模型的 batch size。🚀
