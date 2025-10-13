# YOLOv5WithClassification 實驗腳本總覽

**創建日期**: 2025-10-13  
**目標**: 基於 TWCC.ai 的多模型、多配置 K-fold 交叉驗證實驗

---

## 🎯 實驗設計

### 模型大小 (3種)
- **YOLOv5sc** (Small-C): 較小模型，訓練快速
- **YOLOv5mc** (Medium-C): 中等模型，平衡性能和速度  
- **YOLOv5lc** (Large-C): 大型模型，最高性能

### 配置類型 (4種)
- **classify_backbone**: 分類骨幹網絡配置
- **p3**: P3 特徵金字塔配置
- **p4**: P4 特徵金字塔配置  
- **p5**: P5 特徵金字塔配置

### 總實驗數量: **3 × 4 = 12 種配置**
每種配置都訓練 V1-V5 (K-fold)，總共 **60 個訓練任務**

---

## 📁 創建的實驗腳本

| 腳本檔案 | 模型配置 | 批次大小 | 超參數 | 輸出名稱格式 |
|----------|----------|----------|---------|--------------|
| **yolov5scbackbone.sh** | yolov5sc_classify_backbone.yaml | 128 | hyp.default.yaml | yolov5sc_backbone_v1-v5 |
| **yolov5sc_p3.sh** | yolov5sc_p3.yaml | 128 | hyp.default.yaml | yolov5sc_p3_v1-v5 |
| **yolov5sc_p4.sh** | yolov5sc_p4.yaml | 128 | hyp.default.yaml | yolov5sc_p4_v1-v5 |
| **yolov5sc_p5.sh** | yolov5sc_p5.yaml | 128 | hyp.default.yaml | yolov5sc_p5_v1-v5 |
| **yolov5mc_classify_backbone.sh** | yolov5mc_classify_backbone.yaml | 128 | hyp.default.yaml | yolov5mc_backbone_v1-v5 |
| **yolov5mc_p3.sh** | yolov5mc_p3.yaml | 128 | hyp.default.yaml | yolov5mc_p3_v1-v5 |
| **yolov5mc_p4.sh** | yolov5mc_p4.yaml | 128 | hyp.default.yaml | yolov5mc_p4_v1-v5 |
| **yolov5mc_p5.sh** | yolov5mc_p5.yaml | 128 | hyp.default.yaml | yolov5mc_p5_v1-v5 |
| **yolov5lc_classify_backbone.sh** | yolov5lc_classify_backbone.yaml | 64 | hyp.default.yaml | yolov5lc_backbone_v1-v5 |
| **yolov5lc_p3.sh** | yolov5lc_p3.yaml | 64 | hyp.default.yaml | yolov5lc_p3_v1-v5 |
| **yolov5lc_p4.sh** | yolov5lc_p4.yaml | 64 | hyp.default.yaml | yolov5lc_p4_v1-v5 |
| **yolov5lc_p5.sh** | yolov5lc_p5.yaml | 64 | hyp.default.yaml | yolov5lc_p5_v1-v5 |

---

## ⚙️ 配置詳情

### 批次大小策略
- **YOLOv5sc/mc**: `batch_size=128` (小到中型模型)
- **YOLOv5lc**: `batch_size=64` (大型模型，減少GPU記憶體使用)

### 超參數選擇
- **所有模型**: 統一使用 `hyp.default.yaml` (一致的優化設定，無約束)

### 共同參數
- **epochs**: 300 (充分訓練)
- **imgsz**: 416 (適合醫學圖像)  
- **cache**: 啟用 (加速訓練)
- **nosave**: 啟用 (節省硬碟空間)
- **patience**: 0 (關閉早停，獲得完整訓練曲線)

---

## 🚀 使用方式

### 單一實驗執行
```bash
# 執行特定配置
chmod +x yolov5sc_p3.sh
./yolov5sc_p3.sh
```

### 批次實驗執行
```bash
# 執行所有 sc 配置
for script in yolov5sc*.sh; do
    echo "Running $script..."
    chmod +x "$script"
    ./"$script"
done

# 執行所有 mc 配置  
for script in yolov5mc*.sh; do
    echo "Running $script..."
    chmod +x "$script"
    ./"$script"
done

# 執行所有 lc 配置
for script in yolov5lc*.sh; do
    echo "Running $script..."
    chmod +x "$script" 
    ./"$script"
done
```

---

## 📊 預期結果

### 訓練輸出位置
所有模型將儲存在 `runs/train/` 下：
```
runs/train/
├── yolov5sc_backbone_v1/
├── yolov5sc_backbone_v2/
├── ...
├── yolov5mc_p3_v1/
├── yolov5mc_p3_v2/
├── ...
├── yolov5lc_p5_v5/
```

### 時間估算 (TWCC.ai GPU)
| 模型大小 | 每個fold時間 | V1-V5總時間 | 所有配置時間 |
|----------|--------------|-------------|--------------|
| **sc** | ~2.5 小時 | ~12.5 小時 | ~50 小時 (4配置) |
| **mc** | ~3.5 小時 | ~17.5 小時 | ~70 小時 (4配置) |
| **lc** | ~5.0 小時 | ~25 小時 | ~100 小時 (4配置) |

**總估算時間**: ~220 小時 (所有60個訓練任務)

---

## 🎯 實驗策略建議

### 階段式執行
1. **Phase 1**: 先執行所有 sc 配置 (較快，驗證設定)
2. **Phase 2**: 執行 mc 配置 (中等時間，平衡性能) 
3. **Phase 3**: 執行 lc 配置 (最長時間，最高性能)

### 優先順序建議
1. `yolov5sc_classify_backbone.sh` - 基準實驗
2. `yolov5mc_classify_backbone.sh` - 中等性能基準
3. `yolov5sc_p4.sh` - P4 配置比較
4. 其餘依需要執行

---

**🎉 12 個完整的實驗腳本已準備就緒，可在 TWCC.ai 上進行大規模 K-fold 交叉驗證！**
