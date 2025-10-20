# 權重配置更新總結

## ✅ 最新更新（2025-10-20）

### MLC 模型權重變更
**從 `yolov5m.pt` 改為 `yolov5l.pt`**

由於 YOLOv5MLC 是 **Medium-Large** 架構（depth=0.83, width=0.875），使用 Large 預訓練權重更合適。

---

## 📊 最終權重配置表

| 模型架構 | 預訓練權重 | 參數量 | 說明 |
|---------|-----------|--------|------|
| **yolov5sc** | `yolov5s.pt` | ~7M | Small 架構 |
| **yolov5mc** | `yolov5m.pt` | ~21M | Medium 架構 |
| **yolov5mlc** | `yolov5l.pt` | ~46M | Medium-Large 架構 ✨ **已更新** |
| **yolov5lc** | `yolov5l.pt` | ~46M | Large 架構 |

---

## 🔧 已更新的腳本

### MLC 模型（已改用 yolov5l.pt）
- ✅ `yolov5c/yolov5mlcbackbone.sh` 
- ✅ `yolov5c/yolov5mlc_p3.sh`
- ✅ `yolov5c/yolov5mlc_p4.sh`
- ✅ `yolov5c/yolov5mlc_p5.sh`

### 所有其他模型腳本（已添加正確權重）
- ✅ `yolov5sc` (4 個腳本) → `yolov5s.pt`
- ✅ `yolov5mc` (4 個腳本) → `yolov5m.pt`
- ✅ `yolov5lc` (4 個腳本) → `yolov5l.pt`

**總計：16 個腳本已修復** ✨

---

## 🎯 為什麼 MLC 使用 Large 權重？

### 架構分析
YOLOv5MLC 的配置：
```yaml
depth_multiple: 0.83  # 介於 M (0.67) 和 L (1.0) 之間，更接近 L
width_multiple: 0.875 # 介於 M (0.75) 和 L (1.0) 之間
```

### 參數量比較
| 模型 | Depth | Width | 參數量 | 最適合的權重 |
|------|-------|-------|--------|-------------|
| YOLOv5m | 0.67 | 0.75 | 21M | yolov5m.pt |
| **YOLOv5mlc** | **0.83** | **0.875** | **~30-35M** | **yolov5l.pt** ✅ |
| YOLOv5l | 1.0 | 1.0 | 46M | yolov5l.pt |

### 結論
- MLC 的架構更接近 Large 而非 Medium
- 使用 `yolov5l.pt` 能提供更好的預訓練起點
- 預期性能提升更明顯

---

## 📥 確保權重文件存在

```bash
cd /work/jonchang3909/yolov5test/yolov5c/

# 檢查所需權重
ls -lh yolov5*.pt

# 如果缺少，下載：
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
wget -nc https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
```

---

## 🚀 驗證更新

### 檢查 MLC 腳本
```bash
cd yolov5c

# 確認所有 MLC 腳本都使用 yolov5l.pt
grep -n "weights yolov5l.pt" yolov5mlc*.sh

# 應該看到每個腳本有 5 行（V1-V5）：
# yolov5mlcbackbone.sh:9:... --weights yolov5l.pt --epochs ...
# yolov5mlcbackbone.sh:11:... --weights yolov5l.pt --epochs ...
# ...（共 20 行，4 個腳本 × 5 個 folds）
```

### 驗證所有模型配置
```bash
echo "=== Small Model (S) ==="
grep -l "weights yolov5s.pt" yolov5sc*.sh | wc -l
# 應該輸出: 4

echo "=== Medium Model (M) ==="
grep -l "weights yolov5m.pt" yolov5mc*.sh | wc -l
# 應該輸出: 4

echo "=== Large Model (L) and MLC ==="
grep -l "weights yolov5l.pt" yolov5mlc*.sh yolov5lc*.sh | wc -l
# 應該輸出: 8 (4 MLC + 4 LC)
```

---

## 📈 預期性能改善

### MLC 模型（使用 yolov5l.pt）
| 指標 | 舊結果 (錯誤/無權重) | 預期結果 (yolov5l.pt) | 改善 |
|------|-------------------|---------------------|------|
| mAP@0.5 | 0.7487 | 0.79-0.81 | +5-8% |
| mAP@0.5:0.95 | 0.2962 | 0.32-0.35 | +8-18% |

### MC 模型（使用 yolov5m.pt）
| 指標 | 舊結果 (錯誤/無權重) | 預期結果 (yolov5m.pt) | 改善 |
|------|-------------------|---------------------|------|
| mAP@0.5 | 0.7488 | 0.78-0.80 | +4-6% |
| mAP@0.5:0.95 | 0.2982 | 0.32-0.34 | +7-14% |

---

## ⚡ 快速重新訓練

### 優先訓練順序
```bash
cd /work/jonchang3909/yolov5test/yolov5c/

# 1. 訓練 MLC (預期最大改善)
bash yolov5mlcbackbone.sh

# 2. 訓練 MC
bash yolov5mcbackbone.sh

# 3. 如果時間允許，訓練 LC
bash yolov5lcbackbone.sh
```

### 訓練時間預估
| 模型 | 使用權重 | 每 Fold | 5 Folds | GPU |
|------|---------|--------|---------|-----|
| yolov5mc | yolov5m.pt | 2-3h | 10-15h | 1× RTX 3090/4090 |
| yolov5mlc | yolov5l.pt | 3-4h | 15-20h | 1× RTX 3090/4090 |
| yolov5lc | yolov5l.pt | 3.5-4.5h | 17-22h | 1× RTX 3090/4090 |

---

## 📊 結果比較模板

訓練完成後，使用此模板比較：

```python
import pandas as pd

# 舊結果（錯誤配置）
old = {
    'yolov5sc_backbone': 0.7945,
    'yolov5mc_backbone': 0.7488,
    'yolov5mlc_backbone': 0.7487,
}

# 新結果（從 CSV 提取）
new = {
    'yolov5mc_backbone': 0.0,   # TODO: 填入訓練結果
    'yolov5mlc_backbone': 0.0,  # TODO: 填入訓練結果
}

# 計算改善
for model in new.keys():
    if new[model] > 0:
        improvement = ((new[model] - old[model]) / old[model]) * 100
        print(f"{model}: {old[model]:.4f} → {new[model]:.4f} (+{improvement:.2f}%)")
```

---

## 🎯 成功指標

### 最低預期
- **MC**: mAP@0.5 ≥ 0.78 (+4%)
- **MLC**: mAP@0.5 ≥ 0.79 (+6%)

### 理想目標
- **MC**: mAP@0.5 ≥ 0.80 (+7%)
- **MLC**: mAP@0.5 ≥ 0.81 (+8%)

### 最佳情境
- **MLC** 超越 **SC** (0.7945)，成為最佳模型

---

## 📚 相關文檔

- `WEIGHTS_FIX_SUMMARY.md` - 完整權重修復總結
- `QUICK_START_RETRAINING.md` - 重新訓練快速指南
- `docs/ARCHITECTURE_COMPARISON_COMPREHENSIVE.md` - 架構比較報告

---

**最後更新**: 2025-10-20  
**關鍵變更**: MLC 模型改用 yolov5l.pt 權重 ✨

