# YOLOv5 模型通道數修正總結

**修正日期**: 2025-10-13  
**問題**: width_multiple 導致實際通道數與配置不符

---

## 🚨 **原始錯誤**

```
RuntimeError: Given groups=1, weight of size [1280, 512, 1, 1], 
expected input[1, 256, 16, 16] to have 512 channels, but got 256 channels instead
```

**原因**: 分類頭部期望的通道數未考慮 width_multiple 縮放

---

## 📊 **修正後的正確通道數配置**

### YOLOv5sc (width_multiple = 0.50)

| 配置 | 來源層 | 配置通道 | **實際通道** (×0.50) | 修正前 | 修正後 | 狀態 |
|------|--------|----------|---------------------|--------|--------|------|
| classify_backbone | Layer 9 (SPPF) | 1024 | **512** | 512 | 512 | ✅ 正確 |
| p3 | Layer 17 (P3) | 256 | **128** | 128 | 128 | ✅ 正確 |
| p4 | Layer 20 (P4) | 512 | **256** | 512 | **256** | ✅ **已修正** |
| p5 | Layer 23 (P5) | 1024 | **512** | 1024 | **512** | ✅ **已修正** |

### YOLOv5mc (width_multiple = 0.75)

| 配置 | 來源層 | 配置通道 | **實際通道** (×0.75) | 修正前 | 修正後 | 狀態 |
|------|--------|----------|---------------------|--------|--------|------|
| classify_backbone | Layer 9 (SPPF) | 1024 | **768** | 512 | **768** | ✅ **已修正** |
| p3 | Layer 17 (P3) | 256 | **192** | 128 | **192** | ✅ **已修正** |
| p4 | Layer 20 (P4) | 512 | **384** | 512 | **384** | ✅ **已修正** |
| p5 | Layer 23 (P5) | 1024 | **768** | 1024 | **768** | ✅ **已修正** |

### YOLOv5lc (width_multiple = 1.0)

| 配置 | 來源層 | 配置通道 | **實際通道** (×1.0) | 修正前 | 修正後 | 狀態 |
|------|--------|----------|---------------------|--------|--------|------|
| classify_backbone | Layer 9 (SPPF) | 1024 | **1024** | 512 | **1024** | ✅ **已修正** |
| p3 | Layer 17 (P3) | 256 | **256** | 128 | **256** | ✅ **已修正** |
| p4 | Layer 20 (P4) | 512 | **512** | 512 | 512 | ✅ 正確 |
| p5 | Layer 23 (P5) | 1024 | **1024** | 1024 | 1024 | ✅ 正確 |

---

## 🔧 **修正的文件**

### YOLOv5sc 模型
- ✅ `models/yolov5sc_p4.yaml` - 修正 P4 通道數: 512 → **256**
- ✅ `models/yolov5sc_p5.yaml` - 修正 P5 通道數: 1024 → **512**

### YOLOv5mc 模型
- ✅ `models/yolov5mc_classify_backbone.yaml` - 修正 Backbone 通道數: 512 → **768**
- ✅ `models/yolov5mc_p3.yaml` - 修正 P3 通道數: 128 → **192**
- ✅ `models/yolov5mc_p4.yaml` - 修正 P4 通道數: 512 → **384**
- ✅ `models/yolov5mc_p5.yaml` - 修正 P5 通道數: 1024 → **768**

### YOLOv5lc 模型
- ✅ `models/yolov5lc_classify_backbone.yaml` - 修正 Backbone 通道數: 512 → **1024**
- ✅ `models/yolov5lc_p3.yaml` - 修正 P3 通道數: 128 → **256**

---

## 🎯 **通道數計算公式**

```python
實際通道數 = 配置通道數 × width_multiple
```

### 範例
- **YOLOv5sc** (width=0.50): 512 channels × 0.50 = **256 channels**
- **YOLOv5mc** (width=0.75): 512 channels × 0.75 = **384 channels**  
- **YOLOv5lc** (width=1.00): 512 channels × 1.00 = **512 channels**

---

## ✅ **驗證檢查清單**

- [x] YOLOv5sc_p4 - 256 通道
- [x] YOLOv5sc_p5 - 512 通道
- [x] YOLOv5mc_classify_backbone - 768 通道
- [x] YOLOv5mc_p3 - 192 通道
- [x] YOLOv5mc_p4 - 384 通道
- [x] YOLOv5mc_p5 - 768 通道
- [x] YOLOv5lc_classify_backbone - 1024 通道
- [x] YOLOv5lc_p3 - 256 通道

---

## 🚀 **現在可以安全執行**

所有12個實驗腳本現在都可以正常運行：

```bash
# 在不同容器中平行執行
./yolov5scbackbone.sh
./yolov5sc_p3.sh
./yolov5sc_p4.sh  # ✅ 已修正
./yolov5sc_p5.sh  # ✅ 已修正

./yolov5mcbackbone.sh  # ✅ 已修正
./yolov5mc_p3.sh       # ✅ 已修正
./yolov5mc_p4.sh       # ✅ 已修正
./yolov5mc_p5.sh       # ✅ 已修正

./yolov5lcbackbone.sh  # ✅ 已修正
./yolov5lc_p3.sh       # ✅ 已修正
./yolov5lc_p4.sh
./yolov5lc_p5.sh
```

---

## 📝 **重要提醒**

1. **清除舊快取**: 執行前清除 `labels*.cache` 檔案
2. **使用不同容器**: 避免快取衝突和資源競爭
3. **監控 GPU 記憶體**: lc 模型需要更多記憶體
4. **batch_size 調整**: 如記憶體不足可降低 batch size

---

**🎉 所有配置已修正完成，可以開始大規模實驗了！**

