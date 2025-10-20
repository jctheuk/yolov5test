# 訓練權重修復總結

## 🔍 問題診斷

### 發現的問題
所有 YOLOv5 聯合訓練腳本（Detection + Classification）都缺少 `--weights` 參數，導致：

1. **Medium (yolov5mc) 模型** 未使用 `yolov5m.pt` 預訓練權重
2. **Medium-Large (yolov5mlc) 模型** 未使用 `yolov5l.pt` 預訓練權重（更適合其架構）
3. **Large (yolov5lc) 模型** 未使用 `yolov5l.pt` 預訓練權重
4. **Small (yolov5sc) 模型** 可能運氣好默認使用了 `yolov5s.pt`，但不明確

### 影響
- M 和 L 模型可能從零開始訓練或使用錯誤的權重
- 性能顯著下降（如架構比較報告所示）
- 無法進行公平的模型比較

---

## ✅ 已修復的腳本

### Small Model (yolov5sc) - 添加 `--weights yolov5s.pt`
- ✅ `yolov5c/yolov5scbackbone.sh`
- ✅ `yolov5c/yolov5sc_p3.sh`
- ✅ `yolov5c/yolov5sc_p4.sh`
- ✅ `yolov5c/yolov5sc_p5.sh`

### Medium Model (yolov5mc) - 添加 `--weights yolov5m.pt`
- ✅ `yolov5c/yolov5mcbackbone.sh`
- ✅ `yolov5c/yolov5mc_p3.sh`
- ✅ `yolov5c/yolov5mc_p4.sh`
- ✅ `yolov5c/yolov5mc_p5.sh`

### Medium-Large Model (yolov5mlc) - 添加 `--weights yolov5l.pt`
- ✅ `yolov5c/yolov5mlcbackbone.sh`
- ✅ `yolov5c/yolov5mlc_p3.sh`
- ✅ `yolov5c/yolov5mlc_p4.sh`
- ✅ `yolov5c/yolov5mlc_p5.sh`

### Large Model (yolov5lc) - 添加 `--weights yolov5l.pt`
- ✅ `yolov5c/yolov5lcbackbone.sh`
- ✅ `yolov5c/yolov5lc_p3.sh`
- ✅ `yolov5c/yolov5lc_p4.sh`
- ✅ `yolov5c/yolov5lc_p5.sh`

**總計：16 個腳本已修復**

---

## 📋 修改示例

### Before (錯誤)
```bash
python train.py \
    --data ../regurgitationV1/data.yaml \
    --cfg models/yolov5mc_classify_backbone.yaml \
    --epochs 300 \
    --batch-size 128 \
    --imgsz 416 \
    --name yolov5mc_backbone_v1 \
    --cache --nosave --patience 0
```

### After (正確)
```bash
python train.py \
    --data ../regurgitationV1/data.yaml \
    --cfg models/yolov5mc_classify_backbone.yaml \
    --weights yolov5m.pt \  # ← 添加了正確的預訓練權重
    --epochs 300 \
    --batch-size 128 \
    --imgsz 416 \
    --name yolov5mc_backbone_v1 \
    --cache --nosave --patience 0
```

---

## 🎯 正確的權重映射表

| 模型架構 | 應使用的預訓練權重 | 參數量 | 說明 |
|---------|------------------|--------|------|
| **yolov5sc** | `yolov5s.pt` | ~7M | Small 架構 |
| **yolov5mc** | `yolov5m.pt` | ~21M | Medium 架構 |
| **yolov5mlc** | `yolov5l.pt` | ~46M | Medium-Large (depth=0.83, width=0.875) |
| **yolov5lc** | `yolov5l.pt` | ~46M | Large 架構 |

---

## 📥 下載預訓練權重

如果 TWCC 伺服器上沒有這些權重文件，請執行：

```bash
cd /work/jonchang3909/yolov5test/yolov5c/

# 下載 yolov5s.pt (如果沒有)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# 下載 yolov5m.pt (必需)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt

# 下載 yolov5l.pt (必需)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
```

### 確認權重文件存在
```bash
ls -lh yolov5*.pt
```

應該看到：
```
-rw-r--r-- 1 user user  14M yolov5s.pt
-rw-r--r-- 1 user user  41M yolov5m.pt
-rw-r--r-- 1 user user  90M yolov5l.pt
```

---

## 🚀 下一步行動

### 1. 清理舊的快取文件
在重新訓練之前，清除資料集快取：

```bash
# PowerShell (本地測試)
$DATASET = "regurgitationV1"
$sets = @("train", "valid", "test")
foreach ($d in $sets) {
  $labels = Join-Path (Join-Path $DATASET $d) "labels"
  Remove-Item -Path (Join-Path $labels "labels.cache*") -ErrorAction SilentlyContinue -Force
}
```

### 2. 重新訓練 M 和 L 模型
優先重新訓練表現不佳的模型：

```bash
# 在 TWCC 上執行
cd /work/jonchang3909/yolov5test/yolov5c/

# 重新訓練 yolov5mc_backbone (最重要)
bash yolov5mcbackbone.sh

# 重新訓練 yolov5mlc_backbone
bash yolov5mlcbackbone.sh

# 重新訓練 yolov5lc_backbone (如果需要)
bash yolov5lcbackbone.sh
```

### 3. 比較新舊結果

創建對比表格：

| 模型 | 舊 mAP@0.5 (錯誤權重) | 新 mAP@0.5 (正確權重) | 改善 |
|------|---------------------|---------------------|------|
| yolov5sc_backbone | 0.7945 | ??? | ??? |
| yolov5mc_backbone | 0.7488 | ??? | ??? |
| yolov5mlc_backbone | 0.7487 | ??? | ??? |

預期：
- **yolov5mc** 和 **yolov5mlc** 的性能應該顯著提升
- 可能接近或超過 **yolov5sc** 的性能

### 4. 更新架構比較文檔
訓練完成後，更新 `docs/ARCHITECTURE_COMPARISON_COMPREHENSIVE.md`

---

## 📊 預期改善

### 為什麼使用正確的權重很重要？

1. **遷移學習優勢**
   - 預訓練權重在 COCO 數據集上已經學到通用特徵
   - 減少訓練時間
   - 提高收斂速度和最終性能

2. **架構匹配**
   - `yolov5m.pt` 的層數和通道數與 yolov5mc/yolov5mlc 架構匹配
   - `yolov5l.pt` 的架構與 yolov5lc 匹配
   - 使用錯誤的權重會導致層不匹配或性能下降

3. **公平比較**
   - 所有模型都使用相同質量的預訓練起點
   - 可以準確評估模型架構的優劣

### 預期性能提升

| 模型 | 當前 mAP@0.5 | 預期 mAP@0.5 | 預期提升 |
|------|------------|------------|---------|
| yolov5mc | 0.7488 | 0.78-0.80 | +4-6% |
| yolov5mlc | 0.7487 | 0.78-0.80 | +4-6% |
| yolov5lc | - | 0.80+ | 新結果 |

---

## ⚠️ 重要提醒

### 訓練前檢查清單
- [ ] 確認所有權重文件已下載（yolov5s.pt, yolov5m.pt, yolov5l.pt）
- [ ] 清理資料集快取文件
- [ ] 確認 GPU 記憶體足夠（L 模型需要更多記憶體）
- [ ] 確認使用 `--patience 0` 關閉早停機制

### 訓練配置
所有修復後的腳本使用：
- **Epochs**: 300
- **Batch Size**: 128
- **Image Size**: 416
- **Patience**: 0 (關閉早停)
- **Hyp**: `data/hyps/hyp.default.yaml`

### 記憶體考量
如果遇到 OOM（記憶體不足）：

```bash
# 減少 batch size
--batch-size 64  # 對於 Medium 模型
--batch-size 32  # 對於 Large 模型
```

---

## 📝 相關文檔

- `docs/ARCHITECTURE_COMPARISON_COMPREHENSIVE.md` - 架構比較報告
- `.cursorrules` - 專案訓練規則
- `yolov5c/YOLOV5MLC_SHELL_SCRIPTS_README.md` - 腳本說明

---

## ✨ 總結

### 核心修復
✅ 為所有 16 個訓練腳本添加了正確的 `--weights` 參數

### 預期效果
🎯 M 和 L 模型的性能應該顯著提升，提供更公平的模型比較

### 下一步
🚀 重新訓練 M 和 L 模型，並比較新舊結果

---

**修復日期**: 2025-10-20  
**修復者**: AI Assistant  
**影響範圍**: 所有聯合訓練腳本（16 個文件）

