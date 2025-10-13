# V2-V5 資料集格式轉換完成報告

## 轉換概要

✅ **成功重新建立 regurgitationV2-V5 資料集**

從分割格式（多邊形）轉換為檢測格式（邊界框），保持分類功能完整。

## 資料集統計

| 資料集 | Train | Valid | Test | 總計 |
|--------|-------|-------|------|------|
| V2 | 1,013 | 179 | 293 | 1,485 |
| V3 | 1,010 | 180 | 295 | 1,485 |
| V4 | 1,006 | 181 | 298 | 1,485 |
| V5 | 1,010 | 182 | 293 | 1,485 |

## 資料格式

每個標籤檔案包含：
```
2 0.449125 0.360058 0.111540 0.135066   # 檢測：class_id x_center y_center width height
0 1 0                                    # 分類：A4C PSAX PLAX 的 one-hot 編碼
```

## 配置特點

✅ **聯合訓練支援**：每個 data.yaml 都配置為支持檢測和分類聯合訓練
✅ **格式統一**：所有資料集使用一致的 V1 格式（5,936 個檔案已轉換）
✅ **快取清理**：所有資料集的快取檔案已清理，確保訓練時重新建立
✅ **醫學圖像優化**：保持原始特徵，適合醫學診斷需求

## 檢測類別（4類）
- AR (Aortic Regurgitation)
- MR (Mitral Regurgitation) 
- PR (Pulmonary Regurgitation)
- TR (Tricuspid Regurgitation)

## 分類類別（3類）
- A4C (Apical Four Chamber)
- PSAX (Parasternal Short Axis)
- PLAX (Parasternal Long Axis)

## 準備訓練

資料集已準備就緒，可用於 YOLOv5WithClassification 聯合訓練：

```bash
# 訓練範例（遵循專案規則）
python train.py \
    --data regurgitationV2/data.yaml \
    --epochs 50 \
    --batch-size 16 \
    --device auto \
    --patience 0
```

## 重要規則遵循

✅ **聯合訓練必須啟用**：所有資料集都保持分類功能啟用
✅ **早停機制關閉**：使用 --patience 0 確保完整訓練
✅ **醫學圖像保護**：避免過度數據擴增，保持診斷準確性
✅ **快取清理**：每次訓練前自動重建快取檔案

---

**狀態**：✅ 完成 - 所有 V2-V5 資料集已成功轉換並準備用於聯合訓練
