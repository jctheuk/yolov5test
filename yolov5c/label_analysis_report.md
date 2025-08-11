# YOLOv5 標註分析報告

## 數據配置
- 檢測類別: ['AR', 'MR', 'PR', 'TR']
- 分類類別: ['PSAX', 'PLAX', 'A4C']
- 訓練路徑: ../Regurgitation-YOLODataset-Detection/train/images
- 驗證路徑: ../Regurgitation-YOLODataset-Detection/valid/images

## 問題診斷
1. **分類標註缺失**: 所有圖像都沒有分類標註
2. **聯合訓練失敗**: YOLOv5sc 需要檢測和分類標註
3. **性能差**: 分類任務無法學習

## 解決方案
1. 運行 `python fix_classification_labels.py` 修復分類標註
2. 重新訓練模型
3. 驗證性能改善
