# Original YOLOv5 Dataset Conversion Summary

## 📋 Overview
成功將 5 個 YOLOv5c 聯合訓練資料集轉換為適用於原版 YOLOv5 的純檢測和純分類資料集。

## 🎯 Created Datasets

### Detection Datasets (檢測資料集)
適用於原版 YOLOv5 物體檢測訓練：

- **regurgitationV1-Detection/** - 1,484 個檢測標註
- **regurgitationV2-Detection/** - 1,484 個檢測標註
- **regurgitationV3-Detection/** - 1,484 個檢測標註
- **regurgitationV4-Detection/** - 1,484 個檢測標註
- **regurgitationV5-Detection/** - 1,484 個檢測標註

#### Detection Dataset Features:
- **格式**: 純 YOLO 檢測格式 (單行標註)
- **類別**: 4 個檢測類別 ['AR', 'MR', 'PR', 'TR']
- **結構**: train/valid/test 標準 YOLO 目錄結構
- **配置**: 每個資料集包含 data.yaml 配置文件

### Classification Datasets (分類資料集)
適用於原版 YOLOv5 分類訓練：

- **regurgitationV1-Classification/** - 1,484 張分類圖片
- **regurgitationV2-Classification/** - 1,484 張分類圖片
- **regurgitationV3-Classification/** - 1,484 張分類圖片
- **regurgitationV4-Classification/** - 1,484 張分類圖片
- **regurgitationV5-Classification/** - 1,484 張分類圖片

#### Classification Dataset Features:
- **格式**: 資料夾結構分類格式
- **類別**: 3 個分類類別 ['A4C', 'PLAX', 'PSAX']
- **結構**: train/valid/test/[class_name] 標準分類目錄結構
- **分佈**: 
  - A4C: 481 張圖片
  - PLAX: 698 張圖片
  - PSAX: 305 張圖片

## 🔄 Conversion Process

### 1. Detection Dataset Creation
```
原始標註 (2行):
2 0.449125 0.360058 0.111540 0.135066  # 檢測標註
0 1 0                                  # 分類標註

轉換後 (1行):
2 0.449125 0.360058 0.111540 0.135066  # 僅保留檢測標註
```

### 2. Classification Dataset Creation
```
分類標註解析:
0 1 0 → [A4C, PSAX, PLAX] → 選擇 A4C (index 0, value=1)

圖片移動:
原始位置: regurgitationV1/train/images/image.png
目標位置: regurgitationV1-Classification/train/A4C/image.png
```

## 📊 Verification Results

✅ **完整性驗證通過**
- 原始資料集：1,484 個標籤文件
- 檢測資料集：1,484 個標籤文件 (100% 匹配)
- 分類資料集：1,484 張圖片 (100% 匹配)

✅ **格式驗證通過**
- 檢測標註：從 2 行減少為 1 行
- 分類資料夾：正確創建 A4C/PLAX/PSAX 目錄結構
- 配置文件：所有檢測資料集包含 data.yaml

✅ **內容驗證通過**
- 檢測標註內容與原始文件第一行完全匹配
- 分類圖片根據標註正確分配到對應資料夾
- 各資料集間分佈一致

## 🚀 Usage Instructions

### For Detection Training (檢測訓練)
```bash
# 使用原版 YOLOv5 訓練檢測模型
cd yolov5original
python train.py --data ../regurgitationV1-Detection/data.yaml --epochs 50 --batch-size 16
```

### For Classification Training (分類訓練)
```bash
# 使用原版 YOLOv5 訓練分類模型
cd yolov5original
python classify/train.py --data ../regurgitationV1-Classification --epochs 50 --batch-size 32
```

## 📁 Directory Structure

```
├── regurgitationV1-Detection/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── regurgitationV1-Classification/
│   ├── train/
│   │   ├── A4C/
│   │   ├── PLAX/
│   │   └── PSAX/
│   ├── valid/
│   │   ├── A4C/
│   │   ├── PLAX/
│   │   └── PSAX/
│   └── test/
│       ├── A4C/
│       ├── PLAX/
│       └── PSAX/
```

## ✨ Key Benefits

1. **原版 YOLOv5 兼容性**: 完全符合原版 YOLOv5 的資料格式要求
2. **數據完整性**: 保留所有原始數據，無數據損失
3. **靈活訓練**: 可以分別訓練檢測和分類模型
4. **標準化格式**: 遵循 YOLO 和分類的標準資料集格式
5. **易於使用**: 包含完整的配置文件和標準目錄結構

## 🎉 Status: COMPLETED
所有 5 個原始資料集已成功轉換為 10 個新資料集（5個檢測 + 5個分類），準備用於原版 YOLOv5 訓練。

