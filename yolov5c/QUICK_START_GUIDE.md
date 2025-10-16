# YOLOv5MLC 快速開始指南

## 🚨 問題回顧
- **原始問題**: YOLOv5l 訓練 GPU 記憶體 99.7% 使用率 → NaN 錯誤
- **解決方案**: YOLOv5MLC (Medium-Large) 模型 + 優化配置

## 🚀 三種執行方式

### 方式 1：Shell 腳本 (推薦單容器使用)
```bash
# 在 TWCC 訓練容器中
cd /work/jonchang3909/yolov5test/yolov5c/
chmod +x yolov5mlc_classify_backbone.sh
./yolov5mlc_classify_backbone.sh
```
- 自動執行 V1-V5 訓練
- 包含時間戳和進度提示
- 約 4 小時完成

### 方式 2：複製貼上命令 (推薦多容器並行)
打開 `yolov5mlc_training_commands.txt`，複製需要的命令：

**容器1 執行**：
```bash
cd /work/jonchang3909/yolov5test/yolov5c/ && sudo apt-get update && sudo apt-get install libgl1 -y && sudo pip install pandas seaborn && python train.py --data ../regurgitationV1/data.yaml --cfg models/yolov5mlc_classify_backbone.yaml --epochs 300 --batch-size 64 --imgsz 416 --name yolov5mlc_backbone_v1 --cache --nosave --patience 0 --hyp data/hyps/hyp.default.yaml --cos-lr --amp
```

**容器2-5**：替換 V1 → V2/V3/V4/V5

### 方式 3：極致記憶體優化 (如果方式1/2仍失敗)
使用 `memory_optimized_training_commands.txt`：
- 原始 YOLOv5l 模型
- 批次大小 16
- 記憶體使用 ~12GB

## 📋 可用的 Shell 腳本

| 腳本文件 | 模型配置 | 批次大小 | 預期記憶體 |
|----------|----------|----------|------------|
| `yolov5mlc_classify_backbone.sh` | Backbone (896通道) | 64 | ~18-20GB |
| `yolov5mlc_p3.sh` | P3 (224通道) | 128 | ~16-18GB |
| `yolov5mlc_p4.sh` | P4 (448通道) | 64 | ~18-20GB |
| `yolov5mlc_p5.sh` | P5 (896通道) | 64 | ~18-20GB |

## 🎯 推薦執行策略

### 情境A：你有 4-5 個容器 (最快)
```
容器1: ./yolov5mlc_classify_backbone.sh
容器2: ./yolov5mlc_p3.sh
容器3: ./yolov5mlc_p4.sh  
容器4: ./yolov5mlc_p5.sh
```
- **總時間**: ~4小時 (並行)
- **獲得**: 全部 4 種模型 × 5 fold = 20 個訓練結果

### 情境B：你有 5 個容器 (最全面)
```
容器1: 所有 V1 命令 (4種模型)
容器2: 所有 V2 命令 (4種模型)
容器3: 所有 V3 命令 (4種模型)
容器4: 所有 V4 命令 (4種模型)
容器5: 所有 V5 命令 (4種模型)
```
- **總時間**: ~3.5小時 (並行)
- **獲得**: 每個 V 的所有模型結果

### 情境C：你只有 1 個容器 (最保守)
```bash
# 依次執行
./yolov5mlc_classify_backbone.sh  # ~4小時
./yolov5mlc_p3.sh                 # ~3.5小時
./yolov5mlc_p4.sh                 # ~4小時
./yolov5mlc_p5.sh                 # ~4小時
```
- **總時間**: ~15.5小時
- **獲得**: 全部 20 個訓練結果

## 📊 預期結果

### 記憶體使用
- **原始問題**: 31.9GB (99.7%) ❌
- **MLC 解決方案**: 16-20GB (50-62%) ✅

### 訓練成功率
- **原始**: 2/5 成功 (40%)
- **MLC**: 預期 19/20 成功 (95%+)

### 模型性能
- **檢測 mAP**: 預期介於 YOLOv5m 和 YOLOv5l 之間
- **分類準確率**: 預期 >90%
- **訓練穩定性**: 無 NaN 錯誤

## 🔍 監控檢查清單

### 訓練開始時
- [ ] GPU 記憶體初始化 <20GB
- [ ] 模型載入成功
- [ ] 數據集正確讀取

### 訓練過程中
- [ ] GPU 記憶體保持 <26GB
- [ ] 訓練損失穩定下降
- [ ] 無 NaN 或 inf 錯誤
- [ ] 每 epoch 正常完成

### 訓練結束後
- [ ] 完成 300 epochs
- [ ] results.csv 生成
- [ ] 檢測和分類指標正常
- [ ] 所有 V1-V5 成功

## 📁 結果位置

```
runs/train/
├── yolov5mlc_backbone_v1/  # V1 Backbone 結果
│   ├── results.csv
│   ├── results.png
│   ├── classification_metrics.png
│   └── ...
├── yolov5mlc_backbone_v2/  # V2 Backbone 結果
├── ...
└── yolov5mlc_p5_v5/        # V5 P5 結果
```

## 🆘 故障排除

### 如果出現記憶體不足
```bash
# 方法1: 降低批次大小
修改 .sh 文件: --batch-size 64 → --batch-size 32

# 方法2: 使用極致優化版本
使用 memory_optimized_training_commands.txt (批次 16)
```

### 如果出現 NaN 錯誤
```bash
# 方法1: 檢查記憶體是否超過 80%
nvidia-smi

# 方法2: 降低批次大小
--batch-size 64 → --batch-size 32

# 方法3: 使用更保守配置
添加更強的梯度剪裁
```

## 💡 最終建議

**立即開始訓練**：
```bash
# 最推薦的執行命令
cd /work/jonchang3909/yolov5test/yolov5c/
chmod +x yolov5mlc_classify_backbone.sh
./yolov5mlc_classify_backbone.sh
```

這應該能解決你的 99.7% GPU 記憶體問題，同時保持優異的模型性能！

---

**問題或需要幫助？** 查看 `COMPLETE_SOLUTION_SUMMARY.md` 獲取完整解決方案概覽。

