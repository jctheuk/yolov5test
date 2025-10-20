# Architecture Performance Comparison

## Complete Performance Table (V1-V5 Averaged)

### Detection + Classification Metrics

| Architecture | mAP@0.5 | mAP@0.5:0.95 | Det.Precision | Det.Recall | Cls.Accuracy | Cls.Precision | Cls.Recall | Cls.F1 | Versions |
|---|---|---|---|---|---|---|---|---|---|
| **yolov5mc_backbone** | 0.7488 | 0.2982 | 0.8231 | 0.7589 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5mc_p3** | 0.7330 | 0.2947 | 0.8322 | 0.7361 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5mc_p4** | 0.7326 | 0.2908 | 0.8195 | 0.7258 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5mc_p5** | 0.7246 | 0.2932 | 0.8163 | 0.7307 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
|---|---|---|---|---|---|---|---|---|---|
| **yolov5mlc_backbone** | 0.7487 | 0.2962 | 0.8310 | 0.7465 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5mlc_p3** | 0.7203 | 0.2716 | 0.8085 | 0.7258 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5mlc_p4** | 0.7306 | 0.2836 | 0.8237 | 0.7359 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5mlc_p5** | 0.7123 | 0.2842 | 0.8168 | 0.7072 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
|---|---|---|---|---|---|---|---|---|---|
| **yolov5sc_backbone** | 0.7945 | 0.3494 | 0.8514 | 0.7980 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5sc_p3** | 0.7654 | 0.3389 | 0.8387 | 0.7700 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5sc_p4** | 0.7657 | 0.3353 | 0.8449 | 0.7730 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |
| **yolov5sc_p5** | 0.7775 | 0.3463 | 0.8366 | 0.7675 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5/5 |

## Summary Statistics

### 🏆 Best Performers

- **Best mAP@0.5**: yolov5sc_backbone (0.7945)
- **Best mAP@0.5:0.95**: yolov5sc_backbone (0.3494)
- **Best Classification Accuracy**: yolov5mc_backbone (0.0000)
- **Best Classification F1**: yolov5mc_backbone (0.0000)

### 📊 Model-wise Averages

| Model | Avg mAP@0.5 | Avg Cls.Accuracy | Avg Cls.F1 |
|-------|-------------|------------------|------------|
| **yolov5sc** | 0.7758 | 0.0000 | 0.0000 |
| **yolov5mc** | 0.7348 | 0.0000 | 0.0000 |
| **yolov5mlc** | 0.7280 | 0.0000 | 0.0000 |

### 🔧 Configuration Comparison

| Config | Avg mAP@0.5 | Avg Cls.Accuracy | Count |
|--------|-------------|------------------|-------|
| **backbone** | 0.7640 | 0.0000 | 3 |
| **p3** | 0.7396 | 0.0000 | 3 |
| **p4** | 0.7430 | 0.0000 | 3 |
| **p5** | 0.7381 | 0.0000 | 3 |

## Notes

- **Versions**: Number of dataset versions (V1-V5) successfully processed
- **Detection Metrics**: Overall detection performance across all classes (AR, MR, PR, TR)
- **Classification Metrics**: Overall classification performance across all views (A4C, PLAX, PSAX)
- All values are averaged across V1-V5 datasets

## Per-Class Metrics

⚠️ **Detection per-class metrics (AR, MR, PR, TR)** require running validation.
⚠️ **Classification per-class metrics (A4C, PLAX, PSAX)** require validation with confusion matrix.

To extract per-class detection metrics, run:
```bash
cd yolov5c
python val.py --weights "thesis results/yolov5sc_backbone_v1/weights/last.pt" \
    --data "../Regurgitation-YOLODataset-1/data.yaml" \
    --batch-size 32 --img 416 --task test --verbose
```
