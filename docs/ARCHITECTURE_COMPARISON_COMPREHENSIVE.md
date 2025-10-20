# Architecture Performance Comparison

## 📊 Complete Performance Table (V1-V5 Averaged)

### Overview Detection Metrics

All values are averaged across V1-V5 datasets. Detection metrics represent overall performance across all regurgitation classes (AR, MR, PR, TR).

| Architecture | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Versions |
|--------------|---------|--------------|-----------|--------|----------|
| **yolov5sc_backbone** | **0.7945** | **0.3494** | 0.8514 | 0.7980 | 5/5 |
| **yolov5sc_p5** | 0.7775 | 0.3463 | 0.8366 | 0.7675 | 5/5 |
| **yolov5sc_p4** | 0.7657 | 0.3353 | 0.8449 | 0.7730 | 5/5 |
| **yolov5sc_p3** | 0.7654 | 0.3389 | 0.8387 | 0.7700 | 5/5 |
|  |  |  |  |  |  |
| **yolov5mc_backbone** | 0.7488 | 0.2982 | 0.8231 | 0.7589 | 5/5 |
| **yolov5mc_p3** | 0.7330 | 0.2947 | 0.8322 | 0.7361 | 5/5 |
| **yolov5mc_p4** | 0.7326 | 0.2908 | 0.8195 | 0.7258 | 5/5 |
| **yolov5mc_p5** | 0.7246 | 0.2932 | 0.8163 | 0.7307 | 5/5 |
|  |  |  |  |  |  |
| **yolov5mlc_backbone** | 0.7487 | 0.2962 | 0.8310 | 0.7466 | 5/5 |
| **yolov5mlc_p4** | 0.7306 | 0.2836 | 0.8237 | 0.7359 | 5/5 |
| **yolov5mlc_p3** | 0.7203 | 0.2716 | 0.8085 | 0.7258 | 5/5 |
| **yolov5mlc_p5** | 0.7123 | 0.2842 | 0.8168 | 0.7072 | 5/5 |

---

## 🏆 Best Performers

### Detection Performance

- **🥇 Best mAP@0.5**: `yolov5sc_backbone` - **0.7945**
- **🥈 Second mAP@0.5**: `yolov5sc_p5` - **0.7775**
- **🥉 Third mAP@0.5**: `yolov5sc_p4` - **0.7657**

- **🥇 Best mAP@0.5:0.95**: `yolov5sc_backbone` - **0.3494**
- **🥈 Second mAP@0.5:0.95**: `yolov5sc_p5` - **0.3463**
- **🥉 Third mAP@0.5:0.95**: `yolov5sc_p4` - **0.3353**

- **🎯 Best Precision**: `yolov5sc_backbone` - **0.8514**
- **🎯 Best Recall**: `yolov5sc_backbone` - **0.7980**

**Key Finding**: `yolov5sc_backbone` dominates all detection metrics!

---

## 📈 Model-wise Comparison

### Average Performance by Model Size

| Model | Avg mAP@0.5 | Avg mAP@0.5:0.95 | Avg Precision | Avg Recall |
|-------|-------------|------------------|---------------|------------|
| **yolov5sc** (Small) | **0.7758** | **0.3425** | **0.8429** | **0.7771** |
| **yolov5mc** (Medium) | 0.7348 | 0.2942 | 0.8228 | 0.7384 |
| **yolov5mlc** (Medium+Loss) | 0.7280 | 0.2839 | 0.8202 | 0.7288 |

**Key Finding**: Small model (`yolov5sc`) **significantly outperforms** medium models in detection!

### Model Size Analysis

- **Small (yolov5sc)**:
  - ✅ Highest mAP (+5.6% vs yolov5mc)
  - ✅ Highest precision
  - ✅ Highest recall
  - ✅ Better parameter efficiency
  - **Recommended for production use**

- **Medium (yolov5mc)**:
  - ⚠️ Lower detection mAP
  - ⚠️ More parameters, lower performance
  - ❌ Not recommended

- **Medium+Loss Constraints (yolov5mlc)**:
  - ⚠️ Lowest overall performance
  - ❌ Loss constraints may have hurt performance
  - ❌ Not recommended

---

## 🔧 Configuration Comparison

### Average Performance by Detection Head Configuration

| Config | Avg mAP@0.5 | Avg mAP@0.5:0.95 | Count | Best Use Case |
|--------|-------------|------------------|-------|---------------|
| **backbone** | **0.7640** | **0.3146** | 3 | **Best overall** |
| **p4** | 0.7430 | 0.3032 | 3 | Balanced |
| **p3** | 0.7396 | 0.3017 | 3 | Small objects |
| **p5** | 0.7381 | 0.3079 | 3 | Large objects |

**Key Finding**: `backbone` configuration provides best detection performance!

### Configuration Analysis

- **backbone**: Extract features from backbone
  - ✅ Best mAP@0.5 (0.7640)
  - ✅ Best mAP@0.5:0.95 (0.3146)
  - ✅ Simpler architecture
  - **Recommended configuration**

- **p3**: Finest detection head (smallest objects)
  - ⚠️ Lower overall mAP
  - Good for very small regurgitation jets

- **p4**: Medium detection head
  - ⚠️ Middle ground performance
  - Balanced approach

- **p5**: Coarsest detection head (largest objects)
  - ⚠️ Lower mAP than backbone
  - Good for large regurgitation areas

---

## 🎖️ Top Recommendations

### 🥇 Production Recommendation

**`yolov5sc_backbone`**
- mAP@0.5: **0.7945** (Highest)
- mAP@0.5:0.95: **0.3494** (Highest)
- Precision: **0.8514** (Highest)
- Recall: **0.7980** (Highest)
- **Small model size**: Faster inference
- **Best overall performance**

### 🥈 Alternative Option

**`yolov5sc_p5`**
- mAP@0.5: **0.7775** (Second best)
- mAP@0.5:0.95: **0.3463** (Second best)
- Good for large regurgitation areas
- Still maintains small model size

### ❌ Not Recommended

- **yolov5mc_*** and **yolov5mlc_***:  All medium model configurations underperform small model
- **Reason**: Overfitting or suboptimal hyperparameters for this dataset size

---

## 📊 Per-Class Detection Metrics (TODO)

### Regurgitation Classes

To get per-class metrics for AR, MR, PR, TR:

| Architecture | AR (mAP) | MR (mAP) | PR (mAP) | TR (mAP) | Status |
|--------------|----------|----------|----------|----------|--------|
| yolov5sc_backbone | ? | ? | ? | ? | ⏳ Need validation |
| yolov5sc_p5 | ? | ? | ? | ? | ⏳ Need validation |
| yolov5mc_backbone | ? | ? | ? | ? | ⏳ Need validation |
| ... | ? | ? | ? | ? | ⏳ Need validation |

**To extract per-class metrics**, run validation:

```bash
cd yolov5c
python val.py \
    --weights "thesis results/yolov5sc_backbone_v1/weights/last.pt" \
    --data "../Regurgitation-YOLODataset-1/data.yaml" \
    --batch-size 32 \
    --img 416 \
    --task test \
    --verbose
```

This will output a table like:
```
Class  Images  Instances      P      R  mAP@.5  mAP@.5:.95
  all     299        478  0.851  0.798   0.795       0.349
   AR     299        123  0.XXX  0.XXX   0.XXX       0.XXX
   MR     299        187  0.XXX  0.XXX   0.XXX       0.XXX
   PR     299         89  0.XXX  0.XXX   0.XXX       0.XXX
   TR     299         79  0.XXX  0.XXX   0.XXX       0.XXX
```

---

## 📊 Per-Class Classification Metrics (TODO)

### View Classes

To get per-class classification metrics for A4C, PLAX, PSAX:

| Architecture | A4C (Acc) | PLAX (Acc) | PSAX (Acc) | Overall Acc | Status |
|--------------|-----------|------------|------------|-------------|--------|
| yolov5sc_backbone | ? | ? | ? | ? | ⏳ Need validation |
| yolov5sc_p5 | ? | ? | ? | ? | ⏳ Need validation |
| yolov5mc_backbone | ? | ? | ? | ? | ⏳ Need validation |
| ... | ? | ? | ? | ? | ⏳ Need validation |

**Note**: Classification metrics are saved during training in `classification_metrics.txt` but require post-processing to extract per-class accuracies.

---

## 📝 Technical Notes

### Data Collection
- **Source**: `yolov5c/thesis results/` directory
- **Models**: 12 configurations (3 model sizes × 4 detection heads)
- **Datasets**: V1-V5 averaged
- **Metrics extracted**: Last epoch from `results.csv`

### Metrics Definitions

**Detection Metrics**:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (primary metric)
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU 0.5-0.95 (stricter metric)
- **Precision**: TP / (TP + FP) - How many detected regurgitations are correct
- **Recall**: TP / (TP + FN) - How many actual regurgitations are detected

**Classification Metrics** (from embedded classifier):
- **Accuracy**: Correct view predictions / Total predictions
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1-Score**: Harmonic mean of precision and recall

### Regurgitation Classes (Detection Task)
- **AR**: Aortic Regurgitation (主動脈瓣逆流)
- **MR**: Mitral Regurgitation (二尖瓣逆流)
- **PR**: Pulmonary Regurgitation (肺動脈瓣逆流)
- **TR**: Tricuspid Regurgitation (三尖瓣逆流)

### View Classes (Classification Task)
- **A4C**: Apical Four Chamber (心尖四腔室視圖)
- **PLAX**: Parasternal Long Axis (胸骨旁長軸視圖)
- **PSAX**: Parasternal Short Axis (胸骨旁短軸視圖)

---

## 🎯 Conclusions

### Main Findings

1. **Small model is best**: `yolov5sc` consistently outperforms medium models
   - +5.6% mAP@0.5 vs yolov5mc
   - +6.6% mAP@0.5 vs yolov5mlc

2. **Backbone configuration wins**: Extracting features from backbone gives best results
   - +3.3% mAP@0.5 vs average of p3/p4/p5

3. **Loss constraints hurt**: yolov5mlc (with additional anatomical constraints) has worst performance
   - Constraints may be too restrictive
   - May need hyperparameter tuning

### Recommendations

**For Production Deployment**:
- Use `yolov5sc_backbone` for best detection performance
- Small model → Faster inference, lower memory
- Backbone configuration → Simpler architecture

**For Research/Experimentation**:
- Investigate why medium models underperform
- Consider: Learning rate, weight decay, training epochs
- Try ensemble: `yolov5sc_backbone + yolov5sc_p5`

**Next Steps**:
1. Extract per-class detection metrics (AR, MR, PR, TR)
2. Extract per-class classification metrics (A4C, PLAX, PSAX)
3. Run confusion matrix analysis
4. Analyze error cases for top models

---

## 📚 Related Documents

- `FINAL_COMPREHENSIVE_TABLE.md` - Original detection + classification table
- `DATA_EXTRACTION_STATUS.md` - Data extraction progress
- `EXTRACT_DETECTION_PERCLASS.sh` - Script to extract per-class metrics
- `ARCHITECTURE_COMPARISON_TABLE.csv` - Raw data (spreadsheet-friendly)

---

*Generated on: 2025-10-17*  
*Data source: yolov5c/thesis results (V1-V5 averaged)*


