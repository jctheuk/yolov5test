# K-Fold Training Summary - 20251013_143022

| Dataset | Start Time | End Time | Duration (hrs) | Status | mAP50 |
|---------|------------|----------|----------------|--------|-------|
| regurgitationV1 | 14:30:22 | 15:15:47 | 0.76 | ✅ Success | 0.847 |
| regurgitationV2 | 15:15:50 | 16:02:18 | 0.77 | ✅ Success | 0.834 |
| regurgitationV3 | 16:02:21 | 16:47:33 | 0.75 | ✅ Success | 0.851 |
| regurgitationV4 | 16:47:36 | 17:34:12 | 0.78 | ✅ Success | 0.829 |
| regurgitationV5 | 17:34:15 | 18:19:47 | 0.76 | ✅ Success | 0.843 |

## Summary Statistics
- **Total Training Time**: 3.82 hours
- **Average per Fold**: 0.76 hours (45.6 minutes)
- **Successful Folds**: 5/5
- **Failed Folds**: 0/5

## Individual Results
- **regurgitationV1**: 0.76h - Success - mAP50: 0.847
- **regurgitationV2**: 0.77h - Success - mAP50: 0.834
- **regurgitationV3**: 0.75h - Success - mAP50: 0.851
- **regurgitationV4**: 0.78h - Success - mAP50: 0.829
- **regurgitationV5**: 0.76h - Success - mAP50: 0.843

## K-Fold Cross-Validation Results
- **Mean mAP50**: 0.841 ± 0.009
- **Best Fold**: V3 (mAP50: 0.851)
- **Most Consistent**: All folds within 2.2% variance
- **Total Training Time**: 3.82 hours
- **Average per Fold**: 45.6 minutes

## Recommendations
- Model shows consistent performance across all folds
- Low variance indicates good generalization
- Ready for production deployment
