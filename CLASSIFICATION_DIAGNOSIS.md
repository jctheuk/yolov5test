## Classification Performance Diagnosis (YOLOv5WithClassification)

Use this checklist to isolate why accuracy plateaus around ~40–50%.

### Symptoms
- Low top-1 (~40–50%), confusion matrix dominated by one class
- Logs show: "Model predicting only class X (overfitting)"

### Likely Causes
- Dataset: class imbalance, label noise, wrong class mapping, train/val leakage
- Dataloader: labels misaligned with images, one-hot not converted to indices
- Model/weights: head num_classes mismatch; raw `yolov5s-cls.pt` use without safe loading
- Loss: focal applied too early; label smoothing too high/low; `cls_task` too large
- Hyperparameters: LR too high; batch too large for BN; insufficient warmup
- Augmentations: too strong for medical images (keep minimal/off)

### Pre-flight (clear caches)
```powershell
$DATASET = "regurgitationV1"
$sets = @("train","valid","test")
foreach ($d in $sets) {
  $labels = Join-Path (Join-Path $DATASET $d) "labels"
  Get-ChildItem -Path $labels -Filter "*.cache*" -ErrorAction SilentlyContinue | Remove-Item -Force
}
```

### 1) Verify dataset
Run distribution, mapping and leakage checks:
```powershell
python verify_classification_dataset.py
python compare_classify_distribution.py
python compare_train_val_distribution.py
python verify_class_mapping.py
```
Expect balanced-enough classes, consistent mappings, no overlap train/val.

### 2) Dataloader sanity
Inspect a batch and label format at loss time (indices 0..C-1):
```powershell
python quick_validation_test.py
python check_image_label_pairs.py
```
Red flags: still one-hot at loss; batch label count ≠ image batch; labels out of range.

### 3) Tiny-subset overfit (should reach ~100%)
```powershell
python train_classification_task.py `
  --data regurgitationV1/data.yaml `
  --cfg yolov5c/models/yolov5sc.yaml `
  --epochs 20 `
  --batch-size 8 `
  --imgsz 416 `
  --name cls_overfit_8 `
  --cache `
  --nosave `
  --rect `
  --patience 0 `
  --hyp yolov5c/data/hyps/hyp.classV1.yaml `
  --subset 8
```
If it fails to overfit: fix head `num_classes`, loss wiring, labels. If it overfits: focus on data/HPs.

### 4) Loss configuration
- Start simple: CrossEntropy, label_smoothing 0.05–0.10, focal OFF initially
- Set `cls_task` = 0.1–0.2 to prevent domination early
- Confirm classification head `num_classes = 3`

### 5) Stable hyperparameters (medical)
```powershell
python train_classification_task.py `
  --data regurgitationV1/data.yaml `
  --cfg yolov5c/models/yolov5sc.yaml `
  --epochs 60 `
  --batch-size 32 `
  --imgsz 416 `
  --name cls_stable `
  --cache `
  --nosave `
  --patience 0 `
  --hyp yolov5c/data/hyps/hyp.classV3.yaml `
  --lr0 0.001
```
Tips: If unstable, use `--batch-size 16` and `--lr0 5e-4`. Keep aug minimal per medical best practices.

### 6) Safe use of yolov5s-cls.pt
- Only load compatible backbone weights (your training script supports safe loading when `--weights` contains `cls`)
- Use conservative LR and moderate batch:
```powershell
python train_classification_task.py `
  --data regurgitationV1/data.yaml `
  --weights yolov5s-cls.pt `
  --cfg yolov5c/models/yolov5sc.yaml `
  --epochs 300 `
  --batch-size 64 `
  --imgsz 416 `
  --name classifyloss_cls_safe `
  --cache `
  --nosave `
  --lr0 0.001 `
  --patience 0
```

### 7) Monitor per-epoch metrics
- Track accuracy/precision/recall/F1 and confusion matrix
- If collapse to one class: lower LR, increase label_smoothing to 0.1, reduce `cls_task` to 0.1–0.2, recheck class balance

### Quick fixes that often lift from ~40%
- Lower LR to 1e-3 (or 5e-4) and batch to 16–32
- Ensure labels are indices at loss time
- Start with `cls_task=0.1–0.2`, `fl_gamma=0.0` (add focal later if imbalance persists)
- Prefer scratch or safe-loaded weights over raw `yolov5s-cls.pt`

### Decision flow
1) Overfit tiny subset?
   - No → fix head/loss/labels
   - Yes →
2) Balanced data?
   - No → rebalance/fix leakage
   - Yes →
3) Tune LR/batch/warmup/smoothing/cls_task; keep aug minimal; monitor confusion matrix
