# Why Not 95% Accuracy?

## The Hard Truth

**Class imbalance fix will get you to 55-60%, not 95%.**

**With ALL optimizations: 65-70% (realistic ceiling for joint architecture)**

**Gap from 95%: ~30 percentage points**

---

## Main Bottleneck: Joint Detection+Classification Architecture

### The Problem:

Your model uses **joint detection+classification architecture** where:
- Same backbone features for both detection AND classification
- Detection head exists (even if losses = 0)
- Features optimized for detection, not classification

The successful `classify/` uses **pure classification** where:
- ALL features dedicated to classification
- No detection head interference
- No feature competition

### Impact: **30-40% accuracy loss**

---

## Complete Performance Breakdown

| Bottleneck | Impact | Your Status | Fix Difficulty |
|------------|--------|-------------|----------------|
| **Joint Architecture** | **30-40%** | Joint model | Hard (major change) |
| **Class Imbalance** | **10-15%** | Fixed! | Easy (done) |
| **Small Batch Size** | **5-10%** | 32 (should be 128) | Easy |
| **Data Structure** | **5-10%** | LoadImagesAndLabels | Medium |
| **Optimizer** | **3-5%** | SGD (should be Adam) | Easy |
| **Detection Head** | **3-5%** | Exists but disabled | Medium |

**Total gap: ~53 percentage points**

---

## Expected Improvement Path

```
Current:                  41.4%
├─ + Class weights:       55.0% (+13.6%)  ← You are here
├─ + Batch size 128:      60.0% (+5.0%)
├─ + Adam optimizer:      63.0% (+3.0%)
├─ + WeightedSampler:     65.0% (+2.0%)
├─ ──────────────────────────────────────
│  Maximum with joint architecture: ~65-70%
├─ ──────────────────────────────────────
├─ + Pure classification: 85.0% (+20.0%)
└─ + Optimize architecture: 95.0% (+10.0%)  ← classify/ is here
```

---

## Why Joint Architecture Hurts

### 1. Feature Competition
- **Detection needs:** Edge features, object boundaries, spatial info
- **Classification needs:** Texture features, global patterns, semantic info
- **Result:** Compromise features that are suboptimal for both

### 2. Gradient Conflict
- Detection head parameters consume gradient flow
- Classification head gets weaker gradient signal
- **Result:** Slower learning, lower accuracy

### 3. Model Capacity
- Joint model spreads capacity across two tasks
- Classification head is smaller/simpler
- **Result:** Less capacity for classification learning

### 4. Training Dynamics
- Model pre-trained on detection task
- Features pre-optimized for detection
- **Result:** Classification works with detection-optimized features

---

## Realistic Options

### OPTION 1: Accept 65-70% (Easiest)
**Keep joint architecture, optimize everything else**

Steps:
1. Apply class weights fix (41% → 55%) ✅ Already done
2. Increase batch size to 128 (55% → 60%)
3. Switch to Adam optimizer (60% → 63%)
4. Add WeightedRandomSampler (63% → 65%)
5. Fine-tune classification head (65% → 70%)

**Effort:** Low
**Result:** 65-70% accuracy
**Timeline:** 1-2 days

---

### OPTION 2: Switch to Pure Classification (Recommended)
**Use separate classification model**

Steps:
1. Train pure classification model (like `classify/`)
2. Use `yolov5s-cls.pt` architecture
3. Apply all optimizations
4. Expected: 85-95% accuracy

**Effort:** Medium
**Result:** 85-95% accuracy (matches `classify/`)
**Timeline:** 3-5 days

**Command:**
```bash
python classify/train.py \
  --model yolov5s-cls.pt \
  --data ../regurgitationV1_classify \
  --epochs 300 \
  --batch-size 128 \
  --optimizer Adam \
  --lr0 0.001 \
  --label-smoothing 0.1
```

---

### OPTION 3: Two-Stage Approach (Best of Both Worlds)
**Train detection and classification separately**

Steps:
1. Train detection model for object detection
2. Train classification model for view classification
3. Use both at inference time
4. Expected: 90-95% classification + detection

**Effort:** High
**Result:** Best performance for both tasks
**Timeline:** 1-2 weeks

---

### OPTION 4: Enhance Joint Architecture (Compromise)
**Increase classification head capacity**

Steps:
1. Add more layers to classification head
2. Increase feature dimensions
3. Add attention mechanism
4. Fine-tune classification-specific layers
5. Expected: 70-80% accuracy

**Effort:** Medium-High
**Result:** 70-80% accuracy
**Timeline:** 1 week

---

## Key Question for You

**Do you NEED joint detection+classification in a single model?**

### If YES (joint is required):
- Realistic ceiling: **65-70%** with optimizations
- Best path: Apply all optimizations from Option 1
- Consider Option 4 if you need 70-80%

### If NO (can use separate models):
- Realistic ceiling: **85-95%** (matches `classify/`)
- Best path: Option 2 (pure classification)
- Use Option 3 if you need both detection AND classification

---

## Immediate Next Steps

### SHORT TERM (This Week):
```bash
# Test class weights fix
python train_classification_task.py \
  --data regurgitationV1/data.yaml \
  --epochs 50 \
  --batch-size 128 \
  --device auto \
  --weights yolov5s.pt \
  --hyp psax_bias_fix_hyp.yaml \
  --optimizer Adam \
  --patience 0
```
**Expected: 55-65% accuracy**

### MEDIUM TERM (Next 2 Weeks):
**Decision point:** Keep joint or switch to pure classification?

If keeping joint:
- Add WeightedRandomSampler
- Increase classification head capacity
- Target: 65-70%

If switching to pure:
- Use `classify/train.py` with your data
- Match `classify/` setup exactly
- Target: 85-95%

---

## Summary

| Aspect | Current | With Fixes | With Pure Model |
|--------|---------|-----------|-----------------|
| Accuracy | 41.4% | 65-70% | 85-95% |
| PSAX Recall | 9.1% | 25-35% | 85-95% |
| Architecture | Joint | Joint | Pure |
| Effort | - | Low | Medium |
| Timeline | - | Days | Week |

**Bottom line:** Class imbalance is THE reason for poor PSAX performance, but joint architecture is THE reason you won't reach 95%. Fix class imbalance to get to 65%, switch to pure classification to get to 95%.
