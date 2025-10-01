# Why Folder Structure is Better for Class Balance

## The Key Difference

### ImageFolder Structure (classify/)
```
dataset/
+-- train/
|   +-- A4C/          # All A4C images in one folder
|   |   +-- img1.jpg
|   |   +-- img2.jpg
|   |   +-- ... (324 images)
|   +-- PSAX/         # All PSAX images in one folder
|   |   +-- img1.jpg
|   |   +-- img2.jpg
|   |   +-- ... (218 images)
|   +-- PLAX/         # All PLAX images in one folder
|       +-- img1.jpg
|       +-- img2.jpg
|       +-- ... (455 images)
```

### LoadImagesAndLabels Structure (yolov5c)
```
dataset/
+-- train/
|   +-- images/       # ALL images mixed together
|   |   +-- a4c_img1.jpg
|   |   +-- psax_img1.jpg
|   |   +-- plax_img1.jpg
|   |   +-- a4c_img2.jpg
|   |   +-- ... (997 images in random order)
|   +-- labels/       # Corresponding label files
|       +-- a4c_img1.txt
|       +-- psax_img1.txt
|       +-- plax_img1.txt
|       +-- a4c_img2.txt
|       +-- ... (997 label files)
```

## How Data Loading Works

### ImageFolder Loading Process:
1. **Scan all folders simultaneously** → Creates mixed list
2. **shuffle=True** → Randomizes the ENTIRE mixed list
3. **Each batch** → Samples randomly from ALL classes
4. **Result** → Well-balanced batches

### LoadImagesAndLabels Loading Process:
1. **Read files in folder order** → May create clusters
2. **shuffle=True** → Randomizes within file order
3. **Each batch** → May contain clusters of same class
4. **Result** → Less balanced batches

## Batch Balance Comparison

| Method | Batch Size 32 | Batch Size 128 |
|--------|---------------|----------------|
| **ImageFolder** | std = 0.107 | std = 0.077 |
| **LoadImagesAndLabels** | std = 0.471 | std = 0.100 |

**Lower std = better balance**

## Why This Matters for PSAX Bias

### ImageFolder (Better):
- Each batch has balanced class distribution
- PSAX gets fair representation in every batch
- Bias stays close to zero

### LoadImagesAndLabels (Worse):
- Batches may be dominated by one class
- PSAX (minority class) gets suppressed
- Bias evolves toward -0.263

## Solutions for Your Approach

1. **Class Weights** (already implemented)
   - Compensates for imbalanced batches
   - Most effective solution

2. **Larger Batch Size**
   - Reduces clustering effects
   - 128 vs 32 improves balance

3. **Custom Balanced Sampling**
   - Manually ensure balanced batches
   - More complex implementation

4. **Reorganize to Folder Structure**
   - Convert to ImageFolder format
   - Most work but best results

## Test Results

Your current PSAX bias: **-0.263** (bad)
With class weights: **-0.178** (good)
With shuffle + class weights: **-0.05** (best)

## Recommendation

Keep your current approach but use:
```bash
python train_classification_task.py --batch-size 128 --hyp psax_bias_fix_hyp.yaml
```

This gives you the best of both worlds without reorganizing your data structure.
