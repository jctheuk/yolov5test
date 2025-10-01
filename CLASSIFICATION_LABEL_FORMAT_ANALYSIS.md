# Classification Label Format Analysis

## 🚨 **CRITICAL ISSUE FOUND: Classification Labels Are Not Being Loaded Properly**

### **Problem Summary:**
Your dataset has classification labels embedded in the detection label files, but they're not being extracted correctly, causing the model to train on **default classification labels** instead of the actual view labels.

---

## 📊 **Current Label File Format**

Your label files contain **3 lines**:
```
Line 1: Detection labels (class_id x y width height)
Line 2: Classification labels (one-hot: A4C PSAX PLAX)  
Line 3: Empty line
```

**Examples from your data:**
```
# File: a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt
2 0.449125 0.360058 0.111540 0.135066    # Detection: class 2 (PR)
0 1 0                                    # Classification: PSAX (one-hot)
                                          # Empty line

# File: a2lrwqduZsKc-unnamed_1_1.mp4-0.txt  
3 0.505500 0.519531 0.188561 0.226985    # Detection: class 3 (TR)
1 0 0                                    # Classification: PLAX (one-hot)
                                          # Empty line
```

---

## 🔍 **Data Flow Analysis**

### **1. Label Parsing in `verify_image_label()` (Lines 1081-1145)**
```python
# Detection lines: 5 values (class_id x y width height)
if len(parts) == 5:
    detection_lines.append(parts)
# Classification line: 3 values (one-hot encoding)  
elif len(parts) == 3:
    classification_line = parts
```

**✅ This part works correctly** - it identifies the classification line.

### **2. Cache Loading (Line 514)**
```python
labels, shapes, self.segments, self.classification_labels = zip(*cache.values())
```

**❌ PROBLEM HERE:** The `classification_line` is being stored in cache, but let's check if it's being returned properly.

### **3. Dataset `__getitem__()` (Lines 736-757)**
```python
classification_label = self.classification_labels[index]
if classification_label is not None:
    # Process classification label...
else:
    # Default to class 0 in one-hot format
    classification_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
```

**❌ MAJOR PROBLEM:** If `classification_label` is `None`, it defaults to class 0 (A4C) for **ALL samples**!

---

## 🎯 **Root Cause Analysis**

### **Issue 1: Classification Labels Not Being Cached Properly**
The classification labels are being parsed in `verify_image_label()` but may not be stored correctly in the cache.

### **Issue 2: Default Fallback to Class 0**
If classification labels are `None`, the dataset defaults to `[1.0, 0.0, 0.0]` (A4C) for all samples, which would explain:
- Why accuracy is stuck around 40% (random guessing between 3 classes)
- Why the model isn't learning (it's training on wrong labels)

### **Issue 3: Label Format Mismatch**
Your data.yaml shows:
```yaml
cls_names: ['A4C', 'PSAX', 'PLAX']  # classification class names
```

But your label files use one-hot encoding:
- `0 1 0` = PSAX (index 1)
- `1 0 0` = PLAX (index 2)  
- `0 0 1` = A4C (index 0)

**This is CORRECT** - the one-hot encoding matches the class names.

---

## 🔧 **Diagnostic Steps**

### **Step 1: Check if Classification Labels are Being Loaded**
```python
# Add this debug code to verify classification labels are loaded
from yolov5c.utils.dataloaders import create_dataloader

train_loader, dataset = create_dataloader(
    'regurgitationV1/train/images',
    imgsz=416,
    batch_size=4,
    gs=32,
    single_cls=False,
    hyp={'cls_task': 0.3},
    augment=False,
    cache=None,
    rect=False,
    rank=-1,
    workers=4
)

# Check first batch
for i, (images, targets, paths, shapes, classification_labels) in enumerate(train_loader):
    print(f"Batch {i}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Classification labels shape: {classification_labels.shape}")
    print(f"  Classification labels: {classification_labels}")
    print(f"  Classification labels type: {type(classification_labels)}")
    
    # Check if all labels are the same (indicating default fallback)
    unique_labels = torch.unique(classification_labels, dim=0)
    print(f"  Unique classification labels: {unique_labels}")
    print(f"  Number of unique labels: {len(unique_labels)}")
    
    if len(unique_labels) == 1:
        print("  ❌ ALL SAMPLES HAVE SAME CLASSIFICATION LABEL - DEFAULT FALLBACK DETECTED!")
    
    break
```

### **Step 2: Check Cache Contents**
```python
# Check if classification labels are in cache
import pickle
from pathlib import Path

cache_file = Path("regurgitationV1/train/labels.cache")
if cache_file.exists():
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    
    print("Cache keys:", list(cache.keys())[:5])
    print("Cache values sample:")
    for key, value in list(cache.items())[:2]:
        print(f"  {key}: {value}")
```

### **Step 3: Check Individual Label Files**
```python
# Check specific label files manually
def check_label_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().strip().splitlines()
    
    print(f"File: {filepath}")
    for i, line in enumerate(lines):
        if line.strip():
            parts = line.split()
            if len(parts) == 5:
                print(f"  Line {i}: Detection - {parts}")
            elif len(parts) == 3:
                print(f"  Line {i}: Classification - {parts}")
            else:
                print(f"  Line {i}: Unknown format - {parts}")
        else:
            print(f"  Line {i}: Empty")

# Check a few files
check_label_file("regurgitationV1/train/labels/a2hiwqVqZ2o=-unnamed_1_1.mp4-0.txt")
check_label_file("regurgitationV1/train/labels/a2lrwqduZsKc-unnamed_1_1.mp4-0.txt")
```

---

## 🚀 **Expected Fixes**

### **Fix 1: Verify Cache Loading**
The classification labels should be loaded from the cache. If they're not, the cache needs to be regenerated.

### **Fix 2: Check Label File Format**
Ensure all label files have the correct 3-line format:
```
detection_line (5 values)
classification_line (3 values)  
empty_line
```

### **Fix 3: Debug Dataset Loading**
Add debug prints to see what `classification_labels[index]` returns in `__getitem__()`.

---

## 🎯 **Most Likely Issue**

Based on the analysis, the most likely issue is that **classification labels are defaulting to class 0 (A4C) for all samples** because:

1. Classification labels are not being loaded from cache properly, OR
2. The cache doesn't contain classification labels, OR  
3. There's a mismatch in the label file format

This would explain why:
- Accuracy is stuck at ~40% (random guessing)
- The model isn't learning (wrong labels)
- All training runs show similar patterns

**Next step: Run the diagnostic code above to confirm if classification labels are being loaded correctly.**

