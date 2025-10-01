# PSAX (Class 1) Bug - ROOT CAUSE IDENTIFIED

## 🚨 **THE SMOKING GUN**

### **Classification Head Bias Values:**

```
A4C (class 0):  -0.086670
PSAX (class 1): -0.262939  ← HIGHLY NEGATIVE!
PLAX (class 2): +0.402832  ← HIGHLY POSITIVE!
```

## **🎯 Why This Causes 9% PSAX Recall:**

In softmax classification, the bias affects the output logits:
```
logit = weight · features + bias
```

**With biases:**
- PSAX bias = -0.263 (strongly suppresses PSAX predictions)
- PLAX bias = +0.403 (strongly boosts PLAX predictions)
- A4C bias = -0.087 (neutral)

**Result:**
- PSAX is **systematically underpredicted** (negative bias suppresses it)
- PLAX is **systematically overpredicted** (positive bias boosts it)
- This explains the confusion matrix:
  - 57.6% of PSAX mispredicted as A4C
  - 33.3% of PSAX mispredicted as PLAX
  - Only 9.1% correctly predicted

## **🔍 Why Did This Bias Develop?**

### **The Bug:**

The bias should be **~0** at initialization (line 945 in models/common.py sets bias to 0).

But after 300 epochs of training:
- PSAX bias became -0.263 (moved DOWN)
- PLAX bias became +0.403 (moved UP)

**This means there's a bug in the training loop that causes:**
1. PSAX gradients to push bias DOWN (negative direction)
2. PLAX gradients to push bias UP (positive direction)
3. Unbalanced gradient updates across classes

### **Potential Causes:**

1. **Class weight imbalance** in loss function
   - PLAX (45.6% data) gets more gradient updates
   - PSAX (21.9% data) gets fewer gradient updates
   - But this alone shouldn't cause -0.263 vs +0.403 difference!

2. **Label smoothing bug** affecting class 1 differently
   - Check if label smoothing implementation treats middle class differently

3. **Loss computation bug** for class 1
   - Check if there's a systematic error in computing loss for class 1

4. **Gradient accumulation bug**
   - Check if gradients accumulate incorrectly for class 1

## **🔧 How to Verify:**

Train a fresh model and monitor bias values at each epoch:
- If PSAX bias immediately goes negative → initialization bug
- If PSAX bias gradually becomes negative → training loop bug
- If bias stays near zero but PSAX still underperforms → different issue

## **🚀 Immediate Fix to Test:**

Reset the classification head bias to zero and retrain:

```python
# In train_classification_task.py, after loading weights:
for name, module in model.named_modules():
    if 'linear' in name.lower() and isinstance(module, nn.Linear):
        if module.weight.shape[0] == 3:  # Classification head
            # Reset bias to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            print(f"Reset classification head bias to zero: {name}")
```

If this fixes the issue → confirms bias is the problem
If it doesn't fix → bias is a symptom, not the cause

## **🎯 The Real Question:**

**WHY did PSAX bias become so negative during training?**

This is the actual bug we need to find.

