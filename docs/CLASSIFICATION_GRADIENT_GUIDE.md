## Classification-only Training with Proper Gradient Handling (Detection Loss Kept at Zero)

This guide explains how to mirror YOLOv5 `classify/` gradient behavior when doing joint models, while keeping detection losses present for logging but disabled (fixed zeros). The goal is: only classification loss contributes gradients; detection losses remain constant zeros and do not affect backpropagation.

### Core Principles

- Compute classification loss directly from `model(images)` outputs using standard PyTorch autograd.
- Do NOT call `.detach()` or `.item()` on the loss used for `backward()`.
- Use AMP `GradScaler` as in YOLOv5 classify to keep training numerically stable.
- Ensure labels are `LongTensor` and on the same device as logits.
- Keep detection losses as constant zeros for logging only; do not add them into `total_loss`.

### Minimal Pattern (Recommended)

```python
# Forward
det_out, cls_out = model(images)  # det_out kept for compatibility/logging; cls_out: [batch, num_classes]

# Labels
labels = labels.to(cls_out.device)
if labels.dtype != torch.long:
    labels = labels.long()

# Classification loss (keeps computation graph)
loss_cls = F.cross_entropy(cls_out, labels)

# Detection losses as constant zeros (for logs, not used for backward)
lbox = torch.zeros(1, device=cls_out.device)
lobj = torch.zeros(1, device=cls_out.device)
lcls = torch.zeros(1, device=cls_out.device)

# Total loss used for backward is classification loss only
total_loss = loss_cls

# AMP gradient steps
scaler.scale(total_loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()

# Logging (detach only when reporting)
loss_items = [lbox.detach().view(1), lobj.detach().view(1), lcls.detach().view(1), loss_cls.detach().view(1)]
```

### Common Pitfalls (Avoid)

- Returning or assigning a literal zero like `torch.tensor(0.0)` as a fallback loss. This breaks gradients.
- Doing `loss = loss.item()` or `loss = loss.detach()` before `backward()`.
- Mismatched devices or dtypes (e.g., CPU labels vs. CUDA logits, float labels for CE).
- Fabricating a “default” zero loss (even with `requires_grad=True`) instead of skipping classification loss when unavailable.

### One-hot Labels Handling

If labels arrive as one-hot vectors, convert to class indices first:

```python
if labels.dim() > 1 and labels.shape[-1] > 1:
    labels = labels.argmax(dim=-1)
labels = labels.long().to(cls_out.device)
```

### Why Keep Detection Losses as Zeros?

- Preserves interfaces and logging formats expected by downstream code (plots, trackers, hooks).
- Explicitly communicates that detection is disabled in this experiment.
- Avoids accidental contribution to gradients by not including them in `total_loss`.

### Sanity Checklist

- total_loss is exactly `loss_cls` (no detach, no item).
- `loss_items` for logs may be detached, but not `total_loss`.
- Labels: `LongTensor`, same device as `cls_out`.
- AMP flow: scale → backward → unscale_ → clip → step → update → zero_grad.

Following this pattern reproduces the clean, reliable gradient behavior of YOLOv5 `classify/` while keeping detection losses present (but inert) for reporting.

---

## Your Issue, How `classify/` Solves It, and The Fix

### What Problem You Had

- Backward error: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.
- Batch logs showed `Total batch loss: 0.0000`, meaning the loss used for backward had no gradients.
- Root causes observed in custom loss code:
  - Returned constant zeros like `torch.tensor(0.0)` as fallback losses (these do not carry a grad_fn).
  - Used `.item()` / `.detach()` on the loss before backward.
  - Occasional device/dtype mismatches (CPU labels vs CUDA logits, Float vs Long labels).

### How `classify/` Handles It Correctly

- Computes `loss = CrossEntropyLoss(model(images), labels)` directly, preserving the autograd graph.
- Never detaches or converts the training loss before `backward()`; `.item()` is used only for logging.
- Uses AMP flow strictly: `scale → backward → unscale_ → clip → step → update → zero_grad`.
- Ensures labels are `LongTensor` on the same device as logits.

### The Solution We Applied

- Classification loss now mirrors `classify/`:
  - `loss_cls = F.cross_entropy(cls_out, labels)` is used as `total_loss` without `.detach()`/`.item()`.
  - One-hot labels are converted via `argmax` to class indices (Long).
  - Labels are moved to the same device as `cls_out`.
- Detection losses are kept for logging compatibility but fixed to constant zeros and excluded from `total_loss`.
- Any fallback path that previously returned `torch.tensor(0.0)` now avoids breaking gradients; batches without classification output are skipped or produce a minimal grad-carrying zero only when necessary (not recommended—prefer skip).
- Debug/analysis utilities use `.detach().item()` only for printing, never for training loss.

### Minimal Before vs After

Before (problematic):
```python
# Bad: fallback/initialization or overwrite breaks gradients
lcls_task = torch.tensor(0.0, device=device)  # no grad_fn
total_loss = lcls_task.item()                 # detaches completely
```

After (fixed):
```python
# Good: pure CE for training loss, preserves gradients
labels = labels.to(cls_out.device).long()
total_loss = F.cross_entropy(cls_out, labels)

# Detection losses for logging only
lbox = torch.zeros(1, device=cls_out.device)
lobj = torch.zeros(1, device=cls_out.device)
lcls = torch.zeros(1, device=cls_out.device)
loss_items = [lbox.detach().view(1), lobj.detach().view(1), lcls.detach().view(1), total_loss.detach().view(1)]

# AMP flow
scaler.scale(total_loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

With these changes, gradients flow correctly from the classification head, detection remains inert (zeros), and training matches the stable behavior of `classify/`.


