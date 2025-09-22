# YOLOv5 🚀 Selective Classification Loss Implementation
# This script implements loss calculation only when passing through classification layer
# Based on history from 2025-09-19 and 2025-09-22 discussions

import torch
import torch.nn as nn
from .metrics import bbox_iou
from .torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Return positive, negative label smoothing BCE targets"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class SelectiveClassificationLoss:
    """
    Selective Classification Loss - Only computes classification loss when 
    passing through classification layer, preventing early interference with detection learning.
    """
    
    sort_obj_iou = False

    def __init__(self, model, autobalance=False, 
                 enable_classification=True,
                 classification_epoch_threshold=15,
                 classification_weight_ramp=10,
                 classification_final_weight=0.1):
        
        device = next(model.parameters()).device
        
        # Get hyperparameters from model or use defaults
        if hasattr(model, 'hyp'):
            h = model.hyp
        else:
            # Default hyperparameters
            h = {
                'box': 0.05,
                'cls': 0.5,
                'cls_pw': 1.0,
                'obj': 1.0,
                'obj_pw': 1.0,
                'iou_t': 0.20,
                'anchor_t': 4.0,
                'fl_gamma': 0.0,
                'cls_task': 0.3,
                'label_smoothing': 0.1
            }

        # Define criteria for detection
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Try to get the Detect layer from the model
        try:
            m = de_parallel(model).model[-1]  # Detect() module
            self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
            self.ssi = list(m.stride).index(16) if autobalance else 0
            self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
            self.na = m.na
            self.nc = m.nc
            self.nl = m.nl
            self.anchors = m.anchors
        except (AttributeError, IndexError):
            # Fallback for models without Detect layer
            self.balance = [4.0, 1.0, 0.25]
            self.ssi = 0
            self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
            self.na = getattr(model, 'na', 3)
            self.nc = getattr(model, 'nc', 4)
            self.nl = getattr(model, 'nl', 3)
            self.anchors = getattr(model, 'anchors', torch.tensor([[[10, 13], [16, 30], [33, 23]]]))

        self.device = device
        
        # Classification-specific settings
        self.enable_classification = enable_classification
        self.classification_epoch_threshold = classification_epoch_threshold
        self.classification_weight_ramp = classification_weight_ramp
        self.classification_final_weight = classification_final_weight
        
        # Classification loss function
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Current epoch tracking
        self.current_epoch = 0
        
        print(f"[DEBUG] SelectiveClassificationLoss initialized:")
        print(f"[DEBUG]   enable_classification: {enable_classification}")
        print(f"[DEBUG]   epoch_threshold: {classification_epoch_threshold}")
        print(f"[DEBUG]   weight_ramp: {classification_weight_ramp}")
        print(f"[DEBUG]   final_weight: {classification_final_weight}")
    
    def set_epoch(self, epoch):
        """Update current epoch for dynamic weight calculation"""
        self.current_epoch = epoch
    
    def get_classification_weight(self):
        """Calculate dynamic classification weight based on current epoch"""
        if not self.enable_classification:
            return 0.0
        
        if self.current_epoch < self.classification_epoch_threshold:
            # No classification loss in early epochs
            return 0.0
        elif self.current_epoch < (self.classification_epoch_threshold + self.classification_weight_ramp):
            # Gradual ramp-up
            ramp_progress = (self.current_epoch - self.classification_epoch_threshold) / self.classification_weight_ramp
            return self.classification_final_weight * ramp_progress
        else:
            # Full weight
            return self.classification_final_weight
    
    def __call__(self, p, targets, cls_targets=None):
        """
        Compute losses with selective classification loss calculation.
        
        Args:
            p: predictions (can be tuple of (detection_outputs, classification_output))
            targets: detection targets
            cls_targets: classification targets
        
        Returns:
            total_loss, [lbox, lobj, lcls, lcls_task]
        """
        
        # Handle dual outputs
        if isinstance(p, tuple) and len(p) == 2:
            detection_outputs, classification_output = p
        else:
            detection_outputs = p
            classification_output = None

        # Ensure detection_outputs is a list
        if not isinstance(detection_outputs, list):
            detection_outputs = [detection_outputs]

        # Initialize loss tensors
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lcls_task = torch.zeros(1, device=self.device)  # classification task loss
        
        # Build targets for detection
        tcls, tbox, indices, anchors = self.build_targets(detection_outputs, targets)
        
        # Calculate detection losses
        for i, pi in enumerate(detection_outputs):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=self.device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                # Objectness
                iou_value = iou.detach().clamp(0)
                if iou_value.dim() > 0:
                    iou_value = iou_value.squeeze(-1)
                tobj[b, a, gj, gi] = ((1.0 - self.gr) + self.gr * iou_value).type(tobj.dtype)

                # Classification
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        # Auto-balance adjustment
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        
        # Apply detection loss weights
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        # Calculate classification loss ONLY when conditions are met
        if classification_output is not None and cls_targets is not None:
            cls_weight = self.get_classification_weight()
            
            if cls_weight > 0:
                # Convert classification targets to class indices
                if cls_targets.dim() > 1 and cls_targets.shape[1] > 1:
                    target_indices = torch.argmax(cls_targets, dim=1)
                else:
                    target_indices = cls_targets.long()
                
                # Ensure targets are within valid range
                if target_indices.max() >= classification_output.shape[-1]:
                    target_indices = torch.clamp(target_indices, 0, classification_output.shape[-1] - 1)
                
                # Calculate classification loss
                try:
                    lcls_task = self.classification_loss(classification_output, target_indices)
                    lcls_task *= cls_weight
                    print(f"[DEBUG] Epoch {self.current_epoch}: Classification weight={cls_weight:.3f}, loss={lcls_task.item():.4f}")
                except Exception as e:
                    print(f"[DEBUG] ERROR in classification loss calculation: {e}")
                    lcls_task = torch.tensor(0.0, device=self.device)
            else:
                print(f"[DEBUG] Epoch {self.current_epoch}: Classification weight={cls_weight:.3f} (disabled)")

        # Total loss
        total_loss = (lbox + lobj + lcls + lcls_task) * len(targets)
        
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"[DEBUG] WARNING: NaN/Inf detected in total_loss!")
            print(f"[DEBUG]   lbox: {lbox.item():.6f}")
            print(f"[DEBUG]   lobj: {lobj.item():.6f}")
            print(f"[DEBUG]   lcls: {lcls.item():.6f}")
            print(f"[DEBUG]   lcls_task: {lcls_task.item():.6f}")
        
        # Ensure proper tensor shapes
        def ensure_tensor_shape(tensor):
            if tensor.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            elif tensor.dim() == 0:
                return tensor.unsqueeze(0)
            else:
                return tensor
        
        lbox_final = ensure_tensor_shape(lbox.detach()).view(1)
        lobj_final = ensure_tensor_shape(lobj.detach()).view(1)
        lcls_final = ensure_tensor_shape(lcls.detach()).view(1)
        lcls_task_final = ensure_tensor_shape(lcls_task.detach()).view(1)
        
        return total_loss, [lbox_final, lobj_final, lcls_final, lcls_task_final]

    def build_targets(self, p, targets):
        """Build targets for compute_loss(), input targets(image,class,x,y,w,h)"""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = targets.to(self.device)
        
        # Handle targets format
        if targets.shape[1] == 6:
            targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        elif targets.shape[1] == 7:
            targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        else:
            raise ValueError(f"Unexpected targets shape: {targets.shape}")

        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],
                            ], device=self.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn()"""
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
