# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        
        # Get hyperparameters from model or use defaults
        if hasattr(model, 'hyp'):
            h = model.hyp  # hyperparameters
        else:
            # Default hyperparameters if model.hyp doesn't exist
            h = {
                'box': 0.05,  # box loss gain
                'cls': 0.5,  # cls loss gain
                'cls_pw': 1.0,  # cls BCELoss positive_weight
                'obj': 1.0,  # obj loss gain (scale with pixels)
                'obj_pw': 1.0,  # obj BCELoss positive_weight
                'iou_t': 0.20,  # IoU training threshold
                'anchor_t': 4.0,  # anchor-multiple threshold
                'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
                'cls_task': 0.3,  # classification task loss weight
                'label_smoothing': 0.1
            }

        # Define criteria for detection
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Try to get the Detect layer from the model
        try:
            m = de_parallel(model).model[-1]  # Detect() module
            self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
            self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
            self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
            self.na = m.na  # number of anchors
            self.nc = m.nc  # number of classes
            self.nl = m.nl  # number of layers
            self.anchors = m.anchors
        except (AttributeError, IndexError):
            # Fallback for models without Detect layer
            self.balance = [4.0, 1.0, 0.25]
            self.ssi = 0
            self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
            self.na = getattr(model, 'na', 3)  # number of anchors
            self.nc = getattr(model, 'nc', 4)  # number of classes
            self.nl = getattr(model, 'nl', 3)  # number of layers
            self.anchors = getattr(model, 'anchors', torch.tensor([[[10, 13], [16, 30], [33, 23]]]))

        self.device = device
        
        # Classification loss for dual-task - using Softmax + NLLLoss instead of CrossEntropy
        self.softmax = nn.Softmax(dim=1)
        self.nll_loss = nn.NLLLoss()
        self.cls_task_loss_weight = h.get('cls_task', 0.3)  # Original weight
        self.temperature = h.get('temperature', 1.0)  # Temperature for softmax sharpness
        
        print(f"[DEBUG] Using Softmax + NLLLoss for classification")
        print(f"[DEBUG] Classification loss weight: {self.cls_task_loss_weight}")
        print(f"[DEBUG] Softmax temperature: {self.temperature}")
        
        print(f"[DEBUG] Using Softmax + NLLLoss for classification")
        print(f"[DEBUG] Classification loss weight: {self.cls_task_loss_weight}")

    def __call__(self, p, targets, cls_targets=None):  # predictions, targets, classification_targets
        # Debug: Print function inputs
        print(f"[DEBUG] ComputeLoss.__call__ inputs:")
        print(f"[DEBUG]   p type: {type(p)}")
        print(f"[DEBUG]   p is tuple: {isinstance(p, tuple)}")
        if isinstance(p, tuple):
            print(f"[DEBUG]   p length: {len(p)}")
            print(f"[DEBUG]   p[0] type: {type(p[0])}")
            print(f"[DEBUG]   p[1] type: {type(p[1])}")
        
        # Handle dual outputs: p can be either detection outputs only or (detection_outputs, classification_output)
        if isinstance(p, tuple) and len(p) == 2:
            detection_outputs, classification_output = p
            print(f"[DEBUG] Using dual outputs (detection + classification)")
            
            # Debug classification output details
            if classification_output is not None:
                print(f"[DEBUG] Classification output details:")
                print(f"[DEBUG]   Shape: {classification_output.shape}")
                print(f"[DEBUG]   Device: {classification_output.device}")
                print(f"[DEBUG]   Dtype: {classification_output.dtype}")
                print(f"[DEBUG]   Range: {classification_output.min():.4f} to {classification_output.max():.4f}")
                print(f"[DEBUG]   Mean: {classification_output.mean():.4f}")
                print(f"[DEBUG]   Std: {classification_output.std():.4f}")
                
                # Check for NaN or inf values
                if torch.isnan(classification_output).any():
                    print(f"[DEBUG] WARNING: NaN values found in classification output!")
                if torch.isinf(classification_output).any():
                    print(f"[DEBUG] WARNING: Inf values found in classification output!")
                
                # Check if output is all zeros or very small
                if classification_output.abs().max() < 1e-6:
                    print(f"[DEBUG] WARNING: Classification output is very small (max abs: {classification_output.abs().max():.2e})")
        else:
            detection_outputs = p
            classification_output = None
            print(f"[DEBUG] Using single output (detection only)")

        # Ensure detection_outputs is a list
        if not isinstance(detection_outputs, list):
            detection_outputs = [detection_outputs]

        # Debug: Print targets information
        print(f"[DEBUG] Targets shape: {targets.shape}")
        print(f"[DEBUG] cls_targets type: {type(cls_targets)}")
        if cls_targets is not None:
            print(f"[DEBUG] cls_targets shape: {cls_targets.shape}")
            print(f"[DEBUG] cls_targets dtype: {cls_targets.dtype}")
            print(f"[DEBUG] cls_targets device: {cls_targets.device}")
            print(f"[DEBUG] cls_targets sample values: {cls_targets[:5] if cls_targets.numel() > 0 else 'empty'}")

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lcls_task = torch.zeros(1, device=self.device)  # classification task loss
        
        tcls, tbox, indices, anchors = self.build_targets(detection_outputs, targets)  # targets
        
        # Debug targets
        print(f"[DEBUG] Built targets:")
        print(f"[DEBUG]   tcls: {len(tcls)} layers")
        print(f"[DEBUG]   tbox: {len(tbox)} layers")
        print(f"[DEBUG]   indices: {len(indices)} layers")
        print(f"[DEBUG]   anchors: {len(anchors)} layers")
        
        # Check if we have any targets
        total_targets = sum(len(idx[0]) for idx in indices)
        print(f"[DEBUG]   Total targets: {total_targets}")
        
        if total_targets == 0:
            print(f"[DEBUG] WARNING: No targets found! This will cause objectness loss to be 0.")

        # Detection losses
        for i, pi in enumerate(detection_outputs):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            
            # Debug objectness loss
            if i == 0:  # Only print for first layer to avoid spam
                print(f"[DEBUG] Layer {i} objectness loss:")
                print(f"[DEBUG]   pi[..., 4] range: {pi[..., 4].min():.4f} to {pi[..., 4].max():.4f}")
                print(f"[DEBUG]   tobj range: {tobj.min():.4f} to {tobj.max():.4f}")
                print(f"[DEBUG]   obji: {obji.item():.6f}")
                print(f"[DEBUG]   balance[i]: {self.balance[i]}")
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # Classification task loss
        if classification_output is not None and cls_targets is not None:
            # Debug: Print classification inputs
            print(f"[DEBUG] Classification output shape: {classification_output.shape}")
            print(f"[DEBUG] Classification targets shape: {cls_targets.shape}")
            print(f"[DEBUG] Classification targets dtype: {cls_targets.dtype}")
            print(f"[DEBUG] Classification targets range: {cls_targets.min()} to {cls_targets.max()}")
            
            # Handle one-hot encoded targets - convert to class indices
            if cls_targets.dim() > 1 and cls_targets.shape[-1] > 1:
                print(f"[DEBUG] Converting one-hot targets to indices")
                # Convert one-hot to class indices
                target_indices = cls_targets.argmax(dim=-1)
                print(f"[DEBUG] After argmax, targets shape: {target_indices.shape}")
                print(f"[DEBUG] After argmax, targets sample: {target_indices[:5]}")
            else:
                # Already class indices
                target_indices = cls_targets.long()
            
            # Ensure targets are within valid range
            if target_indices.max() >= classification_output.shape[-1]:
                print(f"[DEBUG] WARNING: Targets max ({target_indices.max()}) >= output classes ({classification_output.shape[-1]})")
                target_indices = torch.clamp(target_indices, 0, classification_output.shape[-1] - 1)
                print(f"[DEBUG] After clamp, targets range: {target_indices.min()} to {target_indices.max()}")
            
            # Calculate classification loss using Softmax + NLLLoss
            try:
                # Apply temperature-scaled softmax to get probabilities
                scaled_logits = classification_output / self.temperature
                probs = self.softmax(scaled_logits)
                print(f"[DEBUG] Softmax probabilities range: {probs.min():.4f} to {probs.max():.4f}")
                print(f"[DEBUG] Softmax probabilities sum per sample: {probs.sum(dim=1)[:5]}")
                
                # Take log of probabilities for NLLLoss
                log_probs = torch.log(probs + 1e-8)  # Add small epsilon to avoid log(0)
                
                # Calculate NLLLoss
                lcls_task = self.nll_loss(log_probs, target_indices) * self.cls_task_loss_weight
                print(f"[DEBUG] Classification loss calculated successfully: {lcls_task.item():.6f}")
                
                # Additional debug info
                with torch.no_grad():
                    pred_classes = torch.argmax(probs, dim=1)
                    correct = (pred_classes == target_indices).sum().item()
                    accuracy = correct / target_indices.shape[0]
                    print(f"[DEBUG] Classification accuracy: {accuracy:.4f} ({correct}/{target_indices.shape[0]})")
                    print(f"[DEBUG] Predicted classes sample: {pred_classes[:5]}")
                    print(f"[DEBUG] True classes sample: {target_indices[:5]}")
                    
                    # Check class distribution
                    unique_preds, pred_counts = torch.unique(pred_classes, return_counts=True)
                    unique_targets, target_counts = torch.unique(target_indices, return_counts=True)
                    print(f"[DEBUG] Predicted class distribution: {dict(zip(unique_preds.tolist(), pred_counts.tolist()))}")
                    print(f"[DEBUG] True class distribution: {dict(zip(unique_targets.tolist(), target_counts.tolist()))}")
                    
                    # Check if model is predicting all same class
                    if len(unique_preds) == 1:
                        print(f"[DEBUG] WARNING: Model is predicting only class {unique_preds[0]}")
                    
                    # Check if targets are balanced
                    if len(unique_targets) < 3:
                        print(f"[DEBUG] WARNING: Only {len(unique_targets)} classes in targets")
                    
            except Exception as e:
                print(f"[DEBUG] ERROR in classification loss calculation: {e}")
                import traceback
                traceback.print_exc()
                lcls_task = torch.tensor(0.0, device=self.device)
        else:
            print(f"[DEBUG] Classification loss not calculated:")
            print(f"[DEBUG]   classification_output is None: {classification_output is None}")
            print(f"[DEBUG]   cls_targets is None: {cls_targets is None}")
            if classification_output is not None:
                print(f"[DEBUG]   classification_output shape: {classification_output.shape}")
            if cls_targets is not None:
                print(f"[DEBUG]   cls_targets shape: {cls_targets.shape}")

        # Total loss
        total_loss = (lbox + lobj + lcls + lcls_task) * bs
        
        # Debug total loss components
        print(f"[DEBUG] Loss components:")
        print(f"[DEBUG]   lbox: {lbox.item():.6f}")
        print(f"[DEBUG]   lobj: {lobj.item():.6f}")
        print(f"[DEBUG]   lcls: {lcls.item():.6f}")
        print(f"[DEBUG]   lcls_task: {lcls_task.item():.6f}")
        print(f"[DEBUG]   total_loss: {total_loss.item():.6f}")
        print(f"[DEBUG]   batch_size: {bs}")
        
        # Ensure all loss components are properly shaped tensors (not empty) and have consistent shapes
        def ensure_tensor_shape(tensor):
            if tensor.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            elif tensor.dim() == 0:
                return tensor.unsqueeze(0)
            else:
                return tensor
        
        # Ensure each final component is a scalar tensor with shape [1]
        lbox_final = ensure_tensor_shape(lbox.detach()).view(1)
        lobj_final = ensure_tensor_shape(lobj.detach()).view(1)
        lcls_final = ensure_tensor_shape(lcls.detach()).view(1)
        lcls_task_final = ensure_tensor_shape(lcls_task.detach()).view(1)
        
        # Return total loss and individual losses as a list
        return total_loss, [lbox_final, lobj_final, lcls_final, lcls_task_final]
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # Ensure targets is on the same device as other tensors
        targets = targets.to(self.device)
        
        # Handle targets format - should be (image_id, class, x, y, w, h) = 6 columns
        if targets.shape[1] == 6:
            # Add anchor index column to make it 7 columns
            targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        elif targets.shape[1] == 7:
            # Already has anchor indices
            targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
        else:
            raise ValueError(f"Targets should have 6 or 7 columns, got {targets.shape[1]}")

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            # Handle anchors structure
            if isinstance(self.anchors, torch.Tensor):
                if self.anchors.dim() == 3:
                    anchors = self.anchors[i] if i < self.anchors.shape[0] else self.anchors[0]
                else:
                    anchors = self.anchors
            else:
                anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], device=self.device)
            
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
