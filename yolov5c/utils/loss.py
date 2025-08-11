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
        
        # Classification loss for dual-task
        self.BCEcls_task = nn.CrossEntropyLoss(label_smoothing=h.get('label_smoothing', 0.1))
        self.cls_task_loss_weight = h.get('cls_task', 0.3)

    def __call__(self, p, targets, cls_targets=None):  # predictions, targets, classification_targets
        # Handle dual outputs: p can be either detection outputs only or (detection_outputs, classification_output)
        if isinstance(p, tuple) and len(p) == 2:
            detection_outputs, classification_output = p
        else:
            detection_outputs = p
            classification_output = None

        # Ensure detection_outputs is a list
        if not isinstance(detection_outputs, list):
            detection_outputs = [detection_outputs]

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lcls_task = torch.zeros(1, device=self.device)  # classification task loss
        
        tcls, tbox, indices, anchors = self.build_targets(detection_outputs, targets)  # targets

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
            # Ensure cls_targets are long tensors for CrossEntropyLoss
            if cls_targets.dtype != torch.long:
                cls_targets = cls_targets.long()
            
            # Handle one-hot encoded targets
            if cls_targets.dim() > 1 and cls_targets.shape[-1] > 1:
                cls_targets = cls_targets.argmax(dim=-1)
            
            # Ensure targets are within valid range
            if cls_targets.max() >= classification_output.shape[-1]:
                cls_targets = torch.clamp(cls_targets, 0, classification_output.shape[-1] - 1)
            
            lcls_task = self.BCEcls_task(classification_output, cls_targets) * self.cls_task_loss_weight

        # Total loss
        total_loss = (lbox + lobj + lcls + lcls_task) * bs
        
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
