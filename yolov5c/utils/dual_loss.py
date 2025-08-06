import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import xywh2xyxy
from utils.metrics import bbox_iou, box_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
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


class ComputeDualLoss:
    """
    Compute losses for dual-task YOLOv5 model (detection + classification)
    """
    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp  # hyperparameters

        # Define criteria for detection
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Define criteria for classification
        self.BCEcls_task = nn.CrossEntropyLoss()  # For classification task

        # Class label smoothing
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))

        # Focal loss
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Get the Detect layer
        detect_layer = None
        for layer in de_parallel(model).model:
            if hasattr(layer, 'nl'):  # Detect layer has nl attribute
                detect_layer = layer
                break

        if detect_layer:
            m = detect_layer
            self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
            self.ssi = list(m.stride).index(16) if autobalance else 0
            self.na = m.na
            self.nc = m.nc
            self.nl = m.nl
            self.anchors = m.anchors
        else:
            # Fallback values if no Detect layer found
            self.balance = [4.0, 1.0, 0.25]
            self.ssi = 0
            self.na = 3
            self.nc = 4  # Default detection classes
            self.nl = 3
            self.anchors = torch.tensor([[10, 13], [16, 30], [33, 23]])

        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.device = device

    def __call__(self, model_output, targets, cls_targets=None):
        """
        Compute both detection and classification losses
        
        Args:
            model_output: Tuple of (detection_outputs, classification_output)
            targets: Detection targets
            cls_targets: Classification targets (optional)
        """
        # Parse model output
        if isinstance(model_output, tuple) and len(model_output) == 2:
            detection_outputs, classification_output = model_output
        else:
            detection_outputs = model_output
            classification_output = None

        # Ensure detection_outputs is a list
        if not isinstance(detection_outputs, list):
            detection_outputs = [detection_outputs]

        # Detection loss
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        tcls, tbox, indices, anchors = self.build_targets(detection_outputs, targets)

        # Detection losses
        for i, pi in enumerate(detection_outputs):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)

            n = b.shape[0]
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                # Classification
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # Classification loss
        lcls_task = torch.zeros(1, device=self.device)
        if classification_output is not None and cls_targets is not None:
            lcls_task = self.BCEcls_task(classification_output, cls_targets)

        # Apply hyperparameters
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lcls_task *= self.hyp.get('cls_task', 1.0)  # Classification task weight

        bs = tobj.shape[0] if len(detection_outputs) > 0 else 1

        # Total loss
        total_loss = (lbox + lobj + lcls + lcls_task) * bs
        
        # Return total loss and individual losses
        # Ensure all losses have the same shape for concatenation
        lbox = lbox.view(1)
        lobj = lobj.view(1) 
        lcls = lcls.view(1)
        lcls_task = lcls_task.view(1)
        
        return total_loss, torch.cat((lbox, lobj, lcls, lcls_task)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = targets.to(self.device)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0],
                           [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                           # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                           ], device=self.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < 4.0  # limit
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
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - 0.5).long()
            gi, gj = gij.T  # grid indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1).long(), gi.clamp_(0, int(gain[2]) - 1).long()))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch 