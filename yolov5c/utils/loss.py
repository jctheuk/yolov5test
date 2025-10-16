# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from .anatomical_constraints import AnatomicalConstraints
from .mutual_constraints import MutuallyExclusiveConstraints


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
    def __init__(self, model, autobalance=False, class_weights=None):
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
        
        # Class weights for classification task (optional)
        self.class_weights = class_weights
        if self.class_weights is not None:
            if isinstance(self.class_weights, (list, tuple)):
                self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32, device=device)
            elif isinstance(self.class_weights, torch.Tensor):
                self.class_weights = self.class_weights.to(device)
            # print(f"[INFO] Using class weights for classification: {self.class_weights}")

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
        
        # Classification loss for dual-task - using standard CrossEntropy (NO FOCAL LOSS)
        self.softmax = nn.Softmax(dim=1)
        self.cls_task_loss_weight = h.get('cls_task', 0.3)  # Original weight
        self.temperature = h.get('temperature', 1.0)  # Temperature for softmax sharpness
        
        # Manual CrossEntropy implementation for PyTorch compatibility (like train_classification_task.py)
        self.classification_criterion = None  # Use manual implementation to avoid PyTorch version issues
        
        # Initialize anatomical constraints (DISABLED - fixed in dataset)
        self.use_constraints = h.get('use_anatomical_constraints', False)  # Changed default to False
        self.constraint_weight = h.get('constraint_weight', 0.1)
        if self.use_constraints:
            self.anatomical_constraints = AnatomicalConstraints(device=device)
            # print(f"[INFO] Anatomical constraints enabled with weight: {self.constraint_weight}")
        else:
            self.anatomical_constraints = None
            # print(f"[INFO] Anatomical constraints disabled")
        
        # Initialize mutually exclusive constraints
        self.use_mutual_constraints = h.get('use_mutual_constraints', False)
        self.mutual_constraint_weight = h.get('mutual_constraint_weight', 0.15)
        self.mutual_confidence_threshold = h.get('mutual_confidence_threshold', 0.25)
        
        if self.use_mutual_constraints:
            self.mutual_constraints = MutuallyExclusiveConstraints(device=device)
            print(f"[INFO] Mutually exclusive constraints enabled with weight: {self.mutual_constraint_weight}")
        else:
            self.mutual_constraints = None
        
        # Keep only essential debug info
        # print(f"[DEBUG] Classification loss weight: {self.cls_task_loss_weight}")
        # print(f"[DEBUG] Using manual CrossEntropy implementation for PyTorch compatibility")
    
    def manual_cross_entropy_loss(self, logits, targets):
        """
        Manual CrossEntropy loss implementation for PyTorch compatibility.
        Same as train_classification_task.py implementation.
        
        Args:
            logits: Raw classification logits [batch_size, num_classes]
            targets: Target class indices [batch_size]
        
        Returns:
            CrossEntropy loss value
        """
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=1)
        
        # Gather the log probabilities for the target classes
        batch_size = logits.shape[0]
        target_log_probs = log_probs[range(batch_size), targets]
        
        # Return negative log likelihood (CrossEntropy loss)
        return -target_log_probs.mean()

    def __call__(self, p, targets, cls_targets=None):  # predictions, targets, classification_targets
        # Handle dual outputs: p can be either detection outputs only or (detection_outputs, classification_output)
        if isinstance(p, tuple) and len(p) == 2:
            detection_outputs, classification_output = p
            
            # Check for NaN or inf values in classification output
            # if classification_output is not None:
            #     if torch.isnan(classification_output).any():
            #         print(f"[DEBUG] WARNING: NaN values found in classification output!")
            #     if torch.isinf(classification_output).any():
            #         print(f"[DEBUG] WARNING: Inf values found in classification output!")
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
        
        # Calculate detection losses
        for i, pi in enumerate(detection_outputs):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        # Calculate classification loss
        if classification_output is not None and cls_targets is not None:
            # Convert classification targets to class indices
            if cls_targets.dim() > 1 and cls_targets.shape[1] > 1:
                # One-hot encoded targets
                target_indices = torch.argmax(cls_targets, dim=1)
            else:
                # Already class indices
                target_indices = cls_targets.long()
            
            # Ensure targets are within valid range
            if target_indices.max() >= classification_output.shape[-1]:
                target_indices = torch.clamp(target_indices, 0, classification_output.shape[-1] - 1)
            
            # Calculate classification loss with manual implementation (like train_classification_task.py)
            try:
                # Apply class weights if provided (for handling class imbalance like PSAX)
                if self.class_weights is not None:
                    # Manual cross-entropy with class weights
                    log_probs = torch.nn.functional.log_softmax(classification_output, dim=1)
                    batch_size = classification_output.shape[0]
                    target_log_probs = log_probs[range(batch_size), target_indices]
                    
                    # Get weights for each target class
                    target_weights = self.class_weights[target_indices]
                    
                    # Weight the losses
                    weighted_losses = -target_log_probs * target_weights
                    lcls_task = weighted_losses.mean() * self.cls_task_loss_weight
                else:
                    # Manual CrossEntropy loss (no class weights) - same as train_classification_task.py
                    lcls_task = self.manual_cross_entropy_loss(classification_output, target_indices) * self.cls_task_loss_weight
                
                # Check for overfitting (model predicting only one class)
                # with torch.no_grad():
                #     pred_classes = torch.argmax(classification_output, dim=1)
                #     unique_preds = torch.unique(pred_classes)
                #     unique_targets = torch.unique(target_indices)
                    
                #     # Check if model is predicting all same class
                #     if len(unique_preds) == 1:
                #         print(f"[DEBUG] WARNING: Model is predicting only class {unique_preds[0]} (overfitting)")
                    
                #     # Check if targets are balanced
                #     if len(unique_targets) < 3:
                #         print(f"[DEBUG] WARNING: Only {len(unique_targets)} classes in targets")
                    
            except Exception as e:
                # print(f"[DEBUG] ERROR in classification loss calculation: {e}")
                lcls_task = torch.tensor(0.0, device=self.device)

        # Apply anatomical constraints if enabled (DISABLED - fixed in dataset)
        lconstraint = torch.zeros(1, device=self.device)  # constraint loss
        # Constraint calculation disabled - issues fixed in dataset
        # if self.use_constraints and self.anatomical_constraints is not None and classification_output is not None:
        #     try:
        #         # Convert classification output to one-hot for constraint calculation
        #         if classification_output.dim() > 1:
        #             # Already in logit format, convert to probabilities
        #             classification_probs = torch.softmax(classification_output, dim=1)
        #         else:
        #             # Convert to one-hot
        #             classification_probs = torch.zeros(classification_output.shape[0], 3, device=self.device)
        #             classification_probs.scatter_(1, classification_output.long().unsqueeze(1), 1.0)
        #         
        #         # Calculate constraint loss based on detection predictions
        #         # For now, we'll use a simplified constraint loss based on classification probabilities
        #         # This penalizes predictions that violate anatomical constraints
        #         constraint_penalty = 0.0
        #         for i in range(classification_probs.shape[0]):
        #             view_idx = torch.argmax(classification_probs[i]).item()
        #             
        #             # Get constraint weights for this view
        #             constraint_weights = torch.zeros(4, device=self.device)
        #             for det_class in range(4):
        #                 if det_class in self.anatomical_constraints.soft_weights[view_idx]:
        #                     constraint_weights[det_class] = self.anatomical_constraints.soft_weights[view_idx][det_class]
        #                 else:
        #                     constraint_weights[det_class] = 0.1  # Low weight for impossible detections
        #             
        #             # Apply constraint penalty based on detection confidence
        #             # This is a simplified version - in practice, you'd use actual detection predictions
        #             view_confidence = torch.max(classification_probs[i])
        #             constraint_penalty += (1.0 - view_confidence) * 0.1  # Small penalty for uncertain predictions
        #         
        #         lconstraint = torch.tensor(constraint_penalty, device=self.device) * self.constraint_weight
        #         
        #     except Exception as e:
        #         # print(f"[DEBUG] ERROR in constraint loss calculation: {e}")
        #         lconstraint = torch.tensor(0.0, device=self.device)
        
        # Apply mutually exclusive constraints if enabled
        lmutual = torch.zeros(1, device=self.device)  # mutual exclusion loss
        if self.use_mutual_constraints and self.mutual_constraints is not None and classification_targets is not None:
            try:
                # Use detection outputs for mutual exclusion constraint
                mutual_loss, violations, debug_info = self.mutual_constraints.compute_mutual_exclusion_loss(
                    detection_outputs,  # Raw detection outputs
                    classification_targets,  # View labels
                    confidence_threshold=self.mutual_confidence_threshold
                )
                lmutual = mutual_loss * self.mutual_constraint_weight
                
                # Log violations if they occur
                if violations > 0:
                    print(f"[CONSTRAINT] {violations} mutual exclusion violations detected (loss: {lmutual.item():.6f})")
                
            except Exception as e:
                print(f"[DEBUG] ERROR in mutual constraint calculation: {e}")
                lmutual = torch.tensor(0.0, device=self.device)
        
        # Total loss - IMPROVED: Scale detection and classification losses appropriately
        # Get batch size from first detection layer output
        bs = detection_outputs[0].shape[0]  # batch size
        # Scale detection losses by batch size (they are means and need to be in same scale)
        # Keep classification loss at its original scale (already appropriate magnitude)
        # Constraint loss is NOT scaled (already cumulative sum across batch)
        # This provides better balance between detection and classification tasks
        detection_loss = (lbox + lobj + lcls) * bs
        classification_loss = lcls_task
        constraint_loss = lconstraint  # NOT scaled (already sum, not mean)
        mutual_loss = lmutual  # NOT scaled (already normalized by batch size)
        total_loss = detection_loss + classification_loss + constraint_loss + mutual_loss
        
        # Check for NaN/Inf in total loss
        # if torch.isnan(total_loss) or torch.isinf(total_loss):
        #     print(f"[DEBUG] WARNING: NaN/Inf detected in total_loss!")
        #     print(f"[DEBUG]   lbox: {lbox.item():.6f}")
        #     print(f"[DEBUG]   lobj: {lobj.item():.6f}")
        #     print(f"[DEBUG]   lcls: {lcls.item():.6f}")
        #     print(f"[DEBUG]   lcls_task: {lcls_task.item():.6f}")
        #     print(f"[DEBUG]   lconstraint: {lconstraint.item():.6f}")
        #     print(f"[DEBUG]   batch_size: {bs}")
        #     print(f"[DEBUG]   detection_loss (scaled by bs): {detection_loss.item():.6f}")
        #     print(f"[DEBUG]   classification_loss (not scaled): {classification_loss.item():.6f}")
        #     print(f"[DEBUG]   constraint_loss (not scaled, sum): {constraint_loss.item():.6f}")
        #     print(f"[DEBUG]   detection/classification ratio: {detection_loss.item()/(classification_loss.item()+1e-6):.2f}")
        
        # Return total loss and individual losses (like original loss.py line 189)
        # Original format: return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
        # Our format: return total_loss, torch.cat((lbox, lobj, lcls, lcls_task, lconstraint, lmutual)).detach()
        # Ensure all loss components are scalars for concatenation
        lcls_task_scalar = lcls_task if lcls_task.dim() > 0 else lcls_task.unsqueeze(0)
        lconstraint_scalar = lconstraint if lconstraint.dim() > 0 else lconstraint.unsqueeze(0)
        lmutual_scalar = lmutual if lmutual.dim() > 0 else lmutual.unsqueeze(0)
        return total_loss, torch.cat((lbox, lobj, lcls, lcls_task_scalar, lconstraint_scalar, lmutual_scalar)).detach()

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
            raise ValueError(f"Unexpected targets shape: {targets.shape}")

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
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
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
