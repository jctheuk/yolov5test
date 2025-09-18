# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou, compute_ap
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed to this ' \
                              f'script i.e. {data} ({nc} classes). Pass models trained on --data {data}'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    # Use three profiling timers for preprocessing, inference, and NMS
    dt = (Profile(), Profile(), Profile())  # profiles
    loss = torch.zeros(4, device=device)
    jdict, stats = [], []
    
    # Classification metrics tracking
    all_cls_outputs = []
    all_cls_targets = []
    cls_correct = 0
    cls_total = 0

    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    # Match dataloader return order: (images, targets, paths, shapes, classification_labels)
    for batch_i, (im, labels, paths, shapes, classification_labels) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                labels = labels.to(device)
                if classification_labels is not None:
                    classification_labels = classification_labels.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            # Get model outputs
            model_output = model(im)
            
            # Parse model output consistently
            from utils.general import parse_model_output, validate_detection_outputs
            preds, classification_output = parse_model_output(model_output)
            
            # Validate detection outputs - preds should be a list of tensors
            if isinstance(preds, list):
                validate_detection_outputs(preds)
            else:
                # If preds is not a list, convert it to a list
                preds = [preds] if isinstance(preds, torch.Tensor) else list(preds)
                validate_detection_outputs(preds)
            
            # For loss computation, we need the same outputs
            if compute_loss:
                train_output = model_output
            else:
                train_output = None
            
            # Handle different output formats for predictions
            if isinstance(preds, tuple) and len(preds) == 2:
                preds, _ = preds  # Ignore classification for NMS
            else:
                preds = preds

        # Loss
        if compute_loss:
            # For loss computation, we need training-mode outputs (include both detection and classification)
            model.train()  # Switch to training mode temporarily
            with torch.no_grad():
                train_output = model(im)  # Get training-mode outputs
            model.eval()  # Switch back to evaluation mode

            # Compute loss using full output tuple and both label types
            loss_items = compute_loss(train_output, labels, classification_labels)[1]
            # Ensure tensor and consistent shape before accumulating
            if isinstance(loss_items, list):
                loss_items_tensor = torch.stack([
                    (x if isinstance(x, torch.Tensor) else torch.tensor(float(x), device=device)).view(-1)[0]
                    for x in loss_items
                ]).to(device)
            else:
                loss_items_tensor = loss_items.to(device).view(-1)
            loss += loss_items_tensor  # box, obj, cls, cls_task
        labels[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [labels[labels[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        # NMS
        with dt[2]:
            # Process detection predictions - preds is already a list of tensors
            processed_det_preds = []
            for det_pred in preds:
                processed_det = non_max_suppression(det_pred,
                                              conf_thres,
                                              iou_thres,
                                              labels=lb,
                                              multi_label=True,
                                              agnostic=single_cls,
                                              max_det=max_det)
                processed_det_preds.extend(processed_det)
            preds = processed_det_preds

        # Metrics
        for si, pred in enumerate(preds):
            target_labels = labels[labels[:, 0] == si, 1:]
            nl, npr = target_labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), target_labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=target_labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(target_labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((target_labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], target_labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Classification validation (if classification output exists)
        if classification_output is not None and classification_labels is not None:
            # Process classification labels
            if classification_labels.dim() > 1 and classification_labels.shape[-1] > 1:
                # Convert one-hot encoded labels to class indices
                cls_targets = classification_labels.argmax(dim=1)
            else:
                cls_targets = classification_labels.long()
            
            # Ensure batch size matches
            if cls_targets.shape[0] != im.shape[0]:
                if cls_targets.shape[0] < im.shape[0]:
                    # Pad with zeros
                    pad_size = im.shape[0] - cls_targets.shape[0]
                    cls_targets = torch.cat([cls_targets, torch.zeros(pad_size, dtype=torch.long, device=device)])
                else:
                    # Truncate
                    cls_targets = cls_targets[:im.shape[0]]
            
            # Calculate classification accuracy
            if classification_output.dim() > 1:
                pred_classes = torch.argmax(classification_output, dim=1)
            else:
                pred_classes = classification_output.long()
            
            # Ensure predictions and targets have the same shape
            if pred_classes.shape[0] != cls_targets.shape[0]:
                if pred_classes.shape[0] < cls_targets.shape[0]:
                    pred_classes = pred_classes[:cls_targets.shape[0]]
                else:
                    cls_targets = cls_targets[:pred_classes.shape[0]]
            
                        # Calculate correct predictions
            correct_cls = (pred_classes == cls_targets).sum().item()
            cls_correct += correct_cls
            cls_total += cls_targets.shape[0]
            
            # Collect for final metrics
            all_cls_outputs.append(classification_output.cpu())
            all_cls_targets.append(cls_targets.cpu())
            
            # Store classification data for confusion matrix (optional)
            if plots and hasattr(confusion_matrix, 'process_classification_batch'):
                if cls_targets.numel() > 0 and pred_classes.numel() > 0:
                    confusion_matrix.process_classification_batch(cls_targets.cpu(), pred_classes.cpu())
                    # Debug: Log that we're collecting data
                    if batch_i == 0:  # Only log once per epoch
                        LOGGER.info(f"Collecting classification data: batch {batch_i}, targets shape {cls_targets.shape}, preds shape {pred_classes.shape}")

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, labels, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end', batch_i, im, labels, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        tp, fp, p, r, f1, ap, ap_class = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
        ap50, ap = 0.0, 0.0
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(s)  # Print header
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    elif nc > 1 and len(stats) and len(ap_class) > 0:  # Ensure per-class results are always printed when available
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)
    
    # Print classification confusion matrix only (not detection)
    if hasattr(confusion_matrix, 'classification_true_labels') and hasattr(confusion_matrix, 'classification_pred_labels'):
        true_labels = confusion_matrix.classification_true_labels
        pred_labels = confusion_matrix.classification_pred_labels
        
        if len(true_labels) > 0 and len(pred_labels) > 0:
            print('\nClassification Confusion Matrix:')
            confusion_matrix.print_classification_confusion_matrix(true_labels, pred_labels)

    # Compute classification metrics
    cls_results = None
    if cls_total > 0:
        cls_accuracy = cls_correct / cls_total
        
        # Compute additional classification metrics if we have outputs
        if all_cls_outputs and all_cls_targets:
            try:
                from sklearn.metrics import precision_recall_fscore_support, accuracy_score
                
                # Concatenate all outputs and targets
                all_cls_outputs = torch.cat(all_cls_outputs, dim=0)
                all_cls_targets = torch.cat(all_cls_targets, dim=0)
                
                # Convert to numpy for sklearn
                if all_cls_outputs.dim() > 1:
                    # Convert to float32 to avoid Half precision issues
                    all_cls_outputs_f32 = all_cls_outputs.float()
                    pred_classes = torch.argmax(all_cls_outputs_f32, dim=1).cpu().numpy()
                    pred_probs = torch.softmax(all_cls_outputs_f32, dim=1).cpu().numpy()
                else:
                    # Convert to float32 to avoid Half precision issues
                    all_cls_outputs_f32 = all_cls_outputs.float()
                    pred_classes = all_cls_outputs_f32.cpu().numpy()
                    pred_probs = torch.sigmoid(all_cls_outputs_f32).cpu().numpy()
                
                true_classes = all_cls_targets.cpu().numpy()
                
                # Calculate basic metrics
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                    true_classes, pred_classes, average='weighted', zero_division=0
                )
                
                # Classification results without mAP (mAP is not appropriate for classification)
                cls_results = {
                    'accuracy': cls_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'total_samples': cls_total
                }
                
                # Print detailed classification results table
                LOGGER.info('\nClassification Results:')
                LOGGER.info(f"{'Class':>22}{'Images':>11}{'Instances':>11}{'P':>11}{'R':>11}{'F1':>11}{'Acc':>11}")
                LOGGER.info(f"{'all':>22}{cls_total:>11}{cls_total:>11}{precision:>11.3g}{recall:>11.3g}{f1_score:>11.3g}{cls_accuracy:>11.3g}")
                
                # Print per-class results
                num_classes = pred_probs.shape[1] if pred_probs is not None else 0
                if num_classes > 1:
                    from sklearn.metrics import precision_recall_fscore_support
                    # Calculate per-class metrics
                    precision_per_class, recall_per_class, _, _ = precision_recall_fscore_support(
                        true_classes, pred_classes, average=None, zero_division=0
                    )
                    
                    # Per-class metrics are already calculated above
                    
                    # Count instances per class
                    class_counts = np.bincount(true_classes, minlength=num_classes)
                    
                    # Get classification class names from data config
                    cls_names = data.get('cls_names', [f'class_{i}' for i in range(num_classes)])
                    
                    # Print per-class results
                    for i in range(num_classes):
                        class_name = cls_names[i] if i < len(cls_names) else f'class_{i}'
                        # Ensure we don't access out of bounds
                        class_count = class_counts[i] if i < len(class_counts) else 0
                        precision_val = precision_per_class[i] if i < len(precision_per_class) else 0
                        recall_val = recall_per_class[i] if i < len(recall_per_class) else 0
                        # Calculate F1 score for this class
                        class_f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-8)
                        # Calculate accuracy for this class (correct predictions / total predictions for this class)
                        class_accuracy = np.sum((true_classes == i) & (pred_classes == i)) / (class_count + 1e-8)
                        LOGGER.info(f"{class_name:>22}{cls_total:>11}{class_count:>11}{precision_val:>11.3g}{recall_val:>11.3g}{class_f1:>11.3g}{class_accuracy:>11.3g}")
                
            except ImportError:
                LOGGER.warning("sklearn not available, only accuracy will be computed")
                cls_results = {
                    'accuracy': cls_accuracy,
                    'total_samples': cls_total
                }
        else:
            cls_results = {
                'accuracy': cls_accuracy,
                'total_samples': cls_total
            }

    # Save JSON
    if save_json and (jdict or save_dir):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path('../datasets/coco/annotations/instances_val2017.json') if is_coco else data.get('path', '') / 'annotations.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float() if half else None  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    if len(ap_class) > 0:
        for i, c in enumerate(ap_class):
            if i < len(ap) and c < len(maps):
                maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, cls_results


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.info('WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
