# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Plotting utils
"""

import contextlib
import math
import os
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter1d
from ultralytics.utils.plotting import Annotator

from utils import TryExcept, threaded
from utils.general import LOGGER, clip_boxes, increment_path, xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
RANK = int(os.getenv('RANK', -1))
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            LOGGER.info(f'Saving {f}... ({n}/{channels})')
            plt.savefig(f, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    from scipy.signal import butter, filtfilt

    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def output_to_target(output, max_det=300):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    return torch.cat(targets, 0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname='images.jpg', names=None):
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets
            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype('int')
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(box, label, color=color)
    annotator.im.save(fname)  # save


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_val_txt():  # from utils.plots import *; plot_val()
    # Plot val.txt histograms
    x = np.loadtxt('val.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_val_study(file='', dir='', x=None):  # from utils.plots import *; plot_val_study()
    # Plot file=study.txt generated by val.py (or plot all study*.txt in dir)
    save_dir = Path(file).parent if file else Path(dir)
    plot2 = False  # plot additional results
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [save_dir / f'study_coco_{x}.txt' for x in ['yolov5n6', 'yolov5s6', 'yolov5m6', 'yolov5l6', 'yolov5x6']]:
    for f in sorted(save_dir.glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
                ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[5, 1:j],
                 y[3, 1:j] * 1E2,
                 '.-',
                 linewidth=2,
                 markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-',
             linewidth=2,
             markersize=8,
             alpha=.25,
             label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(25, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    f = save_dir / 'study.png'
    print(f'Saving {f}...')
    plt.savefig(f, dpi=300)


@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
def plot_labels(labels, names=(), save_dir=Path('')):
    # plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):  # color histogram bars by class
        [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # known issue #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()


def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path('images.jpg')):
    # Show classification image grid with labels (optional) and predictions (optional)
    from utils.augmentations import denormalize

    names = names or [f'class{i}' for i in range(1000)]
    blocks = torch.chunk(denormalize(im.clone()).cpu().float(), len(im),
                         dim=0)  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n ** 0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis('off')
        if labels is not None:
            s = names[labels[i]] + (f'—{names[pred[i]]}' if pred is not None else '')
            ax[i].set_title(s, fontsize=8, verticalalignment='top')
    plt.savefig(f, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        LOGGER.info(f'Saving {f}')
        if labels is not None:
            LOGGER.info('True:     ' + ' '.join(f'{names[i]:3s}' for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info('Predicted:' + ' '.join(f'{names[i]:3s}' for i in pred[:nmax]))
    return f


def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    print(f'Best results from row {j} of {evolve_csv}:')
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    f = evolve_csv.with_suffix('.png')  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')


def plot_results(file='path/to/results.csv', dir=''):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ':', label='smooth', linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig(save_dir / 'results.png', dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print(f'Warning: Plotting error for {f}; {e}')
    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def plot_confusion_matrix(cm, names, normalize=True, save_dir='', prefix=''):
    """
    Plot confusion matrix for detection results
    Args:
        cm (array): Confusion matrix array
        names (list): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        save_dir (str): Directory to save the confusion matrix plot
        prefix (str): Prefix for the saved file name
    """
    try:
        import seaborn as sn

        array = cm
        if normalize:
            array = array / (array.sum(0).reshape(1, -1) + 1E-6)  # normalize columns
            array[array < 0.005] = 0  # don't annotate small values

        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if len(names) < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == cm.shape[0]  # apply names to ticklabels
        
        sn.heatmap(array, 
                   annot=not (len(names) > 30), 
                   annot_kws={"size": 8}, 
                   cmap='Blues', 
                   fmt='.2f' if normalize else '.0f',
                   square=True, 
                   xticklabels=names if labels else "auto",
                   yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
        
        title = f'{prefix} Confusion Matrix'
        if normalize:
            title += ' (Normalized)'
        plt.title(title, fontsize=14)
        plt.ylabel('True', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        fig.savefig(Path(save_dir) / f'{prefix}_confusion_matrix.png', dpi=250)
        plt.close(fig)
        
        LOGGER.info(f"Confusion matrix saved to {save_dir}/{prefix}_confusion_matrix.png")
        
    except Exception as e:
        LOGGER.warning(f'Warning: Confusion matrix plot error: {e}')


def plot_classification_confusion_matrix(true_labels, pred_labels, names, save_dir='', prefix='classification'):
    """
    Generate and plot a confusion matrix for classification results
    Args:
        true_labels (array): True class labels
        pred_labels (array): Predicted class labels
        names (list): List of class names
        save_dir (str): Directory to save the confusion matrix plot
        prefix (str): Prefix for the saved file name
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sn
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized[np.isnan(cm_normalized)] = 0
        
        # Plot the confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), tight_layout=True)
        
        # Raw counts
        sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                  xticklabels=names, yticklabels=names, ax=ax1)
        ax1.set_title(f'{prefix} Confusion Matrix (Counts)', fontsize=14)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sn.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', square=True, 
                   xticklabels=names, yticklabels=names, ax=ax2)
        ax2.set_title(f'{prefix} Confusion Matrix (Normalized)', fontsize=14)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        fig.savefig(Path(save_dir) / f'{prefix}_confusion_matrix.png', dpi=250)
        plt.close(fig)
        
        LOGGER.info(f"Classification confusion matrix saved to {save_dir}/{prefix}_confusion_matrix.png")
        
    except Exception as e:
        LOGGER.warning(f'Warning: Classification confusion matrix plot error: {e}')


def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop


def plot_detection_metrics(detection_results, class_names, save_dir, prefix=''):
    """
    Plot detection metrics in a bar chart format similar to the log table
    
    Args:
        detection_results: Dictionary with detection metrics
        class_names: List of class names
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    precision = detection_results.get('precision_per_class', [])
    recall = detection_results.get('recall_per_class', [])
    map50 = detection_results.get('map50_per_class', [])
    map50_95 = detection_results.get('map50_95_per_class', [])
    
    # Ensure all arrays have the same length as class_names
    num_classes = len(class_names)
    precision = list(precision) + [0.0] * (num_classes - len(precision))
    recall = list(recall) + [0.0] * (num_classes - len(recall))
    map50 = list(map50) + [0.0] * (num_classes - len(map50))
    map50_95 = list(map50_95) + [0.0] * (num_classes - len(map50_95))
    
    # Truncate to match class_names length
    precision = precision[:num_classes]
    recall = recall[:num_classes]
    map50 = map50[:num_classes]
    map50_95 = map50_95[:num_classes]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detection Metrics by Class', fontsize=16, fontweight='bold')
    
    x = np.arange(len(class_names))
    width = 0.35
    
    # Precision
    bars1 = ax1.bar(x, precision, width, label='Precision', color='skyblue', alpha=0.8)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision per Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, precision):
        height = bar.get_height()
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Recall
    bars2 = ax2.bar(x, recall, width, label='Recall', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, recall):
        height = bar.get_height()
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # mAP@0.5
    bars3 = ax3.bar(x, map50, width, label='mAP@0.5', color='lightgreen', alpha=0.8)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('mAP@0.5')
    ax3.set_title('mAP@0.5 per Class')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars3, map50):
        height = bar.get_height()
        ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # mAP@0.5:0.95
    bars4 = ax4.bar(x, map50_95, width, label='mAP@0.5:0.95', color='gold', alpha=0.8)
    ax4.set_xlabel('Classes')
    ax4.set_ylabel('mAP@0.5:0.95')
    ax4.set_title('mAP@0.5:0.95 per Class')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=45)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars4, map50_95):
        height = bar.get_height()
        ax4.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}detection_metrics_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_classification_metrics(classification_results, class_names, save_dir, prefix=''):
    """
    Plot classification metrics in a bar chart format
    
    Args:
        classification_results: Dictionary with classification metrics
        class_names: List of class names  
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    precision = classification_results.get('precision_per_class', [])
    recall = classification_results.get('recall_per_class', [])
    f1 = classification_results.get('f1_per_class', [])
    
    # Ensure all arrays have the same length as class_names
    num_classes = len(class_names)
    precision = list(precision) + [0.0] * (num_classes - len(precision))
    recall = list(recall) + [0.0] * (num_classes - len(recall))
    f1 = list(f1) + [0.0] * (num_classes - len(f1))
    
    # Truncate to match class_names length
    precision = precision[:num_classes]
    recall = recall[:num_classes]
    f1 = f1[:num_classes]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Classification Metrics by Class', fontsize=16, fontweight='bold')
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # Precision
    bars1 = ax1.bar(x, precision, width, label='Precision', color='skyblue', alpha=0.8)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Precision')
    ax1.set_title('Classification Precision per Class')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, precision):
        height = bar.get_height()
        ax1.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Recall
    bars2 = ax2.bar(x, recall, width, label='Recall', color='lightcoral', alpha=0.8)
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Recall')
    ax2.set_title('Classification Recall per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, recall):
        height = bar.get_height()
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # F1 Score
    bars3 = ax3.bar(x, f1, width, label='F1 Score', color='lightgreen', alpha=0.8)
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Classification F1 Score per Class')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars3, f1):
        height = bar.get_height()
        ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}classification_metrics_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_metrics_table(detection_results, classification_results, class_names, save_dir, prefix=''):
    """
    Create a table-style visualization similar to the log output
    
    Args:
        detection_results: Dictionary with detection metrics
        classification_results: Dictionary with classification metrics
        class_names: List of class names
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with table layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Metrics Summary Table', fontsize=16, fontweight='bold')
    
    # Detection table data
    det_precision = detection_results.get('precision_per_class', [])
    det_recall = detection_results.get('recall_per_class', [])
    det_map50 = detection_results.get('map50_per_class', [])
    det_map50_95 = detection_results.get('map50_95_per_class', [])
    
    # Classification table data  
    cls_precision = classification_results.get('precision_per_class', [])
    cls_recall = classification_results.get('recall_per_class', [])
    cls_f1 = classification_results.get('f1_per_class', [])
    
    # Prepare detection table
    det_headers = ['Class', 'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    det_data = []
    
    for i, class_name in enumerate(class_names):
        row = [
            class_name,
            f"{det_precision[i]:.3f}" if i < len(det_precision) else "0.000",
            f"{det_recall[i]:.3f}" if i < len(det_recall) else "0.000", 
            f"{det_map50[i]:.3f}" if i < len(det_map50) else "0.000",
            f"{det_map50_95[i]:.3f}" if i < len(det_map50_95) else "0.000"
        ]
        det_data.append(row)
    
    # Prepare classification table
    cls_headers = ['Class', 'Precision', 'Recall', 'F1 Score']
    cls_data = []
    
    for i, class_name in enumerate(class_names):
        row = [
            class_name,
            f"{cls_precision[i]:.3f}" if i < len(cls_precision) else "0.000",
            f"{cls_recall[i]:.3f}" if i < len(cls_recall) else "0.000",
            f"{cls_f1[i]:.3f}" if i < len(cls_f1) else "0.000"
        ]
        cls_data.append(row)
    
    # Create detection table
    ax1.axis('tight')
    ax1.axis('off')
    det_table = ax1.table(cellText=det_data, colLabels=det_headers, 
                         cellLoc='center', loc='center',
                         colColours=['lightblue']*len(det_headers))
    det_table.auto_set_font_size(False)
    det_table.set_fontsize(10)
    det_table.scale(1.2, 2)
    ax1.set_title('Detection Metrics', fontsize=14, fontweight='bold', pad=20)
    
    # Create classification table
    ax2.axis('tight')
    ax2.axis('off')
    cls_table = ax2.table(cellText=cls_data, colLabels=cls_headers,
                         cellLoc='center', loc='center',
                         colColours=['lightgreen']*len(cls_headers))
    cls_table.auto_set_font_size(False)
    cls_table.set_fontsize(10)
    cls_table.scale(1.2, 2)
    ax2.set_title('Classification Metrics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{prefix}metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_all_metrics_visualizations(detection_results, classification_results, class_names, save_dir, prefix=''):
    """
    Create all metrics visualizations
    
    Args:
        detection_results: Dictionary with detection metrics
        classification_results: Dictionary with classification metrics
        class_names: List of class names
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    print(f"Creating metrics visualizations in {save_dir}")
    
    # Create individual charts
    plot_detection_metrics(detection_results, class_names, save_dir, prefix)
    plot_classification_metrics(classification_results, class_names, save_dir, prefix)
    plot_combined_metrics_table(detection_results, classification_results, class_names, save_dir, prefix)
    
    print(f"✅ Metrics visualizations saved:")
    print(f"   - {prefix}detection_metrics_chart.png")
    print(f"   - {prefix}classification_metrics_chart.png") 
    print(f"   - {prefix}metrics_table.png")
