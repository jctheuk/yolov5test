# YOLOv5 Dual-Task Training (Detection + Classification)

This repository contains an enhanced version of YOLOv5 that supports simultaneous object detection and image classification tasks, specifically optimized for echocardiogram analysis.

## 🎯 Key Features

- **Dual-Task Architecture**: Simultaneous object detection and image classification
- **Enhanced Classification Head**: Sophisticated feature extraction with multiple pooling strategies
- **Optimized Loss Function**: Balanced loss computation for both tasks
- **Improved Data Loading**: Proper handling of classification labels
- **Comprehensive Training**: Full training pipeline with validation and metrics
- **Seamless Integration**: Works with the existing `train.py` script

## 🏗️ Architecture Overview

The model architecture consists of:

1. **Backbone**: YOLOv5 backbone for feature extraction
2. **Detection Head**: Standard YOLOv5 detection head for object detection
3. **Classification Head**: Enhanced classification head with:
   - Feature extraction layers (Conv2d + BatchNorm + SiLU)
   - Dual pooling (Average + Max)
   - Multi-layer classifier with dropout
   - Proper weight initialization

## 📁 File Structure

```
yolov5c/
├── models/
│   ├── common.py                 # Enhanced YOLOv5WithClassification class
│   ├── yolo.py                   # Modified Model class for dual outputs
│   └── yolov5sc.yaml            # Model configuration
├── utils/
│   ├── dual_loss.py             # Dual-task loss computation
│   ├── dataloaders.py           # Enhanced data loading
│   ├── general.py               # Utility functions (parse_model_output)
│   └── val.py                   # Enhanced validation with classification metrics
├── data/
│   └── hyps/
│       └── hyp.scratch-low.yaml # Optimized hyperparameters
├── train.py                     # Enhanced training script (dual-task ready)
├── test_dual_training.py        # Test script for dual-task functionality
└── README_DUAL_TRAINING.md      # This file
```

## 🚀 Quick Start

### 1. Prepare Your Dataset

Your dataset should be organized as follows:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

### 2. Label Format

Each label file should contain:

**Detection labels** (first line):
```
class_id x_center y_center width height
```

**Classification labels** (second line):
```
class_index
```

Example:
```
0 0.5 0.5 0.2 0.3
1
```

### 3. Dataset Configuration

Create a `data.yaml` file:

```yaml
# Dataset configuration
path: ../path/to/your/dataset  # dataset root directory
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 4  # number of detection classes
names: ['AR', 'MR', 'PR', 'TR']  # detection class names

# Classification
num_cls: 3  # number of classification classes
cls_names: ['PSAX', 'PLAX', 'A4C']  # classification class names
```

### 4. Training

**Use the existing `train.py` script with dual-task support:**

```bash
python train.py \
    --data data.yaml \
    --cfg models/yolov5sc.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/train \
    --name dual_exp
```

### 5. Testing

Before training, you can test the dual-task functionality:

```bash
python test_dual_training.py
```

This will verify that:
- Model creation works correctly
- Forward pass produces dual outputs
- Loss computation handles both tasks
- Data loading works with classification labels

## 🔧 Key Improvements

### 1. Enhanced Classification Head

The `YOLOv5WithClassification` class now includes:

- **Feature Extraction**: Multiple convolutional layers with batch normalization
- **Dual Pooling**: Both average and max pooling for better feature representation
- **Multi-layer Classifier**: Deep classifier with dropout for regularization
- **Proper Initialization**: Kaiming initialization for better training

### 2. Improved Loss Function

The `ComputeDualLoss` class provides:

- **Balanced Loss**: Dynamic weighting between detection and classification losses
- **Label Smoothing**: Improved generalization for classification
- **Gradient Clipping**: Prevents gradient explosion
- **Proper Target Handling**: Handles one-hot and class index formats

### 3. Enhanced Data Loading

- **Proper Label Parsing**: Handles both detection and classification labels
- **Batch Processing**: Efficient batch collation
- **Error Handling**: Robust error handling for malformed labels

### 4. Integrated Training

- **Seamless Integration**: Works with existing `train.py` script
- **Dual Metrics**: Tracks both detection and classification performance
- **Validation**: Enhanced validation with classification metrics
- **Logging**: Comprehensive logging of both tasks

## 📊 Training Metrics

The training script tracks:

- **Detection Loss**: Box, object, and class losses
- **Classification Loss**: Cross-entropy loss for classification
- **Classification Accuracy**: Per-epoch accuracy
- **Validation Metrics**: Separate validation metrics for both tasks

## 🎛️ Hyperparameters

Key hyperparameters in `data/hyps/hyp.scratch-low.yaml`:

```yaml
# Loss coefficients
box: 0.05          # box loss gain
cls: 0.5           # cls loss gain
obj: 1.0           # obj loss gain
cls_task: 0.3      # classification task loss weight

# Classification-specific
label_smoothing: 0.1  # label smoothing for classification
```

## 🔍 Troubleshooting

### Common Issues

1. **Poor Classification Performance**:
   - Check label format and ensure classification labels are correct
   - Verify `num_cls` parameter matches your dataset
   - Try adjusting `cls_task` weight in hyperparameters

2. **Training Instability**:
   - Reduce learning rate
   - Increase batch size if memory allows
   - Check for label inconsistencies

3. **Memory Issues**:
   - Reduce batch size
   - Use smaller image size
   - Enable gradient checkpointing

### Debug Mode

Enable debug logging by setting:

```python
# In train.py
DEBUG = True
```

This will print detailed information about:
- Batch processing
- Model outputs
- Loss computation
- Label parsing

## 📈 Performance Tips

1. **Data Quality**: Ensure high-quality, well-labeled data
2. **Augmentation**: Use appropriate augmentations for your domain
3. **Learning Rate**: Start with lower learning rates for dual-task training
4. **Batch Size**: Use larger batch sizes if memory allows
5. **Regularization**: Use dropout and label smoothing for better generalization

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_dual_training.py
```

This will test:
- Model creation and forward pass
- Dual loss computation
- Data loading with classification labels
- Integration with existing training pipeline

## 📝 Usage Examples

### Basic Training

```bash
# Train from scratch
python train.py --data data.yaml --cfg models/yolov5sc.yaml --epochs 100

# Train from pretrained weights
python train.py --data data.yaml --cfg models/yolov5sc.yaml --weights yolov5s.pt --epochs 100

# Train with custom hyperparameters
python train.py --data data.yaml --cfg models/yolov5sc.yaml --hyp data/hyps/hyp.scratch-low.yaml
```

### Advanced Training

```bash
# Multi-GPU training
python -m torch.distributed.run --nproc_per_node 4 train.py --data data.yaml --cfg models/yolov5sc.yaml --batch-size 64

# Resume training
python train.py --data data.yaml --cfg models/yolov5sc.yaml --weights runs/train/dual_exp/weights/last.pt --resume
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the AGPL-3.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original YOLOv5 implementation by Ultralytics
- Echocardiogram dataset providers
- Research community for feedback and improvements

## 🔄 Migration from train_dual.py

If you were previously using `train_dual.py`, you can now use the existing `train.py` with the same functionality:

**Old (train_dual.py):**
```bash
python train_dual.py --data data.yaml --weights yolov5s.pt --epochs 100
```

**New (train.py):**
```bash
python train.py --data data.yaml --cfg models/yolov5sc.yaml --weights yolov5s.pt --epochs 100
```

The functionality is identical - the dual-task training has been seamlessly integrated into the main training script.
