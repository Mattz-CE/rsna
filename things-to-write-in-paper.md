# RSNA Cancer Detection Project Documentation (WIP)
## Infrastructure and Setup
- Multi-GPU configuration using PyTorch's DataParallel
- Distributed training across available GPUs
- Dynamic batch size scaling based on number of GPUs (16-64 per GPU based on VRAM)
  - 16GB GPU → 16-32 batch size
  - 24GB GPU → 32-48 batch size
  - 40GB GPU → 48-64 batch size
- Organized run directory structure with dedicated folders for:
  - Logs
  - Model checkpoints
  - TensorBoard visualization
  - Configuration files
  - Script versioning
  - Early Stopping (with patience of 3 epoches)

## Data Processing
- Image preprocessing pipeline:
  - Resizing to 512x512 pixels from DICOM
  - Single-channel grayscale input
  - Normalization (mean=0.5, std=0.5)
  - Data augmentation during training:
    - Random horizontal flips
    - Random vertical flips
- Dataset management using PyTorch DataLoader
- Efficient data loading with prefetching and parallel processing
- Dynamic worker scaling based on CPU cores

## Model Architectures
### ResNet Models
1. ResNet50V2 (Baseline)
   - Pretrained on ImageNet
   - Modified first conv layer for single-channel input
   - Custom top layer for binary classification
   - Transfer learning with selective layer unfreezing

2. ResNet101
   - Enhanced capacity compared to ResNet50
   - Similar architecture with deeper layers
   - Improved feature extraction capabilities

### Vision Transformer (ViT) Models
1. ViT Base
   - Patch size: 16x16
   - Pretrained on ImageNet
   - Modified patch embedding for single-channel input
   - Custom classification head

2. ViT MediumD
   - Advanced ViT variant with registers and global average pooling
   - Pretrained on ImageNet-12K and fine-tuned on ImageNet-1K
   - Enhanced patch processing
   - Improved attention mechanisms

### EfficientNetV2
- State-of-the-art efficient architecture
- Modified first conv layer for single-channel input
- Progressive layer unfreezing capability
- Optimized for mobile and edge devices

## Training Configuration
- Initial learning rate: 0.001
- Batch size: Dynamically scaled per GPU
- Training epochs: 35
- Validation split: 20%
- Learning rate scheduling:
  - ReduceLROnPlateau with patience=3
  - Factor=0.5 for learning rate reduction
- Transfer learning strategy:
  - Initial frozen layers except classification head
  - Progressive unfreezing of last n layers (default n=3)
  - Layer-specific learning rates

## Dataset Characteristics
- Severe class imbalance:
  - Positive cases: 1,158 (2.12%)
  - Negative cases: 53,548 (97.88%)
- Implementation of weighted BCE loss with positive class weighting
- Total dataset size: 54,706 images

## Metrics and Evaluation
- Primary metrics:
  - Area Under the Curve (AUC)
  - Binary Cross-Entropy Loss
  - Accuracy
- Secondary metrics:
  - Precision
  - Recall
  - F1 Score
- AUC difference monitoring for overfitting detection
- Validation performance tracking
- Custom metrics logging system

## Model Monitoring and Logging
- Comprehensive TensorBoard integration:
  - Loss curves (train/val)
  - Accuracy metrics
  - AUC tracking
  - Learning rate monitoring
  - Model architecture visualization
- Detailed logging system with both file and console output
- CSV logging of all metrics per epoch
- Configuration file versioning
- Automatic run directory management with epoch tracking

## Training Features
- Early stopping based on AUC difference:
  - Margin threshold: 0.03
  - Patience: 3 epochs
- Best model checkpointing based on:
  - Validation loss
  - Validation AUC
- Automatic learning rate adjustment
- GPU memory optimization
- Kaggle integration support

## Experimentation Space
- Cross-validation implementation
- Model ensemble strategies
- Additional data augmentation techniques
- Model interpretability features
- Automated hyperparameter optimization
- Alternative backbone architectures

## References
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [EfficientNetV2](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Best Practices](https://arxiv.org/abs/1911.02685)

## Models Architecture Details
### 1. Vision Transformer (ViT) Base
- Input: 512x512 single-channel images
- Patch size: 16x16
- Transformer blocks with self-attention
- Classification head with sigmoid activation
- Progressive unfreezing capability
- Pretrained weights from ImageNet

### 2. Vision Transformer (ViT) MediumD
- Enhanced ViT architecture
- Register tokens for improved feature capture
- Global average pooling
- Pretrained on ImageNet-12K
- Fine-tuned on ImageNet-1K
- Adaptive patch processing

### 3. ResNet101
- Deeper architecture compared to ResNet50
- Enhanced feature extraction
- Modified first conv layer for grayscale input
- Transfer learning optimization
- Selective layer unfreezing

### 4. EfficientNetV2
- Mobile-optimized architecture
- Compound scaling for depth/width
- Modified for single-channel input
- Progressive layer unfreezing
- Optimized inference performance
