# RSNA Cancer Detection Project Documentation (WIP)
## Infrastructure and Setup
- Multi-GPU configuration using TensorFlow's MirroredStrategy
- Distributed training across available GPUs
- Dynamic batch size scaling based on number of GPUs
- Organized run directory structure with dedicated folders for:
  - Logs
  - Model checkpoints
  - TensorBoard visualization
  - Configuration files

## Data Processing
- Image preprocessing pipeline:
  - Resizing to 512x512 pixels 1024x1024 pixels from DICOM
  - Normalization (0-1 range)
  - Data augmentation during training:
    - Random horizontal flips
    - Random vertical flips
- Dataset management using TensorFlow data API
- Efficient data loading with prefetching and parallel processing

## Model Architecture
- Base model: ResNet50V2 pretrained on ImageNet
- Transfer learning approach with frozen base layers
- Custom top layer for binary classification
- Input shape: (img_size, img_size)
- Output: Single sigmoid unit for cancer detection

## Training Configuration
- Initial learning rate: 0.001
- Batch size per GPU: 256
- Training epochs: 35
- Validation split: 20%
- Learning rate scheduling:
  - ReduceLROnPlateau with patience=3
  - Factor=0.5 for learning rate reduction

## Dataset Characteristics
- Severe class imbalance:
  - Positive cases: 1,158 (2.12%)
  - Negative cases: 53,548 (97.88%)
- Implementation of class weights to handle imbalance
- Total dataset size: 54,706 images

## Metrics and Evaluation
- Comprehensive evaluation metrics:
  - Area Under the Curve (AUC)
  - Accuracy
  - Loss (Binary Cross-Entropy)
  - Precision
  - Recall
  - F1 Score
- Validation performance monitoring
- Custom callback for metrics logging

## Model Monitoring and Logging
- Detailed logging system with both file and console output
- TensorBoard integration for:
  - Loss and metrics visualization
  - Learning rate monitoring
  - Model architecture visualization
- CSV logging of all metrics per epoch
- Configuration file versioning

## Experiment Tracking
- Unique run IDs based on timestamp
- Complete experiment reproducibility through:
  - Saved configuration files
  - Copied training scripts
  - Detailed logging
  - Model checkpoints

## Additional Features
- Automatic best model checkpointing based on validation loss
  - Maybe should use AUC (but loss is balenced for classweights so doesn't matter)
- Early stopping capability
- Flexible configuration through JSON files
- Support for Kaggle dataset integration
- Error handling and GPU memory management

## Planed Features / TODO List
- RESNET101
- Test with ViT (referencing https://huggingface.co/microsoft/beit-large-patch16-512) finetuning base
- Test with yolov11

## Experimentation Space
- Implement cross-validation
- Experiment with other backbone architectures
- Add model ensemble capabilities
- Implement additional data augmentation techniques
- Add model interpretability features
- Integrate automated hyperparameter optimization

## References
- [How to train Vit](https://arxiv.org/abs/2010.11929)
- [ViT](https://arxiv.org/abs/2010.11929)
- [resnet](https://arxiv.org/abs/1512.03385)
- [Effinet](https://arxiv.org/abs/1905.11946)


## Models
### 1. ResNet50V2 with Class Weights
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 256, 256]           9,408
       BatchNorm2d-2         [-1, 64, 256, 256]             128
              ReLU-3         [-1, 64, 256, 256]               0
         MaxPool2d-4         [-1, 64, 128, 128]               0
            Conv2d-5         [-1, 64, 128, 128]           4,096
       BatchNorm2d-6         [-1, 64, 128, 128]             128
              ReLU-7         [-1, 64, 128, 128]               0
            Conv2d-8         [-1, 64, 128, 128]          36,864
       BatchNorm2d-9         [-1, 64, 128, 128]             128
             ReLU-10         [-1, 64, 128, 128]               0
           Conv2d-11        [-1, 256, 128, 128]          16,384
      BatchNorm2d-12        [-1, 256, 128, 128]             512
           Conv2d-13        [-1, 256, 128, 128]          16,384
      BatchNorm2d-14        [-1, 256, 128, 128]             512
             ReLU-15        [-1, 256, 128, 128]               0
       Bottleneck-16        [-1, 256, 128, 128]               0
           Conv2d-17         [-1, 64, 128, 128]          16,384
      BatchNorm2d-18         [-1, 64, 128, 128]             128
             ReLU-19         [-1, 64, 128, 128]               0
           Conv2d-20         [-1, 64, 128, 128]          36,864
      BatchNorm2d-21         [-1, 64, 128, 128]             128
             ReLU-22         [-1, 64, 128, 128]               0
           Conv2d-23        [-1, 256, 128, 128]          16,384
      BatchNorm2d-24        [-1, 256, 128, 128]             512
             ReLU-25        [-1, 256, 128, 128]               0
       Bottleneck-26        [-1, 256, 128, 128]               0
           Conv2d-27         [-1, 64, 128, 128]          16,384
      BatchNorm2d-28         [-1, 64, 128, 128]             128
             ReLU-29         [-1, 64, 128, 128]               0
           Conv2d-30         [-1, 64, 128, 128]          36,864
      BatchNorm2d-31         [-1, 64, 128, 128]             128
             ReLU-32         [-1, 64, 128, 128]               0
           Conv2d-33        [-1, 256, 128, 128]          16,384
      BatchNorm2d-34        [-1, 256, 128, 128]             512
             ReLU-35        [-1, 256, 128, 128]               0
       Bottleneck-36        [-1, 256, 128, 128]               0
           Conv2d-37        [-1, 128, 128, 128]          32,768
      BatchNorm2d-38        [-1, 128, 128, 128]             256
             ReLU-39        [-1, 128, 128, 128]               0
           Conv2d-40          [-1, 128, 64, 64]         147,456
      BatchNorm2d-41          [-1, 128, 64, 64]             256
             ReLU-42          [-1, 128, 64, 64]               0
           Conv2d-43          [-1, 512, 64, 64]          65,536
      BatchNorm2d-44          [-1, 512, 64, 64]           1,024
           Conv2d-45          [-1, 512, 64, 64]         131,072
      BatchNorm2d-46          [-1, 512, 64, 64]           1,024
             ReLU-47          [-1, 512, 64, 64]               0
       Bottleneck-48          [-1, 512, 64, 64]               0
           Conv2d-49          [-1, 128, 64, 64]          65,536
      BatchNorm2d-50          [-1, 128, 64, 64]             256
             ReLU-51          [-1, 128, 64, 64]               0
           Conv2d-52          [-1, 128, 64, 64]         147,456
      BatchNorm2d-53          [-1, 128, 64, 64]             256
             ReLU-54          [-1, 128, 64, 64]               0
           Conv2d-55          [-1, 512, 64, 64]          65,536
      BatchNorm2d-56          [-1, 512, 64, 64]           1,024
             ReLU-57          [-1, 512, 64, 64]               0
       Bottleneck-58          [-1, 512, 64, 64]               0
           Conv2d-59          [-1, 128, 64, 64]          65,536
      BatchNorm2d-60          [-1, 128, 64, 64]             256
             ReLU-61          [-1, 128, 64, 64]               0
           Conv2d-62          [-1, 128, 64, 64]         147,456
      BatchNorm2d-63          [-1, 128, 64, 64]             256
             ReLU-64          [-1, 128, 64, 64]               0
           Conv2d-65          [-1, 512, 64, 64]          65,536
      BatchNorm2d-66          [-1, 512, 64, 64]           1,024
             ReLU-67          [-1, 512, 64, 64]               0
       Bottleneck-68          [-1, 512, 64, 64]               0
           Conv2d-69          [-1, 128, 64, 64]          65,536
      BatchNorm2d-70          [-1, 128, 64, 64]             256
             ReLU-71          [-1, 128, 64, 64]               0
           Conv2d-72          [-1, 128, 64, 64]         147,456
      BatchNorm2d-73          [-1, 128, 64, 64]             256
             ReLU-74          [-1, 128, 64, 64]               0
           Conv2d-75          [-1, 512, 64, 64]          65,536
      BatchNorm2d-76          [-1, 512, 64, 64]           1,024
             ReLU-77          [-1, 512, 64, 64]               0
       Bottleneck-78          [-1, 512, 64, 64]               0
           Conv2d-79          [-1, 256, 64, 64]         131,072
      BatchNorm2d-80          [-1, 256, 64, 64]             512
             ReLU-81          [-1, 256, 64, 64]               0
           Conv2d-82          [-1, 256, 32, 32]         589,824
      BatchNorm2d-83          [-1, 256, 32, 32]             512
             ReLU-84          [-1, 256, 32, 32]               0
           Conv2d-85         [-1, 1024, 32, 32]         262,144
      BatchNorm2d-86         [-1, 1024, 32, 32]           2,048
           Conv2d-87         [-1, 1024, 32, 32]         524,288
      BatchNorm2d-88         [-1, 1024, 32, 32]           2,048
             ReLU-89         [-1, 1024, 32, 32]               0
       Bottleneck-90         [-1, 1024, 32, 32]               0
           Conv2d-91          [-1, 256, 32, 32]         262,144
      BatchNorm2d-92          [-1, 256, 32, 32]             512
             ReLU-93          [-1, 256, 32, 32]               0
           Conv2d-94          [-1, 256, 32, 32]         589,824
      BatchNorm2d-95          [-1, 256, 32, 32]             512
             ReLU-96          [-1, 256, 32, 32]               0
           Conv2d-97         [-1, 1024, 32, 32]         262,144
      BatchNorm2d-98         [-1, 1024, 32, 32]           2,048
             ReLU-99         [-1, 1024, 32, 32]               0
      Bottleneck-100         [-1, 1024, 32, 32]               0
          Conv2d-101          [-1, 256, 32, 32]         262,144
     BatchNorm2d-102          [-1, 256, 32, 32]             512
            ReLU-103          [-1, 256, 32, 32]               0
          Conv2d-104          [-1, 256, 32, 32]         589,824
     BatchNorm2d-105          [-1, 256, 32, 32]             512
            ReLU-106          [-1, 256, 32, 32]               0
          Conv2d-107         [-1, 1024, 32, 32]         262,144
     BatchNorm2d-108         [-1, 1024, 32, 32]           2,048
            ReLU-109         [-1, 1024, 32, 32]               0
      Bottleneck-110         [-1, 1024, 32, 32]               0
          Conv2d-111          [-1, 256, 32, 32]         262,144
     BatchNorm2d-112          [-1, 256, 32, 32]             512
            ReLU-113          [-1, 256, 32, 32]               0
          Conv2d-114          [-1, 256, 32, 32]         589,824
     BatchNorm2d-115          [-1, 256, 32, 32]             512
            ReLU-116          [-1, 256, 32, 32]               0
          Conv2d-117         [-1, 1024, 32, 32]         262,144
     BatchNorm2d-118         [-1, 1024, 32, 32]           2,048
            ReLU-119         [-1, 1024, 32, 32]               0
      Bottleneck-120         [-1, 1024, 32, 32]               0
          Conv2d-121          [-1, 256, 32, 32]         262,144
     BatchNorm2d-122          [-1, 256, 32, 32]             512
            ReLU-123          [-1, 256, 32, 32]               0
          Conv2d-124          [-1, 256, 32, 32]         589,824
     BatchNorm2d-125          [-1, 256, 32, 32]             512
            ReLU-126          [-1, 256, 32, 32]               0
          Conv2d-127         [-1, 1024, 32, 32]         262,144
     BatchNorm2d-128         [-1, 1024, 32, 32]           2,048
            ReLU-129         [-1, 1024, 32, 32]               0
      Bottleneck-130         [-1, 1024, 32, 32]               0
          Conv2d-131          [-1, 256, 32, 32]         262,144
     BatchNorm2d-132          [-1, 256, 32, 32]             512
            ReLU-133          [-1, 256, 32, 32]               0
          Conv2d-134          [-1, 256, 32, 32]         589,824
     BatchNorm2d-135          [-1, 256, 32, 32]             512
            ReLU-136          [-1, 256, 32, 32]               0
          Conv2d-137         [-1, 1024, 32, 32]         262,144
     BatchNorm2d-138         [-1, 1024, 32, 32]           2,048
            ReLU-139         [-1, 1024, 32, 32]               0
      Bottleneck-140         [-1, 1024, 32, 32]               0
          Conv2d-141          [-1, 512, 32, 32]         524,288
     BatchNorm2d-142          [-1, 512, 32, 32]           1,024
            ReLU-143          [-1, 512, 32, 32]               0
          Conv2d-144          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-145          [-1, 512, 16, 16]           1,024
            ReLU-146          [-1, 512, 16, 16]               0
          Conv2d-147         [-1, 2048, 16, 16]       1,048,576
     BatchNorm2d-148         [-1, 2048, 16, 16]           4,096
          Conv2d-149         [-1, 2048, 16, 16]       2,097,152
     BatchNorm2d-150         [-1, 2048, 16, 16]           4,096
            ReLU-151         [-1, 2048, 16, 16]               0
      Bottleneck-152         [-1, 2048, 16, 16]               0
          Conv2d-153          [-1, 512, 16, 16]       1,048,576
     BatchNorm2d-154          [-1, 512, 16, 16]           1,024
            ReLU-155          [-1, 512, 16, 16]               0
          Conv2d-156          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-157          [-1, 512, 16, 16]           1,024
            ReLU-158          [-1, 512, 16, 16]               0
          Conv2d-159         [-1, 2048, 16, 16]       1,048,576
     BatchNorm2d-160         [-1, 2048, 16, 16]           4,096
            ReLU-161         [-1, 2048, 16, 16]               0
      Bottleneck-162         [-1, 2048, 16, 16]               0
          Conv2d-163          [-1, 512, 16, 16]       1,048,576
     BatchNorm2d-164          [-1, 512, 16, 16]           1,024
            ReLU-165          [-1, 512, 16, 16]               0
          Conv2d-166          [-1, 512, 16, 16]       2,359,296
     BatchNorm2d-167          [-1, 512, 16, 16]           1,024
            ReLU-168          [-1, 512, 16, 16]               0
          Conv2d-169         [-1, 2048, 16, 16]       1,048,576
     BatchNorm2d-170         [-1, 2048, 16, 16]           4,096
            ReLU-171         [-1, 2048, 16, 16]               0
      Bottleneck-172         [-1, 2048, 16, 16]               0
AdaptiveAvgPool2d-173           [-1, 2048, 1, 1]               0
          Linear-174                    [-1, 1]           2,049
          ResNet-175                    [-1, 1]               0
         Sigmoid-176                    [-1, 1]               0
================================================================
Total params: 23,510,081
Trainable params: 23,510,081
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 1497.02
Params size (MB): 89.68
Estimated Total Size (MB): 1589.70
----------------------------------------------------------------

