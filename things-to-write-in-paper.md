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
- Test with ViT
- Test with yolov11

## Experimentation Space
- Implement cross-validation
- Experiment with other backbone architectures
- Add model ensemble capabilities
- Implement additional data augmentation techniques
- Add model interpretability features
- Integrate automated hyperparameter optimization
- 