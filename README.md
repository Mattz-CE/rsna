# RSNA Enhanced Breast Cancer Detection

## Overview

This project is focused on enhancing breast cancer detection in mammography using Vision Transformers (ViT) and CNN architectures. The codebase is designed to facilitate distributed training, model evaluation, and deployment of machine learning models for cancer detection. Weights are planned to be released in the near future.

## Features

### Infrastructure and Setup

- **Multi-GPU Configuration**: Utilizes PyTorch's DataParallel for distributed training across available GPUs.
- **Dynamic Batch Size Scaling**: Adjusts batch size based on GPU VRAM.
- **Organized Directory Structure**: Includes folders for logs, model checkpoints, TensorBoard visualizations, configuration files, and script versioning.

### Data Processing

- **Image Preprocessing**: Resizes DICOM images to 512x512 pixels, converts to single-channel grayscale, and normalizes.
- **Data Augmentation**: Includes random horizontal and vertical flips during training.
- **Efficient Data Loading**: Uses PyTorch DataLoader with prefetching and parallel processing.

### Model Architectures

- **ResNet Models**: Includes ResNet50V2 and ResNet101 with modifications for single-channel input.
- **Vision Transformer Models**: ViT Base and ViT MediumD with custom classification heads.
- **EfficientNetV2**: Optimized for mobile and edge devices with progressive layer unfreezing.

### Training Configuration

- **Learning Rate**: Initial rate set to 0.001 with ReduceLROnPlateau scheduling.
- **Early Stopping**: Based on AUC difference with a margin threshold of 0.03.
- **Transfer Learning**: Progressive unfreezing of layers with layer-specific learning rates.

### Metrics and Evaluation

- **Primary Metrics**: AUC, Binary Cross-Entropy Loss, and Accuracy.
- **Secondary Metrics**: Precision, Recall, and F1 Score.
- **Custom Metrics Logging**: Integrated with TensorBoard for comprehensive tracking.

### Model Monitoring and Logging

- **TensorBoard Integration**: Visualizes loss curves, accuracy metrics, AUC tracking, and more.
- **Detailed Logging**: Outputs to both file and console with CSV logging of metrics per epoch.

### Experimentation Space

- **Cross-Validation**: Implemented for robust model evaluation.
- **Model Ensemble Strategies**: Explored for improved performance.
- **Automated Hyperparameter Optimization**: Supports alternative backbone architectures.

## Getting Started

### Prerequisites

- Python 3.10 or later
- PyTorch, TensorFlow, and other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:   ```bash
   git clone https://github.com/Mattz-CE/rsna   ```

2. Install dependencies:   ```bash
   pip install -r requirements.txt   ```

### Running the Project

- **Training**: Execute `train.py` to start model training.
- **Deployment**: Use `deploy/start.sh` for setting up the deployment environment.

### Notebooks

- **Demo**: `demo.ipynb` provides a walkthrough of model predictions and visualizations.
- **History Viewer**: `history_viewer.ipynb` for analyzing past training runs.

## References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [EfficientNetV2](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Best Practices](https://arxiv.org/abs/1911.02685)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
