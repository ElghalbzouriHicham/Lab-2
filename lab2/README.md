# Lab2 : Classification on MNIST with PyTorch

## Objective
This lab aims to provide hands-on experience with the PyTorch library to develop neural network architectures for computer vision tasks. By the end of this lab, you will be familiar with building CNN, Faster R-CNN, and Vision Transformer (ViT) models, evaluating their performance on the MNIST dataset, and comparing the results.

## Dataset
The dataset used for this exercise is the MNIST handwritten digits dataset. You can download it from Kaggle: URL : [MNIST Dataset on Kaggle] (https://www.kaggle.com/datasets/hojjatk/mnist-dataset) .

## Part 1: CNN Classifier
### 1. Establish a CNN Architecture:
Define a custom CNN architecture using PyTorch.
Use layers such as Convolution, Pooling, and Fully Connected layers.
Set up hyperparameters including kernel size, padding, stride, optimizer, and regularization techniques.
Train the CNN model on the MNIST dataset, utilizing a GPU if available.
Epoch 1, Loss: 0.3812  Epoch 2, Loss: 0.1158  Epoch 3, Loss: 0.0875  Epoch 4, Loss: 0.0740  Epoch 5, Loss: 0.0645  Epoch 6, Loss: 0.0586  Epoch 7, Loss: 0.0524  Epoch 8, Loss: 0.0481  Epoch 9, Loss: 0.0438  Epoch 10, Loss: 0.0411
**Test Accuracy**: 98.5%  
**F1 Score**: 98.5%

### 2. Faster R-CNN for MNIST Classification:
Build a Faster R-CNN model suitable for MNIST classification.
Train the model on the MNIST dataset using GPU mode.

### 3. Fine-Tuning with Pretrained Models (VGG16):
Use pretrained models (VGG16 and AlexNet) from the PyTorch model.
Fine-tune these models on the MNIST dataset.
Epoch 1/5, Loss: 0.1267 Epoch 2/5, Loss: 0.0258 Epoch 3/5, Loss: 0.0174
Epoch 4/5, Loss: 0.0139 Epoch 5/5, Loss: 0.0102
**Training Time**: 4265.38 seconds
**Accuracy**: 99.21%
**F1-Score**: 99.21%


### 4. Fine-Tuning with Pretrained Models (AlexNet):
Epoch 1/10, Loss: 0.3625 Epoch 2/10, Loss: 0.0457 Epoch 3/10, Loss: 0.0317 Epoch 4/10, Loss: 0.0254 Epoch 5/10, Loss: 0.0197 Epoch 6/10, Loss: 0.0177 Epoch 7/10, Loss: 0.0138 Epoch 8/10, Loss: 0.0115 Epoch 9/10, Loss: 0.0101 Epoch 10/10, Loss: 0.0083
**Training Time**: 742.85 seconds
**Accuracy**: 99.6%
**F1-Score**: 99.6%

## Part 2: Vision Transformer (ViT)
### 1.Implement Vision Transformer (ViT):
- **Loss during training**:
- Epoch 1: Loss = 1.2380 Epoch 2: Loss = 0.3352 ... Epoch 19: Loss = 0.0352 Epoch 20: Loss = 0.0329
- **Test Accuracy**: 99.23%  
- **Observations**: ViT demonstrated competitive performance compared to CNNs and pre-trained models while providing a novel transformer-based architecture.

## Results and Comparison
| Model           | Accuracy | F1-Score | Training Time (seconds) | Notes                                         |
|------------------|----------|----------|--------------------------|-----------------------------------------------|
| CNN             | 98.5%   | 98.5%   | -                        | 	Custom architecture; efficient and suitable for simple tasks.      |
| VGG16           | 99.21%   | 99.21%   | 4265.38                  | High accuracy; slower training due to deep layers and complex structure.        |
| AlexNet         | 99.6%  | 99.6%   | 742.85                | Balanced performance; relatively fast training time among pre-trained models.                   |
| Vision Transformer | 99.23%   | 99.23%   | -                        | Strong performance using transformer-based design; effective but resource-intensive.|
| Faster R-CNN    | -        | -        | -                        | Not suited for simple classification; better for object detection tasks; high computational cost.|

## Requirements
### Software and Libraries
**Programming Language**: Python
**Framework**: PyTorch
**Additional Libraries**: torchvision, idx2numpy, tqdm, scikit-learn, matplotlib
### Hardware
**GPU Support**: Recommended for faster training times, especially for large models.
