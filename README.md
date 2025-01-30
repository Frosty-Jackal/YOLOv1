# YOLOv1

## Overview
This repository provides an implementation of YOLOv1 (You Only Look Once) using PyTorch. YOLOv1 is a real-time object detection system that divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell. The provided implementation includes training and evaluation utilities.

## Key Files

- **train.py**: Main training script.
- **dataset.py**: Custom dataset class for loading and preprocessing data.
- **util.py**: Utility functions for various tasks such as calculating IoU (Intersection over Union).
- **eval.py**: Script for evaluating the trained model.
- **convert.py**: Script for converting the txt files to one csv file.

## Training

### Dataset
The training dataset should be placed in the `data` directory. It consists of:
- **train.csv**: A CSV file containing bounding box annotations and class labels for each image.
- **Images**: JPG images corresponding to the annotations in `train.csv`.

### Dependencies
- PyTorch
- torchvision
- OpenCV (cv2)
- PIL (Python Imaging Library)

### Training Script (train.py)
The `train.py` script initializes the training process, including:
- Loading the dataset using `MyDataset` class from `dataset.py`.
- Defining a custom neural network (`Mynet`) based on ResNet-34 pretrained model with additional convolutional and fully connected layers.
- Setting up the training loop, optimizer, and loss calculation.
- Saving the trained model state dictionary.

### Custom Neural Network (Mynet)
The `Mynet` class in `train.py` consists of:
- A ResNet-34 backbone with the last two layers removed.
- Additional convolutional layers with Leaky ReLU activation.
- Fully connected layers to predict bounding boxes, objectness scores, and class probabilities.

### Loss Calculation
The `calculate_loss` method computes the YOLOv1 loss

## Evaluation
The `eval.py` script can be used to evaluate the trained model on a validation or test dataset. It calculates metrics such as precision, recall, and mAP (mean Average Precision).

## Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Frosty-Jackal/YOLOv1.git
   cd YOLOv1
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision opencv-python PIL...
   ```

3. **Prepare the dataset**:
   - Place your images in the `data` directory.
   - Place your labels(should be yolo format) in the `labels` directory
   ```bash
   python convert.py
   ```
   - That's to Prepare a `train.csv` file with annotations in the required format(saved into data/).

4. **Train the model**:
   ```bash
   python train.py
   ```

5. **Evaluate the model** (after training):
   ```bash
   python eval.py
   ```

6. **Apply the model** :
   ```bash
   python application.py
   ```

Feel free to fork and modify this repository to suit your needs!
