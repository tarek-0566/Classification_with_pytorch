PyTorch Circle Classification Model

This repository contains code for building and training a neural network using PyTorch to classify circular patterns from synthetic data. The project demonstrates the following key aspects of machine learning and deep learning:

    Data preparation and visualization.
    Model building with torch.nn.Module and torch.nn.Sequential.
    Forward pass, backward propagation, and gradient descent.
    Model evaluation using classification metrics such as accuracy.
    Visualizing decision boundaries.

Table of Contents

    Project Overview
    Dependencies
    Dataset
    Model Architecture
    Training
    Evaluation
    Usage
    Future Improvements

Project Overview

This project demonstrates binary classification using a synthetic dataset (make_circles) generated with scikit-learn. It showcases the following stages:

    Data Generation and Visualization: Create a dataset of concentric circles using make_circles.
    Neural Network Construction: Build a neural network with multiple hidden layers and non-linear activation functions.
    Training: Use a binary cross-entropy loss function and stochastic gradient descent optimizer to train the model.
    Evaluation: Assess the model's performance using accuracy and visualize its decision boundary.
    Extending to Multi-Class Classification: Demonstrate how the techniques scale to multi-class classification using blobs data.

Dependencies

Before running the code, ensure you have the following packages installed:
pip install torch torchvision matplotlib pandas scikit-learn
Optional:
pip install torchmetrics
Dataset

    Binary Classification:
        The dataset is created using make_circles, generating two classes of data (inner and outer circles).
        Data is split into training and test sets using train_test_split.

    Multi-Class Classification:
        A multi-class dataset is generated using make_blobs with 4 clusters and split into training and test sets.

Model Architecture

    Binary Classification Model:
        Two neural networks are constructed, one using nn.Module and another with nn.Sequential.
        The models contain the following layers:
            Input layer: 2 features (from circle data).
            Hidden layers: ReLU activation functions to introduce non-linearity.
            Output layer: 1 output for binary classification.

    Multi-Class Classification Model:
        The multi-class model has:
            Input layer: 2 features.
            Hidden layers: ReLU activation and linear layers.
            Output layer: 4 outputs (for 4 classes).

Training

The model is trained using the following steps:

    Forward Pass: The input data is passed through the layers of the neural network.
    Loss Calculation: The loss is computed using BCELoss or BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multi-class classification.
    Backpropagation: Gradients are calculated via loss.backward().
    Optimizer Step: The optimizer (SGD) updates the model weights.

Training is conducted over 1000 epochs for binary classification and 100 epochs for multi-class classification.
Evaluation

The evaluation includes:

    Accuracy Calculation: Use a custom accuracy function to calculate the percentage of correct predictions.
    Decision Boundary Visualization: The decision boundaries of both the training and testing datasets are visualized.
    Prediction Probabilities: For multi-class classification, prediction probabilities are calculated using torch.softmax().

Usage

To use the code, follow these steps:

    Clone the repository:
    git clone https://github.com/yourusername/pytorch-circle-classification.git
Install dependencies:
pip install -r requirements.txt
Run the script:
python main.py
    Observe the outputs and visualizations:
        The accuracy, loss, and decision boundaries will be printed and plotted as the training progresses.

Future Improvements

    Add more advanced optimization algorithms: Incorporate optimizers like Adam or RMSprop for faster convergence.
    Experiment with different architectures: Add more hidden layers or different types of activation functions like tanh.
    Add early stopping: Implement early stopping to avoid overfitting.
    Improve evaluation metrics: Use additional metrics such as F1 Score, Precision, and Recall for a more thorough evaluation.
