
# PyTorch Circle Classification Model

This repository contains code for building and training a neural network using PyTorch to classify circular patterns from synthetic data. The project demonstrates the following key aspects of machine learning and deep learning:

- Data preparation and visualization.
- Model building with `torch.nn.Module` and `torch.nn.Sequential`.
- Forward pass, backward propagation, and gradient descent.
- Model evaluation using classification metrics such as accuracy.
- Visualizing decision boundaries.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Future Improvements](#future-improvements)

## Project Overview

This project demonstrates binary classification using a synthetic dataset (`make_circles`) generated with scikit-learn. It showcases the following stages:
1. **Data Generation and Visualization**: Create a dataset of concentric circles using `make_circles`.
2. **Neural Network Construction**: Build a neural network with multiple hidden layers and non-linear activation functions.
3. **Training**: Use a binary cross-entropy loss function and stochastic gradient descent optimizer to train the model.
4. **Evaluation**: Assess the model's performance using accuracy and visualize its decision boundary.
5. **Extending to Multi-Class Classification**: Demonstrate how the techniques scale to multi-class classification using blobs data.

## Dependencies

Before running the code, ensure you have the following packages installed:

```bash
pip install torch torchvision matplotlib pandas scikit-learn
```

Optional:
```bash
pip install torchmetrics
```

## Dataset

1. **Binary Classification**:
   - The dataset is created using `make_circles`, generating two classes of data (inner and outer circles).
   - Data is split into training and test sets using `train_test_split`.

2. **Multi-Class Classification**:
   - A multi-class dataset is generated using `make_blobs` with 4 clusters and split into training and test sets.

## Model Architecture

1. **Binary Classification Model**:
   - Two neural networks are constructed, one using `nn.Module` and another with `nn.Sequential`.
   - The models contain the following layers:
     - Input layer: 2 features (from circle data).
     - Hidden layers: ReLU activation functions to introduce non-linearity.
     - Output layer: 1 output for binary classification.

2. **Multi-Class Classification Model**:
   - The multi-class model has:
     - Input layer: 2 features.
     - Hidden layers: ReLU activation and linear layers.
     - Output layer: 4 outputs (for 4 classes).

## Training

The model is trained using the following steps:

1. **Forward Pass**: The input data is passed through the layers of the neural network.
2. **Loss Calculation**: The loss is computed using `BCELoss` or `BCEWithLogitsLoss` for binary classification and `CrossEntropyLoss` for multi-class classification.
3. **Backpropagation**: Gradients are calculated via `loss.backward()`.
4. **Optimizer Step**: The optimizer (`SGD`) updates the model weights.

Training is conducted over 1000 epochs for binary classification and 100 epochs for multi-class classification.

## Evaluation

The evaluation includes:
1. **Accuracy Calculation**: Use a custom accuracy function to calculate the percentage of correct predictions.
2. **Decision Boundary Visualization**: The decision boundaries of both the training and testing datasets are visualized.
3. **Prediction Probabilities**: For multi-class classification, prediction probabilities are calculated using `torch.softmax()`.

## Usage

To use the code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pytorch-circle-classification.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python main.py
    ```

4. Observe the outputs and visualizations:
    - The accuracy, loss, and decision boundaries will be printed and plotted as the training progresses.

## Future Improvements

- **Add more advanced optimization algorithms**: Incorporate optimizers like Adam or RMSprop for faster convergence.
- **Experiment with different architectures**: Add more hidden layers or different types of activation functions like `tanh`.
- **Add early stopping**: Implement early stopping to avoid overfitting.
- **Improve evaluation metrics**: Use additional metrics such as F1 Score, Precision, and Recall for a more thorough evaluation.
