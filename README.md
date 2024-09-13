Classification with PyTorch

This repository contains code and explanations for building classification models using PyTorch. The goal is to guide you through the steps involved in preparing data, building models (including non-linear models), and evaluating their performance. We also cover multi-class classification, non-linear activation functions, and key classification metrics.
Table of Contents

    Data Preparation
        1.1 Check input and output shapes
        Turn data into tensors and create train and test splits

    Building a Model
        2.1 Setup loss function and optimizer
        Train model
        Make predictions and evaluate the model

    Training a Model on Linear Data
        5.1 Preparing data to see if our model can fit a straight line
        Understand limitations of fitting linear models to non-linear data

    The Missing Piece: Non-Linearity
        6.1 Recreating non-linear data (red and blue circles)
        6.2 Building a model with non-linearity
        6.3 Training a model with non-linearity
        6.4 Evaluating a model trained with a non-linear activation function

    Replicating Non-Linear Activation Functions
        Explore how non-linear activation functions help model more complex patterns.

    Multi-Class Classification with PyTorch
        8.1 Overview of multi-class classification
        8.2 Building a multi-class classification model in PyTorch
        8.3 Creating a loss function and optimizer for a multi-class classification model
        8.4 Getting prediction probabilities for a multi-class PyTorch model
        8.5 Creating a training loop and testing loop for a multi-class PyTorch model
        8.6 Making and evaluating predictions with a PyTorch multi-class model

    Classification Metrics
        A few more classification metrics for evaluating model performance, including accuracy, precision, recall, and F1 score.

Getting Started
Prerequisites

    Python 3.x
    PyTorch installed (torch and torchvision)
    Additional Python libraries: numpy, matplotlib, sklearn

You can install the required libraries by running:

bash

pip install torch torchvision numpy matplotlib scikit-learn

Running the Project

To run the project, follow these steps:

    Clone the repository:

    bash

git clone https://github.com/your-repo/classification-with-pytorch.git
cd classification-with-pytorch

Prepare your data:

    The script will automatically create and split datasets for training and testing.

Run the model training:

bash

    python train.py

    Evaluate the model:
        The evaluation results, including classification metrics and loss, will be printed in the console after training.

Code Structure

    data/: Contains the data and any relevant pre-processing scripts.
    models/: Contains the neural network models for linear and non-linear classification tasks.
    train.py: Main script to train and evaluate the models.
    utils.py: Helper functions for data manipulation, metric calculations, and visualization.

Results

    Linear Model: Fitting simple linear data to see how the model performs.
    Non-Linear Model: Exploring non-linear activation functions (like ReLU) to better capture complex patterns in data.
    Multi-Class Classification: Implementing a PyTorch model for multi-class problems with appropriate loss functions (e.g., CrossEntropyLoss).

Conclusion

Through this project, you will gain hands-on experience with:

    Preparing data for classification.
    Building and training models in PyTorch.
    Handling both linear and non-linear data.
    Implementing and evaluating multi-class classification models.
    Using classification metrics to evaluate performance.
