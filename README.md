# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

This assignment requires you to implement a neural network from scratch using only NumPy. You will build all components including layers, activations, optimizers, and loss functions, then train your network on MNIST or Fashion-MNIST datasets.
# Key Features

## Optimization Algorithms
- Stochastic Gradient Descent (**SGD**)
- **Momentum**
- **Nesterov Accelerated Gradient (NAG)**
- **RMSprop**

## Activation Functions
- **ReLU**
- **Sigmoid**
- **Tanh**

## Loss Functions
- **Cross-Entropy**
- **Mean Squared Error (MSE)**

## Regularization
- **L2 Weight Decay** to reduce overfitting

## Weight Initialization
- **Random Initialization**
- **Xavier (Glorot) Initialization**

## MLOps Integration
- Full **Weights & Biases (W&B)** integration
- Real-time **experiment tracking**
- Automated **hyperparameter sweeps**

# Project Structure

The folder structure is according to the assignment instructions

```
Assignment_1/
├── src/
│   ├── ann/                      # Core Neural Network Modules
│   │   ├── __init__.py
│   │   ├── activations.py        # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── neural_layer.py       # Individual dense layer logic
│   │   ├── neural_network.py     # Full network builder & backprop
│   │   ├── objective_functions.py# Cross-Entropy, MSE
│   │   └── optimizers.py         # SGD, Momentum, NAG, RMSprop
│   │
│   ├── utils/
│   │   └── data_loader.py        # Dataset fetching & preprocessing
│   │
│   ├── train.py                  # Main training script & CLI
│   ├── inference.py              # Evaluation & metric scoring
│   ├── best_config.json          # Saved optimal hyperparameters
│   └── best_model.npy            # Saved trained weights
│
├── force_zip.py                  # Autograder submission packager
├── .gitignore                    # Git tracking rules
└── README.md
```


#  Installation & Setup

## Clone the Repository

```bash
git clone <your-repository-url>
cd Assignment_1
```


## Log in to Weights & Biases

To track experiments:

```bash
wandb login
```


#  Training the Model (CLI)

The `train.py` script provides a **Command Line Interface (CLI)** to configure your neural network architecture and training hyperparameters.

## Basic Training Run

```bash
python src/train.py \
--dataset mnist \
--epochs 10 \
--batch_size 32 \
-lr 0.01 \
-o momentum \
--num_layers 2 \
-sz 128 64 \
-a relu \
-wi xavier \
--loss cross_entropy
```


# Full List of Arguments

| Argument | Flag | Description | Default | Options |
|--------|------|-------------|--------|--------|
| dataset | --dataset | Target dataset | mnist | mnist, fashion_mnist |
| epochs | --epochs | Number of training epochs | 15 | Any integer |
| batch size | -b | Mini-batch size | 32 | Any integer |
| optimizer | -o | Optimization algorithm | rmsprop | sgd, momentum, nag, rmsprop |
| learning rate | -lr | Learning rate | 0.001 | Any float |
| weight decay | -wd | L2 regularization penalty | 0.0 | Any float |
| num layers | -nhl | Number of hidden layers | 3 | Any integer |
| hidden size | -sz | Neurons per hidden layer | 128 128 128 | List of integers |
| activation | -a | Activation function | relu | sigmoid, tanh, relu |
| weight init | -wi | Initialization method | xavier | random, xavier |
| loss | --loss | Loss function | cross_entropy | cross_entropy, mse |
| wandb project | -w_p | W&B project name | dl_1 | Any string |


#  Weights & Biases Sweeps (Hyperparameter Tuning)

This project supports **automated hyperparameter tuning** using **W&B Sweeps**.

The architecture dynamically reshapes based on sweep configurations.

## Initialize a Sweep

```bash
wandb sweep sweep.yaml
```

## Start the Sweep Agent

```bash
wandb agent <your-username/project-name/sweep-id>
```


#  Evaluation & Inference

After training, the model automatically saves the following files inside `src/`:

- `best_model.npy` → trained weights  
- `best_config.json` → architecture configuration  
- `best_model.npy.acc_tracker` → validation metric logs  

## Run Inference

To evaluate the model on the test dataset:

```bash
python src/inference.py
```

This script computes the **F1-score** and other evaluation metrics.


# Running W&B Experiments

To log specific comparisons to your **Weights & Biases dashboard**, run the following commands.  
All runs will automatically sync to your designated **W&B project** (default: `dl_1`).


#  Optimizer Showdown

Compare how different optimization algorithms converge.

```bash
# SGD
python src/train.py -o sgd -lr 0.01 --wandb_project dl_1

# Momentum
python src/train.py -o momentum -lr 0.01 --wandb_project dl_1

# Nesterov Accelerated Gradient (NAG)
python src/train.py -o nag -lr 0.01 --wandb_project dl_1

# RMSprop
python src/train.py -o rmsprop -lr 0.001 --wandb_project dl_1
```

#  The Vanishing Gradient Experiment

Observe how **deep networks behave with different activation functions**.

```bash
# Deep network with Sigmoid (Watch the gradients vanish!)
python src/train.py -nhl 5 -sz 64 64 64 64 64 -a sigmoid --wandb_project dl_1

# Deep network with ReLU (Watch the gradients flow!)
python src/train.py -nhl 5 -sz 64 64 64 64 64 -a relu --wandb_project dl_1
```

#  Weight Initialization Symmetry

See why **Xavier initialization is critical for deep networks**.

```bash
# Random Initialization
python src/train.py -wi random -a tanh --wandb_project dl_1

# Xavier Initialization
python src/train.py -wi xavier -a tanh --wandb_project dl_1
```

#  Loss Function Comparison

Compare **Mean Squared Error** against standard **Cross-Entropy** for classification.

```bash
# Mean Squared Error
python src/train.py --loss mse -lr 0.05 --wandb_project dl_1

# Cross-Entropy
python src/train.py --loss cross_entropy -lr 0.01 --wandb_project dl_1
```


Each run will automatically log:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Hyperparameters
- Model configuration
