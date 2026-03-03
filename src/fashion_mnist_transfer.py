"""
W&B Report Experiment 2.10: The Fashion-MNIST Transfer Challenge
Tests 3 strictly budgeted configurations on the Fashion-MNIST dataset, automatically loading the best MNIST config from JSON as the baseline.
"""

#Import the required libraries and models
import wandb
import json
import os
from argparse import Namespace
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer

def get_custom_args(name, layers, neurons, optimizer):
    """
    Helper function to create a Namespace of arguments for a given configuration.
    This allows us to easily create different configurations for our experiments while keeping the code clean and organized.
    
    Args:
        name (str): A unique name for the configuration, used for W&B logging.
        layers (int): The number of hidden layers in the neural network.
        neurons (int): The number of neurons in each hidden layer.
        optimizer (str): The optimizer to use for training (e.g., 'rmsprop', 'nag').
        
    Returns:
        Namespace: A Namespace object containing all the necessary arguments for training the model.
    """
    hidden_size = [neurons] * layers
    return Namespace(
        dataset='fashion_mnist',
        epochs=15,
        batch_size=64,
        learning_rate=0.001,
        config_name=name,
        num_layers=layers,
        hidden_size=hidden_size,
        activation='relu',
        optimizer=optimizer,
        weight_init='xavier',
        loss='cross_entropy',
        wandb_project='dl_1'
    )

def run_fashion_challenge(args, X_train, y_train, X_test, y_test):
    """
    Runs a single configuration of the Fashion-MNIST Transfer Challenge, training a neural network with the specified arguments and logging results to W&B.
    Args:
        args (Namespace): A Namespace object containing all the necessary arguments for training the model.
        X_train (ndarray): The training data features.
        y_train (ndarray): The training data labels.
        X_test (ndarray): The test data features.
        y_test (ndarray): The test data labels.
    

    Returns:
        None: This function does not return anything, but it logs the training progress and results to W&B. 
    """
    wandb.init(
        project=args.wandb_project, 
        name=f"fashion_{args.config_name}", 
        group="fashion_transfer_challenge_v2", 
        config=vars(args)
    )
    # Initialize the neural network with the specified architecture and hyperparameters from args, and set up the optimizer based on the chosen configuration.
    nn = NeuralNetwork(args)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    # Log the initial configuration and hyperparameters to W&B for tracking and comparison across different runs.
    print(f"\n Training {args.config_name.upper()} on Fashion-MNIST ")
    best_val_acc = 0.0
    # Loop through the specified number of epochs, performing training and evaluation at each epoch, and logging the results to W&B for analysis and comparison across different configurations.
    for epoch in range(args.epochs):
        for i in range(0, X_train.shape[0], args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]
            # Forward pass through the neural network to get the predicted logits for the current batch of training data.
            logits = nn.forward(X_batch)
            grads = nn.backward(y_batch, logits)
            nn.update_weights(grads, optimizer)
        train_loss, train_acc = nn.evaluate(X_train, y_train)
        val_loss, val_acc = nn.evaluate(X_test, y_test)
        # Update the best validation accuracy if the current epoch's validation accuracy is higher than the previously recorded best, which will help us identify the winning configuration at the end of the challenge.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        # Log the training loss, validation loss, and validation accuracy for the current epoch to W&B, allowing us to track the training progress and compare different configurations effectively.
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
    print(f" Best Validation Accuracy for {args.config_name}: {best_val_acc:.4f}")
    wandb.finish()

if __name__ == "__main__":
    print("Loading Fashion-MNIST dataset")
    X_train, y_train, X_test, y_test = get_preprocessed_data('fashion_mnist')
    configs = [
        get_custom_args("MNIST Baseline", layers=1, neurons=128, optimizer='rmsprop'),
        get_custom_args("Deep_Hierarchy", layers=3, neurons=128, optimizer='rmsprop'),
        get_custom_args("Deep_Hierarchy_Nag", layers=3, neurons=128, optimizer='nag')
    ]
    for config in configs:
        run_fashion_challenge(config, X_train, y_train, X_test, y_test)
        
    print("\n Check W&B for the results")