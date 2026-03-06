"""
W&B Report Experiment 2.3: Optimizer Showdown
Compares SGD, Momentum, NAG, RMSprop, using 3 hidden layers, 128 neurons, and ReLU.
"""

#Import the required libraries and models
import wandb
import numpy as np
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from argparse import Namespace

def run_optimizer_experiment(optimizer_name, X_train, y_train, X_test, y_test):
    """
    Runs a training experiment for a given optimizer and logs results to W&B.
    Args:
        optimizer_name (str): Name of the optimizer to test (e.g., 'sgd', 'momentum', 'nag', 'rmsprop').
        X_train (np.array): Training data features.
        y_train (np.array): Training data labels.
        X_test (np.array): Test data features.
        y_test (np.array): Test data labels.
    
    """
    args = Namespace(
        dataset='mnist',
        epochs=10, 
        batch_size=64,
        learning_rate=0.001,
        optimizer=optimizer_name,
        num_layers=3,
        hidden_size=[128, 128, 128],
        activation='relu',
        weight_init='xavier',
        loss='cross_entropy',
        wandb_project='dl_1'
    )
    
    wandb.init(
        project=args.wandb_project, 
        name=f"opt_showdown_{optimizer_name}", 
        group="optimizer_showdown_v2", 
        config=vars(args)
    )
    # Initialize the neural network and optimizer based on the current configuration
    nn = NeuralNetwork(args)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    # Training loop
    print(f"\n Training with {optimizer_name.upper()} ")

    for epoch in range(args.epochs):
        for i in range(0, X_train.shape[0], args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]
            # Forward pass, backward pass, and weight update
            logits = nn.forward(X_batch)
            grads = nn.backward(y_batch, logits)
            nn.update_weights(grads, optimizer)
            
        # Evaluate on training and validation sets
        train_loss, train_acc = nn.evaluate(X_train, y_train)
        val_loss, val_acc = nn.evaluate(X_test, y_test)
        # Log results to W&B
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        # Log the metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
    wandb.finish()

if __name__ == "__main__":
    print("Loading MNIST dataset")
    X_train, y_train, X_test, y_test = get_preprocessed_data('mnist')
    # Define the optimizers to test in the showdown
    optimizers_to_test = ['sgd', 'momentum', 'nag', 'rmsprop']
    # Run experiments for each optimizer
    for opt in optimizers_to_test:
        run_optimizer_experiment(opt, X_train, y_train, X_test, y_test)
        
    print("\n Check W&B for results")