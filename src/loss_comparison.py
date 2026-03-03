"""
W&B Report Experiment: Loss Function Comparison
Compares the convergence rate of Mean Squared Error (MSE) vs. Cross-Entropy using the exact same architecture and learning rate.
"""

#Import the required libraries
import wandb
import numpy as np
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from argparse import Namespace

def run_loss_experiment(loss_name, X_train, y_train, X_test, y_test):
    args = Namespace(
        dataset='mnist',
        epochs=10,
        batch_size=64,
        learning_rate=0.1,  # A high learning rate for SGD to show convergence differences
        optimizer='sgd',  
        num_layers=2,       
        hidden_size=[64, 64], # Updated to list format
        activation='relu',
        weight_init='he',   
        loss=loss_name,
        wandb_project='dl_1'
    )
    # Initialize W&B
    wandb.init(
        project=args.wandb_project, 
        name=f"loss_{loss_name}_comparison", 
        group="loss_function_comparison", 
        config=vars(args)
    )
    # Initialize the neural network and optimizer
    nn = NeuralNetwork(args)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    
    print(f"\n Training with {loss_name.upper()} ")
    # Training loop
    for epoch in range(args.epochs):
        for i in range(0, X_train.shape[0], args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]
            # Forward and backward pass using logits
            logits = nn.forward(X_batch)
            grads = nn.backward(y_batch, logits)
            nn.update_weights(grads, optimizer)
        # Evaluate at the end of every epoch
        train_loss, train_acc = nn.evaluate(X_train, y_train)
        val_loss, val_acc = nn.evaluate(X_test, y_test)
        # Log metrics to W&B
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
        
    wandb.finish()

if __name__ == "__main__":
    print("Loading dataset")
    X_train, y_train, X_test, y_test = get_preprocessed_data('mnist')
    # Compare the two loss functions
    losses_to_test = ['cross_entropy', 'mse']
    for loss in losses_to_test:
        run_loss_experiment(loss, X_train, y_train, X_test, y_test)
        
    print("\nCheck W&B for results.")