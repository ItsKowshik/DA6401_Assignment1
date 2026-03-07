"""
W&B Report Experiment: Dead Neuron Analysis
Demonstrates the 'Dying ReLU' problem using a high learning rate, and compares it against Tanh.
"""
#Import the required libraries
import wandb
import numpy as np
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from argparse import Namespace

def run_activation_experiment(activation_type, X_train, y_train, X_test, y_test):
    """
    Runs an experiment to analyze dead neurons for a given activation type.
    Args:
        activation_type: 'relu' or 'tanh'
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
    Returns:
        None
    """
    args = Namespace(
        dataset='mnist',
        epochs=5, # Keep epochs low to quickly see the effect of dead neurons
        batch_size=64, # Standard batch size
        learning_rate=0.1, # Set to 0.1 to induce dead neurons
        optimizer='sgd', # SGD with high LR is more likely to cause dying ReLUs
        num_layers=3,
        hidden_size=[64, 64, 64], # Must be a list for our updated architecture
        activation=activation_type, 
        weight_init='random', # Random init helps induce dying ReLUs with high LR
        loss='cross_entropy', # Standard loss for classification
        wandb_project='dl_1' 
    )
    # Initialize W&B run
    wandb.init(
        project=args.wandb_project, 
        name=f"dead_neuron_investigation_{activation_type}", 
        group="dead_neuron_experiment_v4", 
        config=vars(args)
    )
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    wandb.define_metric(f"{activation_type}_grad_norm", step_metric="step")
    wandb.define_metric("dead_neuron_percentage", step_metric="step")
    wandb.define_metric("saturated_neuron_percentage", step_metric="step")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_accuracy", step_metric="epoch")
    # Create the neural network and optimizer
    nn = NeuralNetwork(args)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    
    print(f"\n Training with {activation_type.upper()} ")
    step = 0
    # Training loop
    for epoch in range(args.epochs):
        for i in range(0, X_train.shape[0], args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]
            # Forward and backward pass
            logits = nn.forward(X_batch)
            grads = nn.backward(y_batch, logits)
            # Calculate dead neurons
            total_dead_neurons = 0
            total_neurons = 0
            current_A = X_batch
            all_activations = []
            # Loop through hidden layers to track activations and identify dead neurons
            for layer in nn.layers[:-1]:
                current_A = layer.forward(current_A)
                all_activations.append(current_A)
                # A neuron is considered dead if its activation is <= 0 for ReLU or if it's saturated for Tanh 
                dead_neurons_in_layer = np.sum(np.max(current_A, axis=0) <= 0)
                total_dead_neurons += dead_neurons_in_layer
                total_neurons += current_A.shape[1]
                
            dead_percentage = (total_dead_neurons / total_neurons) * 100
            # Combine all hidden layer activations to log the overall distribution
            combined_activations = np.concatenate(all_activations, axis=1)
            # Track Gradients to explain convergence
            first_layer_grad_norm = np.linalg.norm(grads[0][0])
            # Log metrics and distributions based on activation type
        
            log_data = {
                "step": step,
                f"{activation_type}_grad_norm": first_layer_grad_norm,
                #f"{activation_type}_activation_distribution": wandb.Histogram(combined_activations)
            }
            # Log dead neuron percentage for ReLU and saturated neuron percentage for Tanh
            if activation_type == 'relu':
                log_data["dead_neuron_percentage"] = dead_percentage
            elif activation_type == 'tanh':
                # For Tanh, track saturated neurons
                saturated = np.sum(np.mean(np.abs(combined_activations), axis=0) > 0.99)
                log_data["saturated_neuron_percentage"] = (saturated / total_neurons) * 100
            # Log data
            wandb.log(log_data)
            # Update weights
            nn.update_weights(grads, optimizer)
            step += 1
        # Evaluate at the end of epoch    
        train_loss, train_acc = nn.evaluate(X_train, y_train)
        val_loss, val_acc = nn.evaluate(X_test, y_test)
        
        # Determine which metric to print
        perc_metric = log_data.get('dead_neuron_percentage', log_data.get('saturated_neuron_percentage', 0))
        # Print epoch results with the appropriate metric
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Dead/Sat Neurons: {perc_metric:.1f}%")
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_acc})
        
    wandb.finish()

if __name__ == "__main__":
    print("Load dataset")
    X_train, y_train, X_test, y_test = get_preprocessed_data('mnist')
    # Run experiments for both ReLU and Tanh to compare the effects of dead neurons
    activations_to_test = ['relu', 'tanh']
    for act in activations_to_test:
        run_activation_experiment(act, X_train, y_train, X_test, y_test)
    print("\nCheck W&B for results.")