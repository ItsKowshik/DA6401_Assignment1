"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
#Load the libraries
import os
import argparse
import numpy as np
import wandb
import json
#Load the models
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
import ast
from ann.optimizers import get_optimizer

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Dataset and Training Hyperparameters
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], 
                        help="Dataset to train on: 'mnist' or 'fashion_mnist'")
    parser.add_argument('--epochs', type=int, default=15, 
                        help='Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, default=32, 
                        help='Mini-batch size for training')
    #parser.add_argument('-lr','--learning_rate', type=float, default=0.01, 
                        #help='Learning rate for the optimizer')
    # Optimizer Configuration
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    # Architecture Configuration
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size',  nargs='+', default=[128, 128, 128], help='Neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    # Loss Function
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'mse'],
                        help="Loss function to compute gradients")
    # Logging and Saving
    parser.add_argument('--model_path', type=str, default='best_model.npy', help='Path to save/load the model')
    # W&B logging configuration
    parser.add_argument('-w_p', '--wandb_project', type=str, default='dl_1', help='Weights and Biases Project ID')
    
    
    # Parse the command-line arguments
    args, _ = parser.parse_known_args()
    clean_sizes = []
    for item in args.hidden_size:
        item_str = str(item)
        if item_str.startswith('['):
            clean_sizes.extend(ast.literal_eval(item_str))
        else:
            clean_sizes.append(int(item))
    # Update the hidden_size and num_layers attributes
    args.hidden_size = clean_sizes
    args.num_layers = len(clean_sizes)

    
    return args

def get_batches(X, y, batch_size):
    """
    Generate mini-batches from the given data.
    
    Args:
        X: Input data
        y: True labels
        batch_size: Mini-batch size
        
    Returns:
        Batches of data
    """
    m = X.shape[0]
    # Shuffle indices
    indices = np.arange(m)
    np.random.shuffle(indices)
    for i in range(0, m, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield X[batch_idx], y[batch_idx]

def save_model(nn, filepath):
    """
    Save model weights to a .npy file.
    
    Args:
        nn: NeuralNetwork object
        filepath: Path to save .npy file
        
    Returns:
        None
    """
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    model_params = {}
    for i, layer in enumerate(nn.layers):
        model_params[f'W_{i}'] = layer.W
        model_params[f'b_{i}'] = layer.b
        
    np.save(filepath, model_params)
    print(f"Model successfully saved to {filepath}")

def main():
    """
    Main training function.
    
    """
    args = parse_arguments()
    # Initialize Weights & Biases
    run = wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config
    if 'l1_size' in config:
        # Sizes for six possible layers
        master_sizes = [
            config.l1_size, config.l2_size, config.l3_size,
            config.l4_size, config.l5_size, config.l6_size
        ]
        # Slice the list to match the chosen number of layers
        actual_num_layers = config.num_layers
        actual_hidden_sizes = master_sizes[:actual_num_layers]
        args.num_layers = actual_num_layers
        args.hidden_size = actual_hidden_sizes
        # Update config so W&B dashboard
        wandb.config.update({
            "hidden_size": actual_hidden_sizes,
            "actual_architecture": str(actual_hidden_sizes)
        }, allow_val_change=True)
    print(f"Loading {args.dataset} dataset")
    # Preprocess the data
    X_train, y_train, X_test, y_test = get_preprocessed_data(args.dataset)
    print(f"Initializing Neural Network with {args.num_layers} hidden layers and hidden sizes {args.hidden_size}")
    nn = NeuralNetwork(config)
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    # The Training Loop
    
    tracker_file = f"{args.model_path}.acc_tracker"
    global_best_acc = 0.0
    if os.path.exists(tracker_file):
        with open(tracker_file, 'r') as f:
            try:
                global_best_acc = float(f.read().strip())
                print(f"Loaded previous global best accuracy: {global_best_acc:.4f}")
            except ValueError:
                pass
    print("Starting training")
    for epoch in range(args.epochs):
        train_loss_accum = 0.0
        correct_train_preds = 0
        total_train_samples = 0
        # Do mini batch gradient descent
        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            # Forward pass
            y_pred = nn.forward(X_batch)
            # Backward pass
            grads = nn.backward(y_batch, y_pred)
            # Update weights
            nn.update_weights(grads, optimizer)
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            correct_train_preds += np.sum(predictions == true_labels)
            total_train_samples += X_batch.shape[0]
            
        train_accuracy = correct_train_preds / total_train_samples
        # Evaluate on validation set at the end of the epoch
        val_loss, val_accuracy = nn.evaluate(X_test, y_test)
        # Accumulate train loss
        train_loss, _ = nn.evaluate(X_train, y_train)
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Wandb logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    
    # Save the best model
    if val_accuracy > global_best_acc:
            global_best_acc = val_accuracy
            print(f"New best validation accuracy ({global_best_acc:.4f})")
            print("Saving model weights\n")
            best_weights = nn.get_weights()
            np.save(args.model_path, best_weights)
            with open(tracker_file, 'w') as f:
                f.write(str(global_best_acc))
            config_file = "best_config.json"
            with open(config_file, 'w') as f:
                json.dump(vars(args), f, indent=4)
    # Finish W&B run
    wandb.finish()
    print("\nTraining complete")

if __name__ == '__main__':
    main()