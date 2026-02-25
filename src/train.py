"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse

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
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Mini-batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                        help='Learning rate for the optimizer')
    # Optimizer Configuration
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help="Optimizer algorithm to use")
    # Architecture Configuration
    parser.add_argument('--hidden_layers', type=int, default=1, 
                        help='Number of hidden layers in the network')
    parser.add_argument('--num_neurons', type=int, default=64, 
                        help='Number of neurons per hidden layer')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'sigmoid', 'tanh'],
                        help="Activation function for hidden layers")
    parser.add_argument('--weight_init', type=str, default='random',
                        choices=['random', 'zeros', 'xavier'],
                        help="Method to initialize weights")
    # Loss Function
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'mse'],
                        help="Loss function to compute gradients")
    # Logging and Saving
    parser.add_argument('--wandb_project', type=str, default='my-mlp-project', 
                        help='Weights & Biases project name for logging')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/model.npz', 
                        help='Relative path to save trained model weights')
    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    
    print("##Training Configuration\n")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        
        
    # TODO: Initialize WandB logging
    # TODO: Load Data using the dataloader
    # TODO: Initialize Neural Network
    # TODO: Training Loop
    # TODO: Save Model
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
