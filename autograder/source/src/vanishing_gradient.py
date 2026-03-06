"""
W&B Report Experiment: Vanishing Gradient Analysis
Compares gradient norms of the first hidden layer between Sigmoid and ReLU in a deep network (5 hidden layers) using Adam optimizer.
"""
# Import the required libraries
import wandb
import numpy as np
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer

class Args:
    """
    Class to store arguments
    
    Attributes:
        dataset (str): Dataset name
        epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        hidden_layers (int): Number of hidden layers
        
        num_neurons (int): Number of neurons per hidden layer
        activation (str): Activation function
        
        optimizer (str): Optimizer name
        weight_init (str): Weight initialization method
        loss (str): Loss function
    """
    def __init__(self, activation_name, num_layers):
        """
        Initialize the Args class
        
        Args:
            activation_name (str): Activation function
            
        Returns:
            None
        """
        # Set default values for all arguments, but allow activation and depth to be customized
        self.dataset = 'mnist'
        self.epochs = 1 
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.optimizer = 'rmsprop' 
        self.num_layers = num_layers 
        self.hidden_size = [128] * num_layers
        self.activation = activation_name
        self.weight_init = 'he' if activation_name == 'relu' else 'xavier'
        self.loss = 'cross_entropy'
        self.wandb_project = "dl_1"

def run_depth_experiment(X_train, y_train):
    depths_to_test = [2, 3, 4, 5, 6, 7]
    activations = ['sigmoid', 'relu']
    # Initialize a single W&B run to log all comparisons
    wandb.init(
        project="dl_1", 
        name="vanishing_gradient_depth_comparison", 
        group="vanishing_gradient_analysis"
    )
    
    for act in activations:
        print(f"\n Testing {act.upper()} across different depths ")
        for depth in depths_to_test:
            args = Args(act, depth)
            nn = NeuralNetwork(args)
            
            # Calculate the grad norms for the first hidden layer across a few batches  
            grad_norms = []
            for i in range(0, args.batch_size * 5, args.batch_size):
                X_batch = X_train[i:i+args.batch_size]
                y_batch = y_train[i:i+args.batch_size]
                # Forward and Backward pass
                logits = nn.forward(X_batch)
                grads = nn.backward(y_batch, logits)
                # Extract the gradient matrix (dW) for the first hidden layer
                dW_first_layer = grads[0][0]
                grad_norms.append(np.linalg.norm(dW_first_layer))
            # Average the initial gradient norms
            avg_grad_norm = np.mean(grad_norms)
            print(f"Depth: {depth} Layers | First Layer Grad Norm: {avg_grad_norm:.6f}")
            
            # Log to W&B
            wandb.log({
                "network_depth": depth,
                f"{act}_first_layer_grad_norm": avg_grad_norm
            })
            
    wandb.finish()

if __name__ == "__main__":
    print("Loading dataset")
    X_train, y_train, _, _ = get_preprocessed_data('mnist')
    run_depth_experiment(X_train, y_train)
    print("\nCheck W&B for results")