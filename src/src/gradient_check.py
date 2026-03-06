"""
Gradient Check

This script checks the gradients of the network against the analytical gradients computed from backpropagation
This script verifies whether the gradients computed from backpropagation are correct and within the error limit mentioned in the image

"""

#Import the library
import numpy as np
import argparse
from ann.neural_network import NeuralNetwork
from ann.objective_functions import cross_entropy, mse


def run_gradient_check():
    """
    This function performs a gradient check on a simple neural network by comparing the analytical gradients computed from backpropagation with numerical gradients computed using finite differences. It uses a small dummy dataset and a simple network architecture to ensure that the gradients are correct and within the specified error limit.
    """
    # Setup a tiny dummy dataset
    np.random.seed(42)
    X = np.random.randn(5, 784)
    y_true = np.zeros((5, 10))
    for i in range(5):
        y_true[i, np.random.randint(0, 10)] = 1.0
    # Setup dummy configurations
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.num_layers = 1
    args.hidden_size = [64] 
    args.activation = 'relu'
    args.weight_init = 'random'
    args.loss = 'cross_entropy'
    args.dataset = 'mnist'
    
    # Initialize network
    nn = NeuralNetwork(args)
    # Compute loss
    def compute_loss(X_input):
        logits = nn.forward(X_input)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return cross_entropy(y_true, probs)
    
    # Compute both the forward and backward pass
    logits = nn.forward(X)
    analytical_grads = nn.backward(y_true, logits) # Backward now correctly accepts logits
    
    # Check the weights of the first hidden layer
    layer_idx = 0
    W_orig = nn.layers[layer_idx].W.copy()
    dW_analytical = analytical_grads[layer_idx][0]
    
    # Compute the gradients numerically
    epsilon = 1e-5
    dW_numerical = np.zeros((5, 5))
    # Iterate over a few specific weights to check
    for i in range(5):
        for j in range(5):
            # Add epsilon
            nn.layers[layer_idx].W[i, j] = W_orig[i, j] + epsilon
            loss_plus = compute_loss(X) 
            # Subtract epsilon
            nn.layers[layer_idx].W[i, j] = W_orig[i, j] - epsilon
            loss_minus = compute_loss(X)
            # Restore original weight
            nn.layers[layer_idx].W[i, j] = W_orig[i, j]
            # Compute numerical gradient
            dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
    # Compare with the relative error
    subset_analytical = dW_analytical[:5, :5]
    subset_numerical = dW_numerical
    
    numerator = np.linalg.norm(subset_analytical - subset_numerical)
    denominator = np.linalg.norm(subset_analytical) + np.linalg.norm(subset_numerical)
    relative_error = numerator / denominator if denominator != 0 else 0.0
    
    print(f"Relative Error: {relative_error}")
    if relative_error < 1e-7:
        print("Within error limit. Gradients match.")
    else:
        print("Outside error limit. Gradients do not match.")

if __name__ == '__main__':
    run_gradient_check()