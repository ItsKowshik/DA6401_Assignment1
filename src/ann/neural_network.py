"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
#Load the libraries
import numpy as np
from .neural_layer import Layer
from .objective_functions import cross_entropy, cross_entropy_derivative, mse, mse_derivative

class NeuralNetwork:
    """
    Neural Network class
    Handles forward and backward propagation loops
    
    Attributes:
        layers (list): List of neural layers
        loss_name (str): Name of the loss function
        
    Methods:
        forward(X): Forward propagation through all layers
        backward(y_true, y_pred): Backward propagation to compute gradients
    """
    def __init__(self, cli_args):
        """
        Initialize the network
        
        Args:
            cli_args: Command line arguments
            
        Returns:
            None
        """
        self.layers = []
        self.loss_name = cli_args.loss
        # Flattened input size
        input_size = 784 
        # Output is always 10 classes
        output_size = 10 
        # Build Hidden layer
        current_input_size = input_size
        for _ in range(cli_args.hidden_layers):
            self.layers.append(Layer(current_input_size, cli_args.num_neurons, cli_args.activation, cli_args.weight_init))
            current_input_size = cli_args.num_neurons
        # Output layer
        self.layers.append(Layer(current_input_size, output_size,'softmax', cli_args.weight_init))
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        A = X
        for layer in self.layers:
            # Forward Operation
            A = layer.forward(A)
        return A
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        grads = []
        m = y_true.shape[0] # batch size
        # Cross-Entropy
        if self.loss_name == 'cross_entropy':
            dZ = (y_pred - y_true) / m
            dX, dW, db = self.layers[-1].backward(dZ_direct=dZ)
        #MSE
        else:
            dA = mse_derivative(y_true, y_pred)
            dX, dW, db = self.layers[-1].backward(dA=dA)
            
        grads.append((dW, db))
        # Compute gradients for the hidden layers
        for layer in reversed(self.layers[:-1]):
            dA = dX
            dX, dW, db = layer.backward(dA=dA)
            grads.append((dW, db))
        # Reverse the grads list
        grads.reverse()
        #Return the grads
        return grads
    
    def update_weights(self, grads, optimizer):
        """
        Update the weights using the specified optimizer.
        
        Args:
            grads: List of gradients
            optimizer: Optimizer object
            
        Returns:
            None
        """
        optimizer.update(self.layers, grads)
        
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Shift the training section and integrate into the main file to make the modular system better
        """
        pass
    
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            loss: Loss value
            accuracy: Accuracy
        """
        # Get the predictions
        y_pred = self.forward(X)
        # Calculate the loss
        if self.loss_name == 'cross_entropy':
            loss = cross_entropy(y, y_pred)
        else:
            loss = mse(y, y_pred)
        # Calculate the accuracy
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        # Return loss and accuracy
        return loss, accuracy
