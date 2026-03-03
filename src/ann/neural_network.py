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
    def __init__(self, args):
        """
        Initializes the Neural Network.
        Args:
            args: Argument parser object containing configuration parameters

        Attributes:
            dataset (str): Dataset name ('mnist' or 'fashion_mnist')
            num_layers (int): Number of hidden layers
            hidden_size (int or list): Size of hidden layers (can be a single int or a list of ints for each layer)
            activation_str (str): Activation function for hidden layers
            
            weight_init (str): Weight initialization method
        """
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size 
        
        self.activation_str = args.activation
        self.weight_init = args.weight_init
        self.loss = args.loss
        # Determine input and output sizes based on dataset
        if self.dataset == 'mnist':
            self.input_size = 784
            self.output_size = 10
        elif self.dataset == 'fashion_mnist':
            self.input_size = 784
            self.output_size = 10
        self.layers = []
        self._build_network()
        
    def _build_network(self):
        """
        Build the neural network architecture based on the specified parameters.
        This method initializes the hidden layers and the output layer, ensuring that the input and output sizes are correctly connected between layers.
        
        """
        current_input_size = self.input_size
        # Determine the size of each hidden layer. If hidden_size is a single integer, we create a list of that size repeated for each layer.
        sizes = self.hidden_size if isinstance(self.hidden_size, list) else [self.hidden_size] * self.num_layers
        # Build hidden layers
        for i in range(self.num_layers):
            self.layers.append(Layer(current_input_size, sizes[i], self.activation_str, self.weight_init))
            current_input_size = sizes[i]  # Set the next layer's input size to this layer's output size!
        # The output layer expects current_input_size, which MUST now be sizes[-1]
        self.layers.append(Layer(current_input_size, self.output_size, 'linear', self.weight_init))
        
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        # Start with the input data as the initial activation
        A = X
        # Pass through all hidden layers first
        for layer in self.layers[:-1]: 
            A = layer.forward(A)
        # Manually compute the output layer's forward pass to ensure get the raw logits without applying the activation function
        logits = self.layers[-1].forward(A)
        # Return logits
        return logits
    
    def backward(self, y_true, logits):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        # Calculate the gradient of the loss with respect to the output layer's pre-activation (logits)
        grads = []
        m = y_true.shape[0] # batch size
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Cross-Entropy
        if self.loss == 'cross_entropy':
            dZ = (probabilities - y_true)/m
            dA_to_pass = dZ
        elif self.loss == 'mse':
            # MSE derivative with respect to logits (using probabilities for the output layer)
            dA = 2 * (probabilities - y_true) / y_true.shape[0]
            dZ = probabilities * (dA - np.sum(dA * probabilities, axis=1, keepdims=True))
            dA_to_pass = dZ
        # Output layer backward pass
        original_deriv = self.layers[-1].act_deriv
        self.layers[-1].act_deriv = lambda Z: 1.0  
        # Pass the gradient to the output layer
        dX, dW, db = self.layers[-1].backward(dA=dA_to_pass)
        grads.append((dW, db))
        # Restore the original derivative
        self.layers[-1].act_deriv = original_deriv
        # Propagate the gradient back through the hidden layers
        for layer in reversed(self.layers[:-1]):
            dX, dW, db = layer.backward(dA=dX)
            grads.append((dW, db))
        # Return the gradients in the correct order (from input layer to output layer)   
        return grads[::-1]
    
    def get_weights(self):
        """
        Get the current weights of the network.
        Returns:
            weights: Dictionary of weights and biases for each layer
        
        """
        # Create a dictionary to store the weights and biases for each layer
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f'W_{i}'] = layer.W.copy()
            weights[f'b_{i}'] = layer.b.copy()
        return weights
    
    def update_weights(self, grads, optimizer):
        """
        Update the weights using the specified optimizer.
        
        Args:
            grads: List of gradients
            optimizer: Optimizer object
            
        Returns:
            None
        """
        # The optimizer will handle the weight updates based on the gradients and its internal state
        optimizer.update(self.layers, grads)
        
    def set_weights(self, weights_dict):
        """
        Set the weights of the network from a given dictionary.
        Args:
            weights: Dictionary of weights and biases for each layer
            
        """
        # Update the weights and biases of each layer based on the provided dictionary
        weight_keys = sorted([k for k in weights_dict.keys() if k.startswith('W')])
        bias_keys = sorted([k for k in weights_dict.keys() if k.startswith('b')])

        print(f"Loading {len(weight_keys)} layers into a {len(self.layers)}-layer network...")

        if len(weight_keys) != len(self.layers):
            print(f"Warning: Architecture mismatch! File has {len(weight_keys)} layers, "
                f"but Network has {len(self.layers)}. Attempting partial load...")

        for i, (w_k, b_k) in enumerate(zip(weight_keys, bias_keys)):
            if i >= len(self.layers):
                break # Don't try to load more layers than the network has
                
            # Check shapes before assigning to prevent hard crashes
            if self.layers[i].W.shape == weights_dict[w_k].shape:
                self.layers[i].W = weights_dict[w_k].copy()
                self.layers[i].b = weights_dict[b_k].copy()
            else:
                raise ValueError(f"Shape mismatch at Layer {i+1}! "
                                f"Network expects {self.layers[i].W.shape}, "
                                f"but file has {weights_dict[w_k].shape}.")
        
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
        logits = self.forward(X)
        # Convert logits to probabilities using Softmax so the loss functions work correctly
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        # Calculate the loss
        if self.loss == 'cross_entropy':
            loss = cross_entropy(y, probabilities)
        else:
            loss = mse(y, probabilities)
        # Calculate accuracy
        predictions = np.argmax(logits, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        return loss, accuracy
