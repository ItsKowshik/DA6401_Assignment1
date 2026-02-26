"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax
class Layer:
    def __init__(self, input_size, output_size, activation_name, weight_init='random'):
        """
        Initializes a fully connected (Dense) layer.
        Args:
            input_size (int): Number of input features/neurons from the previous layer
            output_size (int): Number of neurons in this layer
            activation_name (str): Name of the activation function ('relu', 'sigmoid', 'tanh', 'softmax', or None)
            weight_init (str): Method to initialize weights ('random', 'zeros', 'xavier')
            
        Attributes:
            input_size (int): Number of input features/neurons from the previous layer
            output_size (int): Number of neurons in this layer
            activation_name (str): Name of the activation function ('relu', 'sigmoid', 'tanh', 'softmax', or None)
            weight_init (str): Method to initialize weights ('random', 'zeros', 'xavier')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation_name.lower() if activation_name else None
        
        # Weight Initialization
        # Default: Small random numbers from a normal distribution
        # Zeros: All weights are initialized to zero
        if weight_init == 'zeros':
            self.W = np.zeros((input_size, output_size))
            self.b = np.zeros((1, output_size))
        # Xavier Initialization
        # Formula: sqrt(2.0 / (input_size + output_size))
        elif weight_init == 'xavier':
            limit = np.sqrt(2.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
            self.b = np.zeros((1, output_size))
        
        else:
            # Small random numbers from a normal distribution
            self.W = np.random.randn(input_size, output_size) * 0.01
            self.b = np.zeros((1, output_size))

        # Cache to store values for the backward pass
        self.cache = {}
        # Map the activation functions
        self._set_activation_functions()

    def _set_activation_functions(self):
        """
        Sets the activation functions based on the activation name.
        Args: self
        Returns: None
        """
        if self.activation_name == 'relu':
            self.act_func = relu
            self.act_deriv = relu_derivative
        elif self.activation_name == 'tanh':
            self.act_func = tanh
            self.act_deriv = tanh_derivative
        elif self.activation_name == 'softmax':
            self.act_func = softmax
            self.act_deriv = None  # No derivative for softmax
        elif self.activation_name == 'sigmoid':
            self.act_func = sigmoid
            self.act_deriv = sigmoid_derivative
        else:
            # Linear activation
            self.act_func = lambda x: x
            self.act_deriv = lambda x: np.ones_like(x)

    def forward(self, X):
        """
        Executes the forward pass for this layer.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Output of the layer
        """
        # Linear Transformation
        Z = np.dot(X, self.W) + self.b
        # Activation
        A = self.act_func(Z)
        # Store backpropagation
        self.cache['X'] = X
        self.cache['Z'] = Z
        self.cache['A'] = A
        #Return output
        return A

    def backward(self, dA=None, dZ_direct=None):
        """
        Executes the backward pass for this layer.
        
        Args:
            dA (np.ndarray): Gradient from the next layer
            dZ_direct (np.ndarray): Gradient from the next layer (used for the output layer)
            
        Returns:
            np.ndarray: Gradient to pass back to the previous layer
        """
        X = self.cache['X']
        Z = self.cache['Z']
        # Calculate local error
        if dZ_direct is not None:
            # Used for output layer: dZ = dA
            dZ = dZ_direct
        else:
            # Backpropagation to the previous layer
            dZ = dA * self.act_deriv(Z)
            
        # Calculate the weights and the biases
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        # Calculate dX
        dX = np.dot(dZ, self.W.T)
        # Return the derivative
        return dX, dW, db