"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function.
    Squashes values to the range (0, 1).
    Formula: 1 / (1 + exp(-x))
    Args:
        x: Input data
        
    Returns:
        Output of the sigmoid function
    """
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid function.
    Formula: sigmoid(x) * (1 - sigmoid(x))
    Args:
        x: Input data
        
    Returns:
        Derivative of the sigmoid function
    """
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh(x):
    """
    Hyperbolic tangent activation function.
    Squashes values to the range (-1, 1).
    
    Args:
        x: Input data
        
    Returns:
        Output of the tanh function
    """
    return np.tanh(x)

def tanh_derivative(x):
    
    """
    Derivative of the tanh function.
    Formula: 1 - tanh(x)**2
    
    Args:
        x: Input data
        
    Returns:
        Derivative of the tanh function
    """
    t = tanh(x)
    return 1.0 - t**2

def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    Squashes negative values to 0.
    
    Args:
        x: Input data
        
    Returns:
        Output of the ReLU function
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU function.
    Returns 1 if x > 0, else 0.
    """
    return (x > 0).astype(float)

def softmax(x):
    """
    Softmax activation function.
    
    Args:
        x: Input data
        
    Returns:
        Output of the softmax function
    """
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted_x)
    return exps / np.sum(exps, axis=-1, keepdims=True)