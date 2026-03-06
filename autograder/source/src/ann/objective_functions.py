"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def mse(y_true, y_pred):
    """
    Computes the Mean Squared Error (MSE) loss.
    
    Args:
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        y_pred: Predicted probabilities/values, shape (batch_size, num_classes)
        
    Returns:
        float: Average MSE loss
    """
    m = y_true.shape[0]
    # Calculate the squared differences, sum them, and divide by m
    loss = np.sum((y_true - y_pred) ** 2) / m
    
    return loss

def cross_entropy(y_true, y_pred):
    """
    Computes the Cross-Entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        y_pred: Predicted probabilities/values, shape (batch_size, num_classes)
        
    Returns:
        float: Average cross-entropy loss
    """
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    # Calculate loss per sample, then average over the batch
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    
    return loss

def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE loss with respect to the predictions.
    
    Args:
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        y_pred: Predicted probabilities/values, shape (batch_size, num_classes)
        
    Returns:
        float: Average MSE loss
    """
    # The derivative of MSE with respect to the predictions is 2 * (y_pred - y_true) / number of samples
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_derivative(y_true, y_pred):
    """
    Derivative of cross-entropy loss with respect to the predictions.
    Args:
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        y_pred: Predicted probabilities/values, shape (batch_size, num_classes)
        
    Returns:
        float: Average cross-entropy loss
    """
    # To prevent division by zero, we clip the predicted probabilities
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true / y_pred_clipped) / y_true.shape[0]
