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
    return np.mean(np.square(y_pred - y_true))

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
    sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
    return np.mean(sample_losses)

def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE loss with respect to the predictions.
    
    Args:
        y_true: True labels (one-hot encoded), shape (batch_size, num_classes)
        y_pred: Predicted probabilities/values, shape (batch_size, num_classes)
        
    Returns:
        float: Average MSE loss
    """
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
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true / y_pred_clipped) / y_true.shape[0]
