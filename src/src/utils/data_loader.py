"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import fashion_mnist, mnist

def load_dataset(dataset_name):
    """
    Load and preprocess dataset
    
    TODO: Add support for other datasets
    
    Args:
        dataset_name: Name of dataset to load ('mnist' or 'fashion_mnist')
        
    Returns:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    """
    if dataset_name.lower() == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name.lower() == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")
    return X_train, y_train, X_test, y_test

def one_hot_encode(y, num_classes=10):
    """
    One-hot encode labels
    
    Args:
        y: Array of labels
        num_classes: Number of classes in the dataset
        
    Returns:
        One-hot encoded labels
    """
    # Create an array of zeros with shape
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1.0
    return one_hot

def get_preprocessed_data(dataset_name='mnist'):
    """
    Load and preprocess dataset
    
    Args:
        dataset_name: Name of dataset to load ('mnist' or 'fashion_mnist')
        
    Returns:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    """
    # Load the raw data
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    # Flatten the images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    # Normalize pixel values
    X_train_norm = X_train_flat.astype(np.float32) / 255.0
    X_test_norm = X_test_flat.astype(np.float32) / 255.0
    # One-hot encode the labels
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)
    return X_train_norm, y_train_encoded, X_test_norm, y_test_encoded


