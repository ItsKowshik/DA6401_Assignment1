"""
W&B Report Experiment: Error Analysis & Visualization
"""

import wandb
import os
import json
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.data_loader import get_preprocessed_data, load_dataset
from ann.neural_network import NeuralNetwork
from inference import load_model

def softmax(Z):
    """
    Applies the softmax function to convert logits into probabilities.
    Args:
        Z: Logits output from the neural network (shape: [batch_size, num_classes])
    Returns:
        Probabilities corresponding to each class (shape: [batch_size, num_classes])
    """
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def get_best_config():
    """
    Loads the best configuration from a JSON file and returns it as a Namespace object.
    Returns:
        Namespace object containing the best configuration parameters.
    """
    config_path = "best_config.json"
    if not os.path.exists(config_path):
        print(f"Could not find {config_path}.")
        exit(1)
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Namespace(**config_dict)

def run_error_analysis():
    # Initialize W&B run for error analysis
    wandb.init(project="dl_1", name="error_analysis_final_fix")
    # Setup model
    args = get_best_config()
    model_path = 'best_model.npy'
    nn = NeuralNetwork(args)
    weights = load_model(nn, model_path)
    nn.set_weights(weights)
    
    # Load data 
    print("Load preprocessed test data")
    _, _, X_test_prep, y_test_prep = get_preprocessed_data('mnist')
    # Load raw images specifically for the visual grid
    X_test_images = X_test_prep.reshape(-1, 28, 28)
    # Run inference
    print("Running inference")
    logits = nn.forward(X_test_prep)
    # Convert raw logits to real probabilities (0.0 to 1.0)
    probabilities = softmax(logits) 
    # Calculate probablities
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.argmax(y_test_prep, axis=1)
    max_probs = np.max(probabilities, axis=1)
    
    # Accuracy calculation
    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask
    final_acc = np.mean(correct_mask) * 100
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Test Accuracy: {final_acc:.1f}%)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    #plt.figtext(0.5, -0.05, 'This confusion matrix highlights the model\'s performance across all classes, showing where it excels and where it struggles.', ha='center', fontsize=10)
    wandb.log({"1_confusion_matrix": wandb.Image(plt)})
    plt.close()

    # Confidence Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs[correct_mask], bins=50, alpha=0.5, color='g', label='Correct')
    plt.hist(max_probs[incorrect_mask], bins=50, alpha=0.5, color='r', label='Incorrect')
    plt.title('Prediction Confidence (Probability Distribution)', fontsize=16, fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.legend()
    wandb.log({"2_confidence_distribution": wandb.Image(plt)})
    plt.close()

    # Error Grid: Showcasing the most confident mistakes
    incorrect_indices = np.where(incorrect_mask)[0]
    # Find the most confident mistakes (High probability assigned to the wrong class)
    top_mistake_indices = incorrect_indices[np.argsort(max_probs[incorrect_indices])][::-1][:10]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("Creative Visualization: High-Confidence Failures", fontsize=16, fontweight='bold')
    # Add a descriptive caption to explain the significance of this grid
    for i, idx in enumerate(top_mistake_indices):
        ax = axes.flatten()[i]
        # Use raw images for the plot, not flattened normalized ones
        ax.imshow(X_test_images[idx], cmap='gray')
        ax.set_title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}\nConf: {max_probs[idx]*100:.1f}%", color='red')
        ax.axis('off')
    
    plt.tight_layout()
    wandb.log({"3_creative_error_grid": wandb.Image(plt)})
    plt.close()
    # Finish W&B run
    wandb.finish()
    print("Check W&B for results.")

if __name__ == "__main__":
    run_error_analysis()