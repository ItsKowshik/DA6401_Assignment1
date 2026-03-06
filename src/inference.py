"""
Inference Script
Evaluate trained models on test sets
"""

#Load the libraries and the models
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import cross_entropy, mse
import json

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], 
                        help="Dataset to train on: 'mnist' or 'fashion_mnist'")
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, default=32, 
                        help='Mini-batch size for training')
    # Optimizer Configuration
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    # Architecture Configuration
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128], help='Neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    # Loss Function
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'mse'],
                        help="Loss function to compute gradients")
    # Logging and Saving
    parser.add_argument('--model_path', type=str, default='best_model.npy', help='Path to save/load the model')
    # W&B logging configuration
    parser.add_argument('-w_p', '--wandb_project', type=str, default='dl_1', help='Weights and Biases Project ID')
    
    return parser.parse_args()

def infer_architecture_from_weights(model_path):
    """
    Infer architecture from saved model weights
    
    Args:
        model_path: Path to saved model weights
    
    Returns:
        hidden_layers: Number of hidden layers
        num_neurons: Number of neurons in each hidden layer
 
    """
    try:
        weights = np.load(model_path, allow_pickle=True).item()
    
        # Safely find all weight keys (works with 'W1', 'W2' or 'W_0', 'W_1')
        w_keys = sorted([k for k in weights.keys() if k.startswith('W')])
        
        if not w_keys:
            raise ValueError("No weight matrices found in the file!")
            
        # The number of hidden layers is total weight matrices minus 1 (the output layer)
        num_hidden_layers = len(w_keys) - 1
        
        # Extract the hidden sizes by looking at the output dimension (shape[1]) 
        # of every weight matrix except the last one.
        hidden_sizes = []
        for key in w_keys[:-1]:
            hidden_sizes.append(weights[key].shape[1])
            
        return num_hidden_layers, hidden_sizes
        
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}")
        exit(1)

def evaluate_model(model, X_test, y_test, loss_name='cross_entropy'): 
    """
    Evaluate model on test data and compute metrics.
        
    Returns:
        Dictionary containing logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test) 
    # Convert logits to probabilities using Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    # Calculate loss using probabilities
    if loss_name == 'cross_entropy':
        loss = cross_entropy(y_test, probabilities)
    else:
        loss = mse(y_test, probabilities)
        
    true_labels = np.argmax(y_test, axis=1)
    # Get maximum probability
    pred_labels = np.argmax(logits, axis=1)
    # Calculate all the metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    # Return metrics
    metrics = {
        'logits': logits,
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics



def load_model(nn, model_path):
    """
    Load model weights from a .npz file.
    
    Args:
        nn: NeuralNetwork object
        model_path: Path to .npy file
        
    Returns:
        None
    """
    try:
        # Load the .npy file
        data = np.load(model_path, allow_pickle=True).item()
        return data
            
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}")
        exit(1)


def main():
    """
    Main inference function.
    
    """
    args = parse_arguments()
    config_path = "best_config.json"
    try:
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        # Load best config values into args
        for key, value in saved_config.items():
            setattr(args, key, value)
            
        print(f"Loaded blueprint from {config_path}")
        print(f"Architecture: {args.num_layers} layers, Sizes: {args.hidden_size}")
        print(f"Activation: {args.activation.upper()} | Loss: {args.loss.upper()}")
        
    except FileNotFoundError:
        print(f"Error: {config_path} missing. Run train.py first!")
        return
    
    # Load test dataset
    print(f"Loading {args.dataset} test dataset")
    _, _, X_test, y_test = get_preprocessed_data(args.dataset)
    # Initialize the model architecture based on the saved config
    print("Initializing Neural Network architecture")
    nn = NeuralNetwork(args)
    # Load the weights
    print("Loading saved weights")
    weights = np.load(args.model_path, allow_pickle=True).item()
    nn.set_weights(weights)
    print("Evaluating model")
    results = evaluate_model(nn, X_test, y_test, args.loss)
    print("Evaluation Results:\n")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    
    
    metrics_to_save = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "loss": float(results['loss']),
        "accuracy": float(results['accuracy']),
        "precision": float(results['precision']),
        "recall": float(results['recall']),
        "f1_score": float(results['f1'])
    }
    
    with open('inference.json', 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print("Results saved to inference.json")
if __name__ == '__main__':
    main()