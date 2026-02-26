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
    # Model Loading
    parser.add_argument('--model_path', type=str, default='best_model.npy',
                        help='Relative path to saved model weights')
    # Architecture (Must match the trained model)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument('--num_neurons', type=int, default=64)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'])
    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--batch_size', type=int, default=32)
    # Weight initialization
    parser.add_argument('--weight_init', type=str, default='random') 
    
    return parser.parse_args()

def evaluate_model(model, X_test, y_test, loss_name='cross_entropy'): 
    """
    Evaluate model on test data and compute metrics.
        
    Returns:
        Dictionary containing logits, loss, accuracy, f1, precision, recall
    """
    # Get softmax probabilities
    y_pred = model.forward(X_test)
    # Calculate loss
    if loss_name == 'cross_entropy':
        loss = cross_entropy(y_test, y_pred)
    else:
        loss = mse(y_test, y_pred)
    true_labels = np.argmax(y_test, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    # Calculate all the metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    # Return metrics
    metrics = {
        'logits': y_pred,
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
        model_path: Path to .npz file
        
    Returns:
        None
    """
    try:
        # Load the .npz file
        saved_weights = np.load(model_path, allow_pickle=True).item()
        # Load the saved weights
        for i, layer in enumerate(nn.layers):
            layer.W = saved_weights[f'W_{i}']
            layer.b = saved_weights[f'b_{i}']
            
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}")
        print("Make sure you have trained the model and saved it first!")
        exit(1)


def main():
    """
    Main inference function.
    """
    args = parse_arguments()
    print(f"Loading {args.dataset} test dataset")
    _, _, X_test, y_test = get_preprocessed_data(args.dataset)
    print("Initializing Neural Network architecture")
    nn = NeuralNetwork(args)
    print("Loading saved weights")
    load_model(nn, args.model_path)
    print("Evaluating model")
    results = evaluate_model(nn, X_test, y_test, args.loss)
    print("EVALUATION RESULTS:\n")
    print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
if __name__ == '__main__':
    main()