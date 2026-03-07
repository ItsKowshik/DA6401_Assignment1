"""
W&B Report Experiment: Weight Initialization & Symmetry
Compares the gradients of 5 individual neurons using Zeros vs. Xavier initialization.
"""
#Load the required libraries and model
import wandb
import numpy as np
from utils.data_loader import get_preprocessed_data
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from argparse import Namespace

def run_symmetry_experiment(init_type, X_train, y_train):
    """Runs the experiment for a given weight initialization strategy."""
    args = Namespace(
        dataset='mnist',
        epochs=1, 
        batch_size=64,
        learning_rate=0.01,
        optimizer='sgd',
        num_layers=1,             
        hidden_size=[64],         
        activation='sigmoid', 
        loss='cross_entropy',
        weight_init=init_type,    # 'zeros' or 'xavier'
        wandb_project="dl_1"
    )

    wandb.init(
        project=args.wandb_project, 
        name=f"init_symmetry_{init_type}", 
        group="weight_initialization_analysis", 
        config=vars(args)
    )

    # Initialize the Neural Network
    nn = NeuralNetwork(args)
    # Force zeros
    if init_type == 'zeros':
        for layer in nn.layers:
            layer.W = np.zeros_like(layer.W)
            layer.b = np.zeros_like(layer.b)

    optimizer = get_optimizer(args.optimizer, args.learning_rate)

    print(f"\n Tracking 5 Neurons: {init_type.upper()} Initialization ")
    # Training Loop for exactly 50 steps
    for step in range(50):
        # Sequential sampling for the first 50 iterations
        start = step * args.batch_size
        end = start + args.batch_size
        X_batch = X_train[start:end]
        y_batch = y_train[start:end]
        # Forward and Backward pass
        logits = nn.forward(X_batch)
        grads = nn.backward(y_batch, logits)
        # Extract gradients for the first layer (input to hidden)
        dW_first_layer = grads[0][0]
        wandb.log({
            "iteration": step,
            "neuron_1_grad": np.linalg.norm(dW_first_layer[:, 0]),
            "neuron_2_grad": np.linalg.norm(dW_first_layer[:, 1]),
            "neuron_3_grad": np.linalg.norm(dW_first_layer[:, 2]),
            "neuron_4_grad": np.linalg.norm(dW_first_layer[:, 3]),
            "neuron_5_grad": np.linalg.norm(dW_first_layer[:, 4]),
        })
        
        # Update weights for BOTH runs to prove the symmetry trap cannot be escaped!
        nn.update_weights(grads, optimizer)

    wandb.finish()

if __name__ == "__main__":
    X_train, y_train, _, _ = get_preprocessed_data('mnist')
    
    for strategy in ['zeros', 'xavier']:
        run_symmetry_experiment(strategy, X_train, y_train)