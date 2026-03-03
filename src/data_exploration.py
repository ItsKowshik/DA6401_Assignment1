"""
W&B Report Experiment 2.1: Data Exploration
Logs a W&B Table containing 5 sample images from each of the 10 classes.


"""
#Import the libraries
import wandb
import numpy as np
from utils.data_loader import load_dataset

def log_sample_images(dataset_name='mnist'):
    """
    Logs a W&B Table containing 5 sample images from each of the 10 classes.
    
    Args:
        dataset_name: Name of dataset to load ('mnist' or 'fashion_mnist')
        
    Returns:
        None
    """
    
    # Initialize W&B run
    wandb.init(project="dl_1", name=f"data_exploration_{dataset_name}")
    print(f"Loading {dataset_name} dataset")
    X_train, y_train, _, _ = load_dataset(dataset_name)
    # Define class names
    if dataset_name == 'mnist':
        class_names = [str(i) for i in range(10)]
    else:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Create a W&B Table with columns
    columns = ["Class Name"] + [f"Sample {i+1}" for i in range(5)]
    table = wandb.Table(columns=columns)
    print("Extracting 5 samples per class")
    for class_idx in range(10):
        # Find indices where the label matches the current class
        indices = np.where(y_train == class_idx)[0]
        # Take the first 5 images for this class
        sample_images = []
        for i in range(5):
            img_array = X_train[indices[i]]
            sample_images.append(wandb.Image(img_array))
        # Add the row to the table
        table.add_data(class_names[class_idx], *sample_images)
    # Log the table to W&B
    wandb.log({"Dataset Samples": table})
    wandb.finish()
    print("Generated W&B Table with 5 samples per class")

if __name__ == "__main__":
    log_sample_images('fashion_mnist')