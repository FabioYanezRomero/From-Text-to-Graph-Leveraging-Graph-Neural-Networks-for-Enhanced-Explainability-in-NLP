"""
Utility functions for GNN training

This module provides helper functions for GNN training, including
seed setting, device selection, optimizer and scheduler creation,
and metric saving.
"""

import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(use_cuda=True):
    """
    Get the device to use for training.
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        torch.device: Device to use
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        if use_cuda and not torch.cuda.is_available():
            print("WARNING: CUDA is not available, using CPU instead.")
        return torch.device("cpu")

def create_optimizer(model, learning_rate, weight_decay):
    """
    Create an optimizer for the model.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer, scheduler_type, num_training_steps, num_warmup_steps):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('linear', 'cosine', 'constant')
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
    """
    if scheduler_type == 'constant':
        return None
    
    if scheduler_type == 'linear':
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def save_metrics(output_dir, **metrics):
    """
    Save metrics to JSON and plot them.
    
    Args:
        output_dir: Directory to save metrics
        **metrics: Metrics to save, where each key is a metric name and each value is a list of values
    """
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot metrics
    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(metric_values)
        plt.title(f"{metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{metric_name}.png"))
        plt.close()

def load_best_model(output_dir, model_class, model_args):
    """
    Load the best model from a training run.
    
    Args:
        output_dir: Directory containing the model
        model_class: Model class to instantiate
        model_args: Arguments to pass to the model constructor
        
    Returns:
        The loaded model
    """
    model = model_class(**model_args)
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    return model
