import torch
import numpy as np
import random
import json
import os
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup, get_constant_schedule


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cuda: bool = False):
    """Return the appropriate torch device (cuda if available and requested, else cpu)."""
    if cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def create_optimizer(model, lr: float = 1e-3, weight_decay: float = 0.0, optimizer_type: str = 'AdamW'):
    """Create and return an optimizer for the model parameters."""
    if optimizer_type.lower() == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(optimizer, scheduler_type: str, num_training_steps: int, warmup_steps: int = 0):
    """Create and return a learning rate scheduler. Supports 'linear' and 'constant'."""
    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_type == 'constant':
        return get_constant_schedule(optimizer)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def save_metrics(metrics: dict, output_dir: str, filename: str = 'metrics.json'):
    """Save metrics dictionary to a JSON file in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


# --- Unit Test Stubs ---
def _test_set_seed():
    set_seed(42)
    assert isinstance(torch.rand(1).item(), float)

def _test_get_device():
    device = get_device(cuda=True)
    assert isinstance(device, torch.device)

def _test_create_optimizer():
    import torch.nn as nn
    model = nn.Linear(10, 2)
    optimizer = create_optimizer(model, lr=1e-3)
    assert hasattr(optimizer, 'step')

def _test_create_scheduler():
    import torch.nn as nn
    model = nn.Linear(10, 2)
    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, 'linear', num_training_steps=10, warmup_steps=2)
    assert hasattr(scheduler, 'step')

def _test_save_metrics():
    metrics = {'accuracy': 0.9}
    save_metrics(metrics, '/tmp', 'test_metrics.json')
    assert os.path.exists('/tmp/test_metrics.json')

if __name__ == "__main__":
    _test_set_seed()
    _test_get_device()
    _test_create_optimizer()
    _test_create_scheduler()
    _test_save_metrics()
