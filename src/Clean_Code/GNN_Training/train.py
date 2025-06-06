"""
GNN Training Script

This script handles the training, validation, and testing of Graph Neural Networks
for text classification tasks using PyTorch Geometric.
"""

import os
import json
import time
import random
import shutil
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score

from .gnn_models import GNN_Classifier, RGNN_Classifier
from .data_loader import load_graph_data
from .utils import set_seed, get_device, save_metrics, create_optimizer, create_scheduler

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train GNN models for text classification")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="setfit/ag_news",
                        help="Name of the dataset (e.g., setfit/ag_news, stanfordnlp/sst2)")
    parser.add_argument("--data_dir", type=str, default="/app/src/Clean_Code/output/embeddings/graphs",
                        help="Directory containing the graph data")
    parser.add_argument("--label_source", type=str, default="llm", choices=["original", "llm"],
                        help="Source of labels to use: 'original' or 'llm'")
    
    # Model arguments
    parser.add_argument("--module", type=str, default="GCNConv", 
                        choices=["GCNConv", "GATConv", "GraphConv", "SAGEConv", "RGCNConv", "RGATConv"],
                        help="GNN module type")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--layer_norm", action="store_true",
                        help="Use layer normalization")
    parser.add_argument("--residual", action="store_true",
                        help="Use residual connections")
    parser.add_argument("--pooling", type=str, default="max", choices=["max", "mean", "add"],
                        help="Graph pooling method")
    parser.add_argument("--num_relations", type=int, default=3,
                        help="Number of relation types (for relational GNNs)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine", "constant"],
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for the scheduler")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA for training")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--output_dir", type=str, default="/app/src/Clean_Code/output/gnn_results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    return args

def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Train the model for one epoch.
    
    Args:
        model: GNN model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training
        criterion: Loss function
        
    Returns:
        tuple: (average loss, true labels, predicted labels)
    """
    model.train()
    total_loss = 0
    all_true_labels = []
    all_pred_labels = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        if hasattr(batch, 'edge_type') and 'R' in model.__class__.__name__:
            # For relational GNNs
            logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        else:
            # For regular GNNs
            logits = model(batch.x, batch.edge_index, batch.batch)
        
        # Compute loss
        loss = criterion(logits, batch.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        pred_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
        true_labels = batch.y.detach().cpu().numpy()
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_true_labels, all_pred_labels

def evaluate(model, dataloader, device, criterion):
    """
    Evaluate the model.
    
    Args:
        model: GNN model
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation
        criterion: Loss function
        
    Returns:
        tuple: (average loss, true labels, predicted labels)
    """
    model.eval()
    total_loss = 0
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            if hasattr(batch, 'edge_type') and 'R' in model.__class__.__name__:
                # For relational GNNs
                logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            else:
                # For regular GNNs
                logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss
            loss = criterion(logits, batch.y)
            
            # Track metrics
            total_loss += loss.item()
            pred_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
            true_labels = batch.y.detach().cpu().numpy()
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_true_labels, all_pred_labels

def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.dataset_name.replace('/', '_')}_{args.module}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = get_device(args.cuda)
    print(f"Using device: {device}")
    
    # Prepare data paths
    dataset_provider, dataset_name = args.dataset_name.split('/')
    train_data_dir = os.path.join(args.data_dir, dataset_provider, dataset_name, f"train_{args.label_source}_labels")
    test_data_dir = os.path.join(args.data_dir, dataset_provider, dataset_name, f"test_{args.label_source}_labels")
    
    # Load data
    print(f"Loading training data from {train_data_dir}")
    train_dataset, train_loader = load_graph_data(
        train_data_dir, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    print(f"Loading test data from {test_data_dir}")
    test_dataset, test_loader = load_graph_data(
        test_data_dir, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Get the number of classes and node features
    num_classes = train_dataset.num_classes
    num_node_features = train_dataset.num_node_features
    print(f"Number of classes: {num_classes}")
    print(f"Number of node features: {num_node_features}")
    
    # Initialize model
    if args.module in ['GCNConv', 'GATConv', 'GraphConv', 'SAGEConv']:
        model = GNN_Classifier(
            input_dim=num_node_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            module=args.module,
            layer_norm=args.layer_norm,
            residual=args.residual,
            pooling=args.pooling
        )
    elif args.module in ['RGCNConv', 'RGATConv']:
        model = RGNN_Classifier(
            input_dim=num_node_features,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            module=args.module,
            layer_norm=args.layer_norm,
            residual=args.residual,
            num_relations=args.num_relations,
            pooling=args.pooling
        )
    else:
        raise ValueError(f"Unsupported module: {args.module}")
    
    # Move model to device
    model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    
    # Initialize optimizer and scheduler
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = create_scheduler(
        optimizer, 
        args.lr_scheduler, 
        args.num_epochs * len(train_loader), 
        args.warmup_steps
    )
    
    # Initialize loss function
    criterion = CrossEntropyLoss()
    
    # Initialize metrics tracking
    train_losses = []
    test_losses = []
    train_f1_scores = []
    test_f1_scores = []
    best_test_f1 = 0.0
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_true, train_pred = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        train_losses.append(train_loss)
        
        # Calculate metrics
        train_acc = accuracy_score(train_true, train_pred)
        train_f1 = f1_score(train_true, train_pred, average='weighted')
        train_f1_scores.append(train_f1)
        
        # Generate and save classification report
        train_report = classification_report(train_true, train_pred, output_dict=True)
        with open(os.path.join(output_dir, f"classification_report_train_epoch{epoch}.json"), "w") as f:
            json.dump(train_report, f, indent=2)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        
        # Evaluate
        test_loss, test_true, test_pred = evaluate(
            model, test_loader, device, criterion
        )
        test_losses.append(test_loss)
        
        # Calculate metrics
        test_acc = accuracy_score(test_true, test_pred)
        test_f1 = f1_score(test_true, test_pred, average='weighted')
        test_f1_scores.append(test_f1)
        
        # Generate and save classification report
        test_report = classification_report(test_true, test_pred, output_dict=True)
        with open(os.path.join(output_dir, f"classification_report_test_epoch{epoch}.json"), "w") as f:
            json.dump(test_report, f, indent=2)
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        
        # Save model if it's the best so far
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved new best model with F1 score: {test_f1:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_f1': train_f1,
            'test_f1': test_f1,
        }, os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    
    # Save metrics
    save_metrics(
        output_dir,
        train_losses=train_losses,
        test_losses=test_losses,
        train_f1_scores=train_f1_scores,
        test_f1_scores=test_f1_scores
    )
    
    print(f"Training complete. Results saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    main()
