"""
Validation Module

Validation loop returning loss, predictions, and metrics.
"""

from typing import Tuple, List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def validate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray, List[str]]:
    """
    Validate model on a dataset.
    
    Args:
        model: Model to validate
        data_loader: Data loader
        criterion: Loss function
        device: Device
        
    Returns:
        Tuple of (loss, accuracy, predictions, labels, paths)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in data_loader:
            images = images.to(device)
            labels_tensor = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels_tensor).sum().item()
            total += labels.size(0)
            
            # Store predictions
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return (
        avg_loss,
        accuracy,
        np.array(all_predictions),
        np.array(all_labels),
        all_paths
    )


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Evaluate model on test set (no loss calculation).
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device
        
    Returns:
        Tuple of (predictions, labels, paths)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        all_paths
    )
