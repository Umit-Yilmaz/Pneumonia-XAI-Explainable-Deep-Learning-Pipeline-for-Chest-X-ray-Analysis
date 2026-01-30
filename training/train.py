"""
Training Module

Training loop with EarlyStopping and ReduceLROnPlateau callbacks.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, model_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{model_name}_training.log"
        self.history_file = self.log_dir / f"{model_name}_history.json"
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epoch_time': []
        }
        
        # Setup file logger
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def log(self, message: str):
        self.logger.info(message)
        print(message)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float
    ):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        self.history['epoch_time'].append(epoch_time)
        
        message = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {lr:.6f} | Time: {epoch_time:.1f}s"
        )
        self.log(message)
    
    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    save_dir: str = 'results/models'
) -> Tuple[nn.Module, Dict]:
    """
    Full training loop with callbacks.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to train on
        save_dir: Directory to save model checkpoints
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    from .validate import validate_epoch
    
    training_config = config.get('training', {})
    results_config = config.get('results', {})
    model_name = config.get('model', {}).get('name', 'model')
    
    # Setup
    model = model.to(device)
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0)
    )
    
    # Loss
    criterion = nn.BCELoss()
    
    # Scheduler
    reduce_lr_config = training_config.get('reduce_lr', {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=reduce_lr_config.get('factor', 0.1),
        patience=reduce_lr_config.get('patience', 3),
        min_lr=reduce_lr_config.get('min_lr', 1e-6)
    )
    
    # Early stopping
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping = EarlyStopping(
        patience=early_stopping_config.get('patience', 5),
        restore_best_weights=early_stopping_config.get('restore_best_weights', True)
    )
    
    # Logger
    log_dir = results_config.get('logs_dir', 'results/logs')
    logger = TrainingLogger(log_dir, model_name)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    epochs = training_config.get('epochs', 50)
    best_val_loss = float('inf')
    
    logger.log(f"Starting training for {model_name}")
    logger.log(f"Device: {device}")
    logger.log(f"Epochs: {epochs}")
    logger.log(f"Batch size: {training_config.get('batch_size', 32)}")
    logger.log(f"Learning rate: {training_config.get('learning_rate', 0.001)}")
    logger.log("-" * 80)
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _, _ = validate_epoch(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        logger.log_epoch(
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path / f"{model_name}_best.pth")
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.log(f"Early stopping triggered at epoch {epoch}")
            break
    
    logger.log("-" * 80)
    logger.log(f"Training completed. Best val loss: {best_val_loss:.4f}")
    logger.save_history()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, save_path / f"{model_name}_final.pth")
    
    return model, logger.history
