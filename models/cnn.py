"""
Custom CNN Model for Pneumonia Classification

5-block CNN with progressive filters (32→256), BatchNorm, MaxPool, Dropout
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, ReLU, MaxPool, Dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 pool_size: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Dropout2d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom 5-block CNN for binary classification.
    
    Args:
        input_size: Input image size (default: 224)
        channels: Number of input channels (default: 3)
        filters: List of filter counts for each conv block
        kernel_size: Convolution kernel size
        pool_size: Max pooling size
        dropout: List of dropout rates for each block
        fc_units: List of fully connected layer units
        num_classes: Number of output classes (1 for binary)
    """
    
    def __init__(
        self,
        input_size: int = 224,
        channels: int = 3,
        filters: List[int] = [32, 64, 128, 128, 256],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: List[float] = [0.3, 0.3, 0.4, 0.4, 0.5],
        fc_units: List[int] = [512, 256],
        num_classes: int = 1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_ch = channels
        
        for i, (out_ch, drop) in enumerate(zip(filters, dropout)):
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, kernel_size, pool_size, drop)
            )
            in_ch = out_ch
        
        # Calculate feature map size after conv blocks
        feature_size = input_size // (pool_size ** len(filters))
        flatten_size = filters[-1] * feature_size * feature_size
        
        # Build classifier
        classifier_layers = []
        in_features = flatten_size
        
        for units in fc_units:
            classifier_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            ])
            in_features = units
        
        classifier_layers.append(nn.Linear(in_features, num_classes))
        
        if num_classes == 1:
            classifier_layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Store last conv layer for Grad-CAM
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        """Hook for gradients."""
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through conv blocks
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            
            # Store activations from last conv block for Grad-CAM
            if i == len(self.conv_blocks) - 1:
                self.activations = x
                if x.requires_grad:
                    x.register_hook(self.activations_hook)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_activations(self) -> Optional[torch.Tensor]:
        """Get activations from last conv block."""
        return self.activations
    
    def get_gradients(self) -> Optional[torch.Tensor]:
        """Get gradients from last conv block."""
        return self.gradients


def create_cnn(config: dict) -> CustomCNN:
    """Create CNN model from config."""
    model_config = config.get('model', {})
    
    return CustomCNN(
        input_size=model_config.get('input_size', 224),
        channels=model_config.get('channels', 3),
        filters=model_config.get('filters', [32, 64, 128, 128, 256]),
        kernel_size=model_config.get('kernel_size', 3),
        pool_size=model_config.get('pool_size', 2),
        dropout=model_config.get('dropout', [0.3, 0.3, 0.4, 0.4, 0.5]),
        fc_units=model_config.get('fc_units', [512, 256]),
        num_classes=model_config.get('num_classes', 1)
    )
