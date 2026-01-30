"""
DenseNet121 Model for Pneumonia Classification

Pretrained on ImageNet with partial fine-tuning
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121 with custom classifier head for binary classification.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        finetune_from: Start fine-tuning from this dense block
        fc_units: List of FC layer units
        dropout: Dropout rate
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        finetune_from: str = 'denseblock3',
        fc_units: List[int] = [512],
        dropout: float = 0.5,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained DenseNet121
        if pretrained:
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.densenet121(weights=None)
        
        # Freeze layers before finetune_from
        finetune_started = False
        for name, module in self.backbone.features.named_children():
            if name == finetune_from:
                finetune_started = True
            
            if not finetune_started:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Get feature dimensions
        in_features = self.backbone.classifier.in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Build custom classifier
        classifier_layers = []
        
        for units in fc_units:
            classifier_layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = units
        
        classifier_layers.append(nn.Linear(in_features, num_classes))
        
        if num_classes == 1:
            classifier_layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # For Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Register hook on denseblock4
        self.backbone.features.denseblock4.register_forward_hook(self._forward_hook)
    
    def _forward_hook(self, module, input, output):
        """Store activations for Grad-CAM."""
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._backward_hook)
    
    def _backward_hook(self, grad):
        """Store gradients for Grad-CAM."""
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through backbone features
        features = self.backbone.features(x)
        
        # Global average pooling
        features = nn.functional.relu(features, inplace=True)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_activations(self) -> Optional[torch.Tensor]:
        return self.activations
    
    def get_gradients(self) -> Optional[torch.Tensor]:
        return self.gradients
    
    def get_target_layer(self):
        """Return target layer for Grad-CAM."""
        return self.backbone.features.denseblock4


def create_densenet(config: dict) -> DenseNet121Classifier:
    """Create DenseNet121 model from config."""
    model_config = config.get('model', {})
    
    pretrained = model_config.get('pretrained', 'imagenet') == 'imagenet'
    
    return DenseNet121Classifier(
        pretrained=pretrained,
        finetune_from=model_config.get('finetune_from', 'denseblock3'),
        fc_units=model_config.get('fc_units', [512]),
        dropout=model_config.get('dropout', 0.5),
        num_classes=model_config.get('num_classes', 1)
    )
