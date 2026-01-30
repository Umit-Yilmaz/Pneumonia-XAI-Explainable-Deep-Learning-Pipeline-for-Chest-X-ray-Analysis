"""
ResNet50 Model for Pneumonia Classification

Pretrained on ImageNet with partial fine-tuning (freeze first 3 blocks, fine-tune last 2)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional


class ResNet50Classifier(nn.Module):
    """
    ResNet50 with custom classifier head for binary classification.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_blocks: List of block indices to freeze (0-4)
        finetune_blocks: List of block indices to fine-tune
        fc_units: List of FC layer units
        dropout: Dropout rate
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_blocks: List[int] = [0, 1, 2],
        finetune_blocks: List[int] = [3, 4],
        fc_units: List[int] = [512],
        dropout: float = 0.5,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get layer references for freezing
        self.layers = {
            0: [self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool],
            1: self.backbone.layer1,
            2: self.backbone.layer2,
            3: self.backbone.layer3,
            4: self.backbone.layer4
        }
        
        # Freeze specified blocks
        for block_idx in freeze_blocks:
            if block_idx == 0:
                for layer in self.layers[0]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for param in self.layers[block_idx].parameters():
                    param.requires_grad = False
        
        # Remove original FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
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
        
        # For Grad-CAM++
        self.gradients = None
        self.activations = None
        
        # Register hook on layer4
        self.backbone.layer4.register_forward_hook(self._forward_hook)
    
    def _forward_hook(self, module, input, output):
        """Store activations for Grad-CAM++."""
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._backward_hook)
    
    def _backward_hook(self, grad):
        """Store gradients for Grad-CAM++."""
        self.gradients = grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through backbone
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_activations(self) -> Optional[torch.Tensor]:
        return self.activations
    
    def get_gradients(self) -> Optional[torch.Tensor]:
        return self.gradients
    
    def get_target_layer(self):
        """Return target layer for Grad-CAM++."""
        return self.backbone.layer4


def create_resnet(config: dict) -> ResNet50Classifier:
    """Create ResNet50 model from config."""
    model_config = config.get('model', {})
    
    pretrained = model_config.get('pretrained', 'imagenet') == 'imagenet'
    
    return ResNet50Classifier(
        pretrained=pretrained,
        freeze_blocks=model_config.get('freeze_blocks', [0, 1, 2]),
        finetune_blocks=model_config.get('finetune_blocks', [3, 4]),
        fc_units=model_config.get('fc_units', [512]),
        dropout=model_config.get('dropout', 0.5),
        num_classes=model_config.get('num_classes', 1)
    )
