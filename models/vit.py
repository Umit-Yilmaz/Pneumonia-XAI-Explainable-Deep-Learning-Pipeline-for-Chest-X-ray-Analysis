"""
Vision Transformer (ViT) Model for Pneumonia Classification

ViT-B/16 pretrained on ImageNet-21k with custom classifier
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ViTClassifier(nn.Module):
    """
    Vision Transformer with custom classifier head for binary classification.
    
    Args:
        model_variant: ViT model variant (e.g., 'vit_base_patch16_224')
        pretrained: Use pretrained weights
        fc_units: List of FC layer units
        dropout: Dropout rate
        num_classes: Number of output classes
    """
    
    def __init__(
        self,
        model_variant: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        fc_units: List[int] = [256],
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for ViT. Install with: pip install timm")
        
        self.num_classes = num_classes
        
        # Load pretrained ViT
        self.backbone = timm.create_model(
            model_variant,
            pretrained=pretrained,
            num_classes=0  # Remove classifier
        )
        
        # Get feature dimensions
        in_features = self.backbone.num_features
        
        # Build custom classifier
        classifier_layers = []
        
        for units in fc_units:
            classifier_layers.extend([
                nn.Linear(in_features, units),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_features = units
        
        classifier_layers.append(nn.Linear(in_features, num_classes))
        
        if num_classes == 1:
            classifier_layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # For Attention Rollout
        self.attention_weights = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clear previous attention weights
        self.attention_weights = []
        
        # Forward through backbone
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from all transformer blocks.
        
        Returns:
            List of attention tensors [batch, heads, tokens, tokens]
        """
        attention_maps = []
        
        # Hook to capture attention
        hooks = []
        
        def attention_hook(module, input, output):
            # For timm ViT, attention is computed in the Attention module
            if hasattr(module, 'attn_drop'):
                # Get attention weights before dropout
                attention_maps.append(output)
        
        # Register hooks on all attention modules
        for block in self.backbone.blocks:
            hook = block.attn.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.backbone.forward_features(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def get_attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention rollout across all transformer layers.
        
        Returns:
            Attention map [batch, height, width]
        """
        batch_size = x.size(0)
        
        # Get all attention maps
        attention_maps = []
        hooks = []
        
        def get_attention(module, input, output):
            # Extract Q, K for attention computation
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            attention_maps.append(attn.detach())
        
        for block in self.backbone.blocks:
            hook = block.attn.register_forward_hook(get_attention)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.backbone.forward_features(x)
        
        for hook in hooks:
            hook.remove()
        
        if not attention_maps:
            return None
        
        # Compute rollout
        # Average attention heads
        result = torch.eye(attention_maps[0].size(-1)).unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        
        for attention in attention_maps:
            # Average across heads
            attention_heads_fused = attention.mean(dim=1)
            
            # Add identity and normalize
            attention_heads_fused = attention_heads_fused + torch.eye(attention_heads_fused.size(-1)).to(x.device)
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Multiply
            result = torch.matmul(attention_heads_fused, result)
        
        # Get attention from CLS token to patches
        mask = result[:, 0, 1:]  # Exclude CLS token
        
        # Reshape to spatial dimensions
        num_patches = mask.size(1)
        patch_size = int(num_patches ** 0.5)
        mask = mask.reshape(batch_size, patch_size, patch_size)
        
        return mask


def create_vit(config: dict) -> ViTClassifier:
    """Create ViT model from config."""
    model_config = config.get('model', {})
    
    pretrained = model_config.get('pretrained', 'imagenet21k') in ['imagenet21k', 'imagenet', True]
    
    return ViTClassifier(
        model_variant=model_config.get('model_variant', 'vit_base_patch16_224'),
        pretrained=pretrained,
        fc_units=model_config.get('fc_units', [256]),
        dropout=model_config.get('dropout', 0.1),
        num_classes=model_config.get('num_classes', 1)
    )
