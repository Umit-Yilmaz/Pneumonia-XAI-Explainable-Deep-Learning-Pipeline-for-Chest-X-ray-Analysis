"""
Attention Rollout Implementation

Attention visualization for Vision Transformers.
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import cv2


class AttentionRollout:
    """
    Attention Rollout for Vision Transformer visualization.
    
    Rolls attention across all transformer layers to visualize
    which patches the model focuses on.
    """
    
    def __init__(self, model, head_fusion: str = 'mean', discard_ratio: float = 0.1):
        """
        Initialize Attention Rollout.
        
        Args:
            model: ViT model (from timm or custom)
            head_fusion: How to fuse attention heads ('mean', 'max', 'min')
            discard_ratio: Ratio of lowest attention to discard
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def get_attention_maps(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from all transformer blocks.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            
        Returns:
            List of attention tensors [num_heads, tokens, tokens]
        """
        self.model.eval()
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
        
        # Register hooks on all attention modules
        for block in self.model.backbone.blocks:
            hook = block.attn.register_forward_hook(get_attention)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model.backbone.forward_features(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def compute_rollout(
        self,
        input_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Compute attention rollout.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            Attention mask [H, W] where H, W are patch grid dimensions
        """
        attention_maps = self.get_attention_maps(input_tensor)
        
        if not attention_maps:
            # Use model's built-in method if available
            if hasattr(self.model, 'get_attention_rollout'):
                mask = self.model.get_attention_rollout(input_tensor)
                return mask.squeeze().cpu().numpy()
            raise ValueError("Could not extract attention maps.")
        
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        num_tokens = attention_maps[0].size(-1)
        
        # Initialize with identity matrix
        result = torch.eye(num_tokens).unsqueeze(0).expand(batch_size, -1, -1).to(device)
        
        for attention in attention_maps:
            # Fuse attention heads
            if self.head_fusion == 'mean':
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == 'max':
                attention_heads_fused = attention.max(dim=1)[0]
            elif self.head_fusion == 'min':
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head fusion: {self.head_fusion}")
            
            # Discard lowest attentions
            flat = attention_heads_fused.view(batch_size, -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=1, keepdim=True)
            threshold = threshold.unsqueeze(-1)
            attention_heads_fused = torch.where(
                attention_heads_fused > threshold,
                attention_heads_fused,
                torch.zeros_like(attention_heads_fused)
            )
            
            # Add identity and normalize
            I = torch.eye(num_tokens).to(device)
            attention_heads_fused = attention_heads_fused + I
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
            # Matrix multiplication
            result = torch.matmul(attention_heads_fused, result)
        
        # Get attention from CLS token to patches (exclude CLS token itself)
        mask = result[0, 0, 1:]
        
        # Reshape to spatial dimensions
        num_patches = mask.size(0)
        patch_size = int(num_patches ** 0.5)
        mask = mask.reshape(patch_size, patch_size).cpu().numpy()
        
        return mask
    
    def generate_heatmap(
        self,
        mask: np.ndarray,
        original_size: tuple = (224, 224)
    ) -> np.ndarray:
        """Convert attention mask to colored heatmap."""
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # Resize to original size
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        return overlay


def visualize_attention_rollout(
    model,
    image_path: str,
    save_path: Optional[str] = None,
    device: torch.device = torch.device('cpu'),
    normalization: str = 'imagenet',
    figsize: tuple = (12, 4)
) -> np.ndarray:
    """
    Generate and visualize Attention Rollout for a single image.
    
    Args:
        model: ViT model
        image_path: Path to input image
        save_path: Path to save visualization
        device: Device to use
        normalization: Image normalization type
        figsize: Figure size
        
    Returns:
        Attention mask
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image.resize((224, 224)), dtype=np.float32)
    
    # Normalize
    if normalization == 'divide_255':
        input_array = image_array / 255.0
    else:  # imagenet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_array = (image_array / 255.0 - mean) / std
    
    # Convert to tensor
    input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Generate attention rollout
    rollout = AttentionRollout(model)
    mask = rollout.compute_rollout(input_tensor)
    
    # Generate heatmap and overlay
    heatmap = rollout.generate_heatmap(mask)
    overlay = rollout.overlay_heatmap(image_array / 255.0, heatmap / 255.0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image_array.astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Attention Rollout')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    return mask
