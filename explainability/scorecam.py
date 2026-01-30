"""
Score-CAM Implementation

Score-weighted Class Activation Mapping (gradient-free method).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import cv2
from tqdm import tqdm


class ScoreCAM:
    """
    Score-CAM implementation for gradient-free attention visualization.
    
    Score-CAM uses activation maps directly without gradients.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Score-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        if target_layer is not None:
            target_layer.register_forward_hook(self._forward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 16
    ) -> np.ndarray:
        """
        Generate Score-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class for CAM
            batch_size: Batch size for processing activation maps
            
        Returns:
            CAM heatmap [H, W]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass to get activations
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = 1 if output[0, 0] >= 0.5 else 0
            
            # Get activations
            if self.target_layer is not None:
                activations = self.activations
            else:
                activations = self.model.get_activations()
            
            if activations is None:
                raise ValueError("Could not get activations.")
            
            # Get dimensions
            b, c, h, w = activations.shape
            input_h, input_w = input_tensor.shape[2:]
            
            # Upsample activations to input size
            upsampled = F.interpolate(
                activations,
                size=(input_h, input_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Normalize each activation map
            upsampled = upsampled.squeeze(0)  # [C, H, W]
            
            # Min-max normalize
            for i in range(c):
                act_map = upsampled[i]
                act_min = act_map.min()
                act_max = act_map.max()
                if act_max - act_min > 0:
                    upsampled[i] = (act_map - act_min) / (act_max - act_min)
            
            # Compute scores
            scores = []
            
            for i in range(0, c, batch_size):
                batch_maps = upsampled[i:i + batch_size]
                
                # Mask input with activation maps
                masked_inputs = input_tensor * batch_maps.unsqueeze(1)
                
                # Get predictions
                batch_outputs = self.model(masked_inputs)
                
                if batch_outputs.shape[1] == 1:
                    batch_scores = batch_outputs if target_class == 1 else (1 - batch_outputs)
                else:
                    batch_scores = batch_outputs[:, target_class:target_class+1]
                
                scores.append(batch_scores.squeeze())
            
            scores = torch.cat(scores)
            
            # Softmax scores
            weights = F.softmax(scores, dim=0)
            
            # Weighted combination
            cam = torch.zeros(h, w).to(input_tensor.device)
            for i, weight in enumerate(weights):
                cam += weight * activations[0, i]
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam
    
    def generate_heatmap(
        self,
        cam: np.ndarray,
        original_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Convert CAM to colored heatmap."""
        cam_resized = cv2.resize(cam, original_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
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


def visualize_scorecam(
    model,
    image_path: str,
    save_path: Optional[str] = None,
    device: torch.device = torch.device('cpu'),
    normalization: str = 'imagenet',
    figsize: tuple = (12, 4)
) -> np.ndarray:
    """
    Generate and visualize Score-CAM for a single image.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image.resize((224, 224)), dtype=np.float32)
    
    # Normalize
    if normalization == 'divide_255':
        input_array = image_array / 255.0
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_array = (image_array / 255.0 - mean) / std
    
    input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    target_layer = model.get_target_layer() if hasattr(model, 'get_target_layer') else None
    
    scorecam = ScoreCAM(model, target_layer)
    cam = scorecam.generate_cam(input_tensor)
    
    heatmap = scorecam.generate_heatmap(cam)
    overlay = scorecam.overlay_heatmap(image_array / 255.0, heatmap / 255.0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image_array.astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Score-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    
    return cam
