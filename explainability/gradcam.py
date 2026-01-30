"""
Grad-CAM Implementation

Gradient-weighted Class Activation Mapping for CNN and DenseNet.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import cv2


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    
    Works with models that have:
    - get_activations() method
    - get_gradients() method
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Target layer for CAM (optional, model should handle it)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks if target layer is provided
        if target_layer is not None:
            target_layer.register_forward_hook(self._forward_hook)
            target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class for CAM (None = predicted class)
            
        Returns:
            CAM heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = 1 if output[0, 0] >= 0.5 else 0
        
        # Backward pass
        self.model.zero_grad()
        
        # For binary classification with sigmoid
        if output.shape[1] == 1:
            target = output[0, 0] if target_class == 1 else (1 - output[0, 0])
        else:
            target = output[0, target_class]
        
        target.backward()
        
        # Get gradients and activations
        if self.target_layer is not None:
            gradients = self.gradients
            activations = self.activations
        else:
            gradients = self.model.get_gradients()
            activations = self.model.get_activations()
        
        if gradients is None or activations is None:
            raise ValueError("Could not get gradients or activations. Check model implementation.")
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap(
        self,
        cam: np.ndarray,
        original_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Convert CAM to colored heatmap.
        
        Args:
            cam: CAM array
            original_size: Size to resize heatmap to
            
        Returns:
            Colored heatmap [H, W, 3]
        """
        # Resize to original size
        cam_resized = cv2.resize(cam, original_size)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image [H, W, 3]
            heatmap: Colored heatmap [H, W, 3]
            alpha: Blending factor
            
        Returns:
            Overlayed image [H, W, 3]
        """
        # Ensure same size
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize image to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Blend
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlay


def visualize_gradcam(
    model,
    image_path: str,
    save_path: Optional[str] = None,
    device: torch.device = torch.device('cpu'),
    normalization: str = 'divide_255',
    figsize: tuple = (12, 4)
) -> np.ndarray:
    """
    Generate and visualize Grad-CAM for a single image.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        save_path: Path to save visualization
        device: Device to use
        normalization: Image normalization type
        figsize: Figure size
        
    Returns:
        CAM heatmap
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
    
    # Generate CAM
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(input_tensor)
    
    # Generate heatmap and overlay
    heatmap = gradcam.generate_heatmap(cam)
    overlay = gradcam.overlay_heatmap(image_array / 255.0, heatmap / 255.0)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image_array.astype(np.uint8))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
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
