"""
Explainability Package

Provides Grad-CAM, Grad-CAM++, Score-CAM, and Attention Rollout methods.
"""

from .gradcam import GradCAM, visualize_gradcam
from .gradcam_pp import GradCAMPlusPlus, visualize_gradcam_pp
from .scorecam import ScoreCAM, visualize_scorecam
from .attention_rollout import AttentionRollout, visualize_attention_rollout
from .sample_selection import SampleSelector, select_explainability_samples, SelectedSample


def get_explainer(method: str, model, target_layer=None):
    """
    Factory function to get appropriate explainer.
    
    Args:
        method: Explainability method name
        model: PyTorch model
        target_layer: Target layer for CAM methods
        
    Returns:
        Explainer instance
    """
    method = method.lower().replace('-', '_').replace(' ', '_')
    
    if method in ['gradcam', 'grad_cam']:
        return GradCAM(model, target_layer)
    elif method in ['gradcam_pp', 'gradcampp', 'grad_cam_pp', 'gradcam++']:
        return GradCAMPlusPlus(model, target_layer)
    elif method in ['scorecam', 'score_cam']:
        return ScoreCAM(model, target_layer)
    elif method in ['attention_rollout', 'attentionrollout']:
        return AttentionRollout(model)
    else:
        raise ValueError(f"Unknown explainability method: {method}")


def visualize(
    method: str,
    model,
    image_path: str,
    save_path=None,
    device=None,
    normalization: str = 'imagenet'
):
    """
    Unified visualization function.
    
    Args:
        method: Explainability method
        model: PyTorch model
        image_path: Path to input image
        save_path: Path to save visualization
        device: Device to use
        normalization: Normalization type
        
    Returns:
        Attention/CAM map
    """
    import torch
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    method = method.lower().replace('-', '_').replace(' ', '_')
    
    if method in ['gradcam', 'grad_cam']:
        return visualize_gradcam(model, image_path, save_path, device, normalization)
    elif method in ['gradcam_pp', 'gradcampp', 'grad_cam_pp', 'gradcam++']:
        return visualize_gradcam_pp(model, image_path, save_path, device, normalization)
    elif method in ['scorecam', 'score_cam']:
        return visualize_scorecam(model, image_path, save_path, device, normalization)
    elif method in ['attention_rollout', 'attentionrollout']:
        return visualize_attention_rollout(model, image_path, save_path, device, normalization)
    else:
        raise ValueError(f"Unknown explainability method: {method}")


__all__ = [
    'GradCAM',
    'GradCAMPlusPlus',
    'ScoreCAM',
    'AttentionRollout',
    'visualize_gradcam',
    'visualize_gradcam_pp',
    'visualize_scorecam',
    'visualize_attention_rollout',
    'get_explainer',
    'visualize'
]
