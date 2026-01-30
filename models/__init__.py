"""
Model Factory

Provides unified interface for creating all model architectures.
"""

from typing import Union
import yaml

from .cnn import CustomCNN, create_cnn
from .resnet import ResNet50Classifier, create_resnet
from .densenet import DenseNet121Classifier, create_densenet
from .vit import ViTClassifier, create_vit


MODEL_REGISTRY = {
    'CustomCNN': create_cnn,
    'CNN': create_cnn,
    'ResNet50': create_resnet,
    'ResNet': create_resnet,
    'DenseNet121': create_densenet,
    'DenseNet': create_densenet,
    'ViT': create_vit,
    'VisionTransformer': create_vit,
}


def create_model(config: dict) -> Union[CustomCNN, ResNet50Classifier, DenseNet121Classifier, ViTClassifier]:
    """
    Create a model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_name = config.get('model', {}).get('name', 'CustomCNN')
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](config)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_from_config_file(config_path: str) -> Union[CustomCNN, ResNet50Classifier, DenseNet121Classifier, ViTClassifier]:
    """
    Create a model from a config file path.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Instantiated model
    """
    config = load_config(config_path)
    return create_model(config)


__all__ = [
    'CustomCNN',
    'ResNet50Classifier', 
    'DenseNet121Classifier',
    'ViTClassifier',
    'create_model',
    'load_config',
    'get_model_from_config_file',
    'create_cnn',
    'create_resnet',
    'create_densenet',
    'create_vit',
]
