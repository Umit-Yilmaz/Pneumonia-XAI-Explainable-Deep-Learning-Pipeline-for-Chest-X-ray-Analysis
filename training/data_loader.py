"""
Data Loader Module

PyTorch Dataset and DataLoader utilities for pneumonia X-ray images.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class PneumoniaDataset(Dataset):
    """
    PyTorch Dataset for pneumonia X-ray images.
    
    Args:
        data_dir: Path to data directory (e.g., data/cnn-normalized/train)
        normalization: 'divide_255' or 'imagenet' or None
        transform: Optional additional transforms
    """
    
    CLASS_MAP = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    def __init__(
        self,
        data_dir: str,
        normalization: str = 'divide_255',
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.normalization = normalization
        self.transform = transform
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
        
        # ImageNet normalization parameters
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
    
    def _load_samples(self):
        """Load all image paths and labels."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for class_name, label in self.CLASS_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        
        # Normalize
        if self.normalization == 'divide_255':
            image = image / 255.0
        elif self.normalization == 'imagenet':
            image = image / 255.0
            image = (image - self.imagenet_mean) / self.imagenet_std
        
        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Apply additional transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)


def create_dataloaders(
    config: dict,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from config.
    
    Args:
        config: Configuration dictionary
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    batch_size = training_config.get('batch_size', 32)
    normalization = data_config.get('normalization', 'divide_255')
    
    # Create datasets
    train_dataset = PneumoniaDataset(
        data_dir=data_config.get('train_path', 'data/cnn-normalized/train'),
        normalization=normalization
    )
    
    val_dataset = PneumoniaDataset(
        data_dir=data_config.get('val_path', 'data/cnn-normalized/val'),
        normalization=normalization
    )
    
    test_dataset = PneumoniaDataset(
        data_dir=data_config.get('test_path', 'data/cnn-normalized/test'),
        normalization=normalization
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_sample_images(
    data_loader: DataLoader,
    predictions: np.ndarray,
    labels: np.ndarray,
    paths: List[str],
    category: str,
    num_samples: int = 5,
    confidence_threshold: Dict[str, float] = {'high': 0.9, 'low': 0.6}
) -> List[Dict]:
    """
    Select samples based on prediction category.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        paths: Image paths
        category: One of 'true_positive', 'true_negative', 'false_positive', 'false_negative'
        num_samples: Number of samples to select
        confidence_threshold: Threshold for high/low confidence
        
    Returns:
        List of dictionaries with sample info
    """
    pred_labels = (predictions >= 0.5).astype(int)
    
    # Filter by category
    if category == 'true_positive':
        mask = (pred_labels == 1) & (labels == 1)
    elif category == 'true_negative':
        mask = (pred_labels == 0) & (labels == 0)
    elif category == 'false_positive':
        mask = (pred_labels == 1) & (labels == 0)
    elif category == 'false_negative':
        mask = (pred_labels == 0) & (labels == 1)
    else:
        raise ValueError(f"Unknown category: {category}")
    
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return []
    
    # Sort by confidence (descending for TP/FP, ascending for TN/FN)
    if category in ['true_positive', 'false_positive']:
        # Higher confidence first
        sorted_indices = indices[np.argsort(-predictions[indices])]
    else:
        # Lower confidence first (more confident of being negative)
        sorted_indices = indices[np.argsort(predictions[indices])]
    
    # Select top samples
    selected_indices = sorted_indices[:num_samples]
    
    samples = []
    for idx in selected_indices:
        samples.append({
            'path': paths[idx],
            'label': int(labels[idx]),
            'prediction': float(predictions[idx]),
            'pred_label': int(pred_labels[idx]),
            'category': category,
            'confidence': 'high' if abs(predictions[idx] - 0.5) > (confidence_threshold['high'] - 0.5) else 'low'
        })
    
    return samples
