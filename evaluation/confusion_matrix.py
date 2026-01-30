"""
Confusion Matrix Module

Generate and save confusion matrix visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list = ['NORMAL', 'PNEUMONIA'],
    save_path: Optional[str] = None,
    model_name: str = "Model",
    threshold: float = 0.5,
    normalize: bool = False,
    figsize: tuple = (8, 6)
) -> np.ndarray:
    """
    Generate and optionally save confusion matrix plot.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        class_names: Names for each class
        save_path: Path to save the figure
        model_name: Name for the plot title
        threshold: Classification threshold
        normalize: Whether to normalize the matrix
        figsize: Figure size
        
    Returns:
        Confusion matrix array
    """
    # Convert probabilities to binary predictions
    pred_labels = (predictions >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = sk_confusion_matrix(labels, pred_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()
    
    return cm


def plot_confusion_matrix_comparison(
    results: dict,
    class_names: list = ['NORMAL', 'PNEUMONIA'],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 4)
):
    """
    Plot confusion matrices for multiple models side by side.
    
    Args:
        results: Dict of {model_name: (predictions, labels)}
        class_names: Class names
        save_path: Path to save figure
        figsize: Figure size
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, (predictions, labels)) in zip(axes, results.items()):
        pred_labels = (predictions >= 0.5).astype(int)
        cm = sk_confusion_matrix(labels, pred_labels)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={'size': 12}
        )
        
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(model_name, fontsize=12)
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.close()
