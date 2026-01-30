"""
ROC Curve Module

Generate and save ROC curve visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    model_name: str = "Model",
    figsize: tuple = (8, 6)
) -> float:
    """
    Generate and optionally save ROC curve plot.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        save_path: Path to save the figure
        model_name: Name for the plot title
        figsize: Figure size
        
    Returns:
        AUC score
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        marker='o', color='red', s=100,
        label=f'Optimal (threshold={optimal_threshold:.2f})'
    )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.close()
    
    return roc_auc


def plot_roc_curves_comparison(
    results: Dict[str, tuple],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> Dict[str, float]:
    """
    Plot ROC curves for multiple models on same plot.
    
    Args:
        results: Dict of {model_name: (predictions, labels)}
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Dict of {model_name: auc_score}
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['darkorange', 'green', 'blue', 'red', 'purple']
    auc_scores = {}
    
    for idx, (model_name, (predictions, labels)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        
        color = colors[idx % len(colors)]
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f'{model_name} (AUC = {roc_auc:.4f})'
        )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC comparison saved to: {save_path}")
    
    plt.close()
    
    return auc_scores
