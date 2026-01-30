"""
Evaluation Metrics Module

Calculate accuracy, precision, recall, F1, and ROC-AUC.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate all classification metrics.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    pred_labels = (predictions >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, pred_labels),
        'precision': precision_score(labels, pred_labels, zero_division=0),
        'recall': recall_score(labels, pred_labels, zero_division=0),
        'f1': f1_score(labels, pred_labels, zero_division=0),
        'roc_auc': roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    }
    
    # Calculate additional metrics
    tn = np.sum((pred_labels == 0) & (labels == 0))
    fp = np.sum((pred_labels == 1) & (labels == 0))
    fn = np.sum((pred_labels == 0) & (labels == 1))
    tp = np.sum((pred_labels == 1) & (labels == 1))
    
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Sensitivity (same as recall)
    metrics['sensitivity'] = metrics['recall']
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Print metrics in a formatted table."""
    print(f"\n{'='*50}")
    print(f"Evaluation Metrics - {model_name}")
    print('='*50)
    
    print(f"{'Metric':<20} {'Value':>10}")
    print('-'*30)
    
    display_order = [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        'sensitivity', 'specificity'
    ]
    
    for metric_name in display_order:
        if metric_name in metrics:
            print(f"{metric_name.capitalize():<20} {metrics[metric_name]:>10.4f}")
    
    print('-'*30)
    print(f"{'True Positives':<20} {metrics.get('true_positives', 0):>10d}")
    print(f"{'True Negatives':<20} {metrics.get('true_negatives', 0):>10d}")
    print(f"{'False Positives':<20} {metrics.get('false_positives', 0):>10d}")
    print(f"{'False Negatives':<20} {metrics.get('false_negatives', 0):>10d}")
    print('='*50)


def save_metrics_to_file(
    metrics: Dict[str, float],
    filepath: str,
    model_name: str = "Model"
):
    """Save metrics to a text file."""
    with open(filepath, 'w') as f:
        f.write(f"Evaluation Metrics - {model_name}\n")
        f.write("="*50 + "\n\n")
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{metric_name}: {value:.4f}\n")
            else:
                f.write(f"{metric_name}: {value}\n")
