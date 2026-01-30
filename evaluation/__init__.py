"""
Evaluation Package

Provides metrics calculation, confusion matrix, and ROC curve utilities.
"""

from .metrics import calculate_metrics, print_metrics, save_metrics_to_file
from .confusion_matrix import plot_confusion_matrix, plot_confusion_matrix_comparison
from .roc_curve import plot_roc_curve, plot_roc_curves_comparison

__all__ = [
    'calculate_metrics',
    'print_metrics',
    'save_metrics_to_file',
    'plot_confusion_matrix',
    'plot_confusion_matrix_comparison',
    'plot_roc_curve',
    'plot_roc_curves_comparison'
]
