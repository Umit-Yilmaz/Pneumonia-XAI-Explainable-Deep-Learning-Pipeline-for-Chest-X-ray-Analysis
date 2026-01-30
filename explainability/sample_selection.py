"""
Explainability Sample Selection Module

Selects samples from test set for explainability analysis based on:
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SelectedSample:
    """Represents a selected sample for explainability."""
    path: str
    label: int
    prediction: float
    pred_label: int
    category: str
    confidence: str
    
    def to_dict(self):
        return asdict(self)


class SampleSelector:
    """
    Select samples for explainability analysis based on prediction categories.
    """
    
    CATEGORIES = ['true_positive', 'true_negative', 'false_positive', 'false_negative']
    
    def __init__(
        self,
        num_samples_per_category: int = 5,
        selection_strategy: List[str] = None,
        confidence_threshold: Dict[str, float] = None
    ):
        """
        Initialize sample selector.
        
        Args:
            num_samples_per_category: Number of samples to select per category
            selection_strategy: List of categories to select from
            confidence_threshold: Confidence thresholds for high/low
        """
        self.num_samples = num_samples_per_category
        self.strategy = selection_strategy or self.CATEGORIES
        self.thresholds = confidence_threshold or {'high': 0.9, 'low': 0.6}
    
    def _get_confidence_level(self, prediction: float) -> str:
        """Determine confidence level based on prediction."""
        distance_from_boundary = abs(prediction - 0.5)
        
        if distance_from_boundary >= (self.thresholds['high'] - 0.5):
            return 'high'
        elif distance_from_boundary <= (self.thresholds['low'] - 0.5):
            return 'low'
        else:
            return 'medium'
    
    def select_samples(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        paths: List[str]
    ) -> Dict[str, List[SelectedSample]]:
        """
        Select samples for each category.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            paths: Image paths
            
        Returns:
            Dictionary mapping category to list of selected samples
        """
        pred_labels = (predictions >= 0.5).astype(int)
        
        selected = {}
        
        for category in self.strategy:
            # Define mask for each category
            if category == 'true_positive':
                mask = (pred_labels == 1) & (labels == 1)
                sort_desc = True  # Higher confidence first
            elif category == 'true_negative':
                mask = (pred_labels == 0) & (labels == 0)
                sort_desc = False  # Lower prediction (more confident negative) first
            elif category == 'false_positive':
                mask = (pred_labels == 1) & (labels == 0)
                sort_desc = True  # Higher confidence first (worst mistakes)
            elif category == 'false_negative':
                mask = (pred_labels == 0) & (labels == 1)
                sort_desc = False  # Lower prediction first (worst mistakes)
            else:
                print(f"Unknown category: {category}")
                continue
            
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                selected[category] = []
                continue
            
            # Sort by prediction confidence
            if sort_desc:
                sorted_indices = indices[np.argsort(-predictions[indices])]
            else:
                sorted_indices = indices[np.argsort(predictions[indices])]
            
            # Select top N samples
            selected_indices = sorted_indices[:self.num_samples]
            
            samples = []
            for idx in selected_indices:
                sample = SelectedSample(
                    path=paths[idx],
                    label=int(labels[idx]),
                    prediction=float(predictions[idx]),
                    pred_label=int(pred_labels[idx]),
                    category=category,
                    confidence=self._get_confidence_level(predictions[idx])
                )
                samples.append(sample)
            
            selected[category] = samples
        
        return selected
    
    def log_selection(
        self,
        selected: Dict[str, List[SelectedSample]],
        save_path: str = None
    ):
        """
        Log selected samples.
        
        Args:
            selected: Dictionary of selected samples
            save_path: Optional path to save log
        """
        print("\n" + "="*60)
        print("EXPLAINABILITY SAMPLE SELECTION")
        print("="*60)
        
        for category, samples in selected.items():
            print(f"\n{category.upper().replace('_', ' ')} ({len(samples)} samples):")
            print("-" * 40)
            
            for i, sample in enumerate(samples, 1):
                filename = Path(sample.path).name
                print(f"  {i}. {filename}")
                print(f"     Label: {sample.label}, Pred: {sample.prediction:.4f}")
                print(f"     Confidence: {sample.confidence}")
        
        # Save to file if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            log_data = {
                category: [s.to_dict() for s in samples]
                for category, samples in selected.items()
            }
            
            with open(save_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"\nSelection log saved to: {save_path}")
    
    def get_all_paths(self, selected: Dict[str, List[SelectedSample]]) -> List[Tuple[str, str]]:
        """
        Get all selected paths with their categories.
        
        Returns:
            List of (path, category) tuples
        """
        all_paths = []
        for category, samples in selected.items():
            for sample in samples:
                all_paths.append((sample.path, category))
        return all_paths


def select_explainability_samples(
    predictions: np.ndarray,
    labels: np.ndarray,
    paths: List[str],
    config: dict
) -> Dict[str, List[SelectedSample]]:
    """
    Convenience function to select samples based on config.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        paths: Image paths
        config: Configuration dictionary
        
    Returns:
        Selected samples dictionary
    """
    explainability_config = config.get('explainability', {})
    
    selector = SampleSelector(
        num_samples_per_category=explainability_config.get('num_samples_per_category', 5),
        selection_strategy=explainability_config.get('selection_strategy'),
        confidence_threshold=explainability_config.get('confidence_threshold')
    )
    
    return selector.select_samples(predictions, labels, paths)
