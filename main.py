"""
Pneumonia-XAI: Fully Automated Experiment Pipeline

This script orchestrates the complete experiment:
1. Train all models (CNN, ResNet50, DenseNet121, ViT)
2. Validate on validation set
3. Evaluate on test set
4. Generate evaluation metrics and plots
5. Run explainability analysis on selected samples
6. Save all results

Usage:
    python main.py
    python main.py --model cnn  # Train only CNN
    python main.py --skip-training  # Only evaluate pre-trained models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_model, load_config
from training import create_dataloaders, train_model, evaluate_model
from evaluation import (
    calculate_metrics, print_metrics, save_metrics_to_file,
    plot_confusion_matrix, plot_confusion_matrix_comparison,
    plot_roc_curve, plot_roc_curves_comparison
)
from explainability import visualize
from explainability.sample_selection import SampleSelector


# Configuration files
CONFIG_FILES = {
    'CNN': 'configs/cnn.yaml',
    'ResNet50': 'configs/resnet.yaml',
    'DenseNet121': 'configs/densenet.yaml',
    'ViT': 'configs/vit.yaml'
}


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def run_single_model_pipeline(
    model_key: str,
    config_path: str,
    device: torch.device,
    skip_training: bool = False,
    models_dir: str = 'results/models'
) -> Dict:
    """
    Run complete pipeline for a single model.
    
    Args:
        model_key: Model identifier (e.g., 'CNN')
        config_path: Path to config file
        device: Device to use
        skip_training: Skip training and load pre-trained model
        models_dir: Directory containing saved models
        
    Returns:
        Dictionary with results
    """
    print("\n" + "#"*60)
    print(f"# RUNNING PIPELINE FOR: {model_key}")
    print("#"*60)
    
    # Load config
    config = load_config(config_path)
    model_name = config['model']['name']
    
    results_config = config.get('results', {})
    logs_dir = Path(results_config.get('logs_dir', 'results/logs'))
    figures_dir = Path(results_config.get('figures_dir', 'results/figures'))
    tables_dir = Path(results_config.get('tables_dir', 'results/tables'))
    
    # Ensure directories exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print(f"\nCreating model: {model_name}")
    model = create_model(config)
    model = model.to(device)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, num_workers=0)
    
    # Training
    if not skip_training:
        print("\n" + "="*50)
        print("TRAINING")
        print("="*50)
        
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            save_dir=models_dir
        )
    else:
        # Load pre-trained model
        model_path = Path(models_dir) / f"{model_name}_best.pth"
        if model_path.exists():
            print(f"\nLoading pre-trained model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: No pre-trained model found at {model_path}")
    
    # Evaluation
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    
    predictions, labels, paths = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    print_metrics(metrics, model_name)
    
    # Save metrics
    save_metrics_to_file(
        metrics,
        str(tables_dir / f"{model_name}_metrics.txt"),
        model_name
    )
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        predictions, labels,
        save_path=str(figures_dir / f"{model_name}_confusion_matrix.png"),
        model_name=model_name
    )
    
    # Generate ROC curve
    print("Generating ROC curve...")
    plot_roc_curve(
        predictions, labels,
        save_path=str(figures_dir / f"{model_name}_roc_curve.png"),
        model_name=model_name
    )
    
    # Explainability
    explainability_config = config.get('explainability', {})
    
    if explainability_config.get('enabled', False):
        print("\n" + "="*50)
        print("EXPLAINABILITY ANALYSIS")
        print("="*50)
        
        method = explainability_config.get('method', 'gradcam')
        normalization = config.get('data', {}).get('normalization', 'imagenet')
        
        # Select samples
        selector = SampleSelector(
            num_samples_per_category=explainability_config.get('num_samples_per_category', 5),
            selection_strategy=explainability_config.get('selection_strategy'),
            confidence_threshold=explainability_config.get('confidence_threshold')
        )
        
        selected = selector.select_samples(predictions, labels, paths)
        
        # Log selection
        selector.log_selection(
            selected,
            save_path=str(logs_dir / f"{model_name}_selected_samples.json")
        )
        
        # Generate visualizations
        explainability_dir = figures_dir / f"{model_name}_explainability"
        explainability_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {method} visualizations...")
        
        for category, samples in selected.items():
            category_dir = explainability_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            for i, sample in enumerate(samples):
                try:
                    save_path = str(category_dir / f"sample_{i+1}.png")
                    visualize(
                        method=method,
                        model=model,
                        image_path=sample.path,
                        save_path=save_path,
                        device=device,
                        normalization=normalization
                    )
                except Exception as e:
                    print(f"  Error visualizing {sample.path}: {e}")
        
        print(f"✓ Explainability visualizations saved to: {explainability_dir}")
    
    # Return results
    return {
        'model_name': model_name,
        'metrics': metrics,
        'predictions': predictions,
        'labels': labels,
        'paths': paths
    }


def run_full_pipeline(
    models: Optional[List[str]] = None,
    skip_training: bool = False
):
    """
    Run the complete experiment pipeline.
    
    Args:
        models: List of model keys to run (None = all models)
        skip_training: Skip training phase
    """
    print("\n" + "="*60)
    print(" PNEUMONIA-XAI: AUTOMATED EXPERIMENT PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Determine which models to run
    if models is None:
        models = list(CONFIG_FILES.keys())
    
    print(f"\nModels to run: {models}")
    print(f"Skip training: {skip_training}")
    
    # Run each model
    all_results = {}
    
    for model_key in models:
        if model_key not in CONFIG_FILES:
            print(f"Warning: Unknown model '{model_key}', skipping...")
            continue
        
        try:
            results = run_single_model_pipeline(
                model_key=model_key,
                config_path=CONFIG_FILES[model_key],
                device=device,
                skip_training=skip_training
            )
            all_results[model_key] = results
        except Exception as e:
            print(f"Error running {model_key}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison plots
    if len(all_results) > 1:
        print("\n" + "="*50)
        print("GENERATING COMPARISON PLOTS")
        print("="*50)
        
        figures_dir = Path('results/figures')
        tables_dir = Path('results/tables')
        
        # Prepare data for comparison
        comparison_data = {
            name: (res['predictions'], res['labels'])
            for name, res in all_results.items()
        }
        
        # ROC comparison
        print("Generating ROC curve comparison...")
        plot_roc_curves_comparison(
            comparison_data,
            save_path=str(figures_dir / 'roc_comparison.png')
        )
        
        # Confusion matrix comparison
        print("Generating confusion matrix comparison...")
        plot_confusion_matrix_comparison(
            comparison_data,
            save_path=str(figures_dir / 'confusion_matrix_comparison.png')
        )
        
        # Metrics summary table
        print("Generating metrics summary...")
        metrics_summary = []
        for name, res in all_results.items():
            row = {'Model': name}
            row.update({k: v for k, v in res['metrics'].items() 
                       if isinstance(v, (int, float)) and k not in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']})
            metrics_summary.append(row)
        
        df = pd.DataFrame(metrics_summary)
        df.to_csv(tables_dir / 'metrics_summary.csv', index=False)
        print(f"\nMetrics summary saved to: {tables_dir / 'metrics_summary.csv'}")
        
        # Print summary table
        print("\n" + "="*60)
        print("FINAL METRICS SUMMARY")
        print("="*60)
        print(df.to_string(index=False))
    
    # Final summary
    print("\n" + "#"*60)
    print("# PIPELINE COMPLETED SUCCESSFULLY!")
    print("#"*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Pneumonia-XAI Experiment Pipeline')
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['cnn', 'resnet', 'densenet', 'vit', 'all'],
        default='all',
        help='Model to train/evaluate (default: all)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and use pre-trained models'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Map model argument to keys
    model_map = {
        'cnn': ['CNN'],
        'resnet': ['ResNet50'],
        'densenet': ['DenseNet121'],
        'vit': ['ViT'],
        'all': None  # Run all
    }
    
    models = model_map.get(args.model, None)
    
    run_full_pipeline(
        models=models,
        skip_training=args.skip_training
    )


if __name__ == '__main__':
    main()
