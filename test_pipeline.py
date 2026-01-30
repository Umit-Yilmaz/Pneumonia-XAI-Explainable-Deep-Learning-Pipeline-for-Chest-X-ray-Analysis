"""
Quick Test Script - Pipeline Testing with Minimal Data

Tests the full pipeline with ALL models:
- CNN, ResNet50, DenseNet121, ViT
- 2 images per class
- 2 epochs
- 1 sample per category for explainability
"""

import os
import sys
import shutil
from pathlib import Path
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def create_test_data():
    """Create minimal test dataset from existing data."""
    print("\n" + "="*50)
    print("Creating minimal test dataset...")
    print("="*50)
    
    # Source and destination
    cnn_source = PROJECT_ROOT / 'data' / 'cnn-normalized'
    other_source = PROJECT_ROOT / 'data' / 'other-normalized'
    
    test_cnn = PROJECT_ROOT / 'data' / 'test-cnn'
    test_other = PROJECT_ROOT / 'data' / 'test-other'
    
    # Clean up existing test data
    for test_dir in [test_cnn, test_other]:
        if test_dir.exists():
            shutil.rmtree(test_dir)
    
    # Copy 2 images per class per split
    for source, dest in [(cnn_source, test_cnn), (other_source, test_other)]:
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                src_dir = source / split / class_name
                dst_dir = dest / split / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                if src_dir.exists():
                    images = list(src_dir.glob('*.png'))[:2]  # Take first 2 images
                    for img_path in images:
                        shutil.copy(img_path, dst_dir / img_path.name)
                    print(f"  Copied {len(images)} images to {dst_dir}")
    
    print("✓ Test dataset created")
    return test_cnn, test_other


def create_all_test_configs(test_cnn: Path, test_other: Path):
    """Create test configurations for ALL models."""
    print("\n" + "="*50)
    print("Creating test configurations for ALL models...")
    print("="*50)
    
    configs_dir = PROJECT_ROOT / 'configs'
    test_configs = {}
    
    # Common test settings
    common_training = """
training:
  optimizer: adam
  batch_size: 2
  epochs: 2
  loss: binary_crossentropy
  early_stopping:
    monitor: val_loss
    patience: 2
    restore_best_weights: true
  reduce_lr:
    monitor: val_loss
    factor: 0.5
    patience: 1
    min_lr: 0.00001
  seed: 42

metrics:
  - accuracy
  - roc_auc

explainability:
  enabled: true
  num_samples_per_category: 1
  selection_strategy:
    - true_positive
  confidence_threshold:
    high: 0.9
    low: 0.6
  save_individual: true
  save_overlay: true

results:
  logs_dir: results/test_logs
  figures_dir: results/test_figures
  tables_dir: results/test_tables
  models_dir: results/test_models
"""

    # 1. CNN Test Config
    cnn_config = f"""# CNN Test Configuration (Minimal)
model:
  name: CustomCNN
  input_size: 224
  channels: 3
  num_classes: 1
  conv_blocks: 3
  filters: [16, 32, 64]
  kernel_size: 3
  pool_size: 2
  dropout: [0.2, 0.2, 0.3]
  fc_units: [64]
  activation: relu
  output_activation: sigmoid

{common_training}
  learning_rate: 0.001

data:
  train_path: {test_cnn}/train
  val_path: {test_cnn}/val
  test_path: {test_cnn}/test
  normalization: divide_255

explainability:
  enabled: true
  method: gradcam
  num_samples_per_category: 1
  selection_strategy:
    - true_positive
  confidence_threshold:
    high: 0.9
    low: 0.6
  save_individual: true
  save_overlay: true
"""
    with open(configs_dir / 'cnn_test.yaml', 'w') as f:
        f.write(cnn_config)
    test_configs['CNN'] = 'configs/cnn_test.yaml'

    # 2. ResNet Test Config
    resnet_config = f"""# ResNet Test Configuration (Minimal)
model:
  name: ResNet50
  input_size: 224
  channels: 3
  num_classes: 1
  pretrained: imagenet
  freeze_blocks: [0, 1, 2, 3]
  finetune_blocks: [4]
  fc_units: [64]
  dropout: 0.3

{common_training}
  learning_rate: 0.0001
  weight_decay: 0.0001

data:
  train_path: {test_other}/train
  val_path: {test_other}/val
  test_path: {test_other}/test
  normalization: imagenet

explainability:
  enabled: true
  method: gradcam_pp
  num_samples_per_category: 1
  selection_strategy:
    - true_positive
  confidence_threshold:
    high: 0.9
    low: 0.6
  save_individual: true
  save_overlay: true
"""
    with open(configs_dir / 'resnet_test.yaml', 'w') as f:
        f.write(resnet_config)
    test_configs['ResNet50'] = 'configs/resnet_test.yaml'

    # 3. DenseNet Test Config
    densenet_config = f"""# DenseNet Test Configuration (Minimal)
model:
  name: DenseNet121
  input_size: 224
  channels: 3
  num_classes: 1
  pretrained: imagenet
  finetune_from: denseblock4
  fc_units: [64]
  dropout: 0.3

{common_training}
  learning_rate: 0.0001

data:
  train_path: {test_other}/train
  val_path: {test_other}/val
  test_path: {test_other}/test
  normalization: imagenet

explainability:
  enabled: true
  method: gradcam
  num_samples_per_category: 1
  selection_strategy:
    - true_positive
  confidence_threshold:
    high: 0.9
    low: 0.6
  save_individual: true
  save_overlay: true
"""
    with open(configs_dir / 'densenet_test.yaml', 'w') as f:
        f.write(densenet_config)
    test_configs['DenseNet121'] = 'configs/densenet_test.yaml'

    # 4. ViT Test Config
    vit_config = f"""# ViT Test Configuration (Minimal)
model:
  name: ViT
  input_size: 224
  channels: 3
  num_classes: 1
  patch_size: 16
  model_variant: vit_tiny_patch16_224
  pretrained: imagenet
  fc_units: [64]
  dropout: 0.1

{common_training}
  learning_rate: 0.00003
  weight_decay: 0.01
  batch_size: 2

data:
  train_path: {test_other}/train
  val_path: {test_other}/val
  test_path: {test_other}/test
  normalization: imagenet

explainability:
  enabled: true
  method: attention_rollout
  num_samples_per_category: 1
  selection_strategy:
    - true_positive
  confidence_threshold:
    high: 0.9
    low: 0.6
  save_individual: true
  save_overlay: true
"""
    with open(configs_dir / 'vit_test.yaml', 'w') as f:
        f.write(vit_config)
    test_configs['ViT'] = 'configs/vit_test.yaml'
    
    print("✓ Test configs created for: CNN, ResNet50, DenseNet121, ViT")
    return test_configs


def test_single_model(model_key: str, config_path: str, device: torch.device):
    """Test a single model through the pipeline."""
    from models import create_model, load_config
    from training import create_dataloaders, train_model, evaluate_model
    from evaluation import calculate_metrics, print_metrics, plot_confusion_matrix
    from explainability import visualize
    from explainability.sample_selection import SampleSelector
    
    print(f"\n{'#'*60}")
    print(f"# TESTING: {model_key}")
    print('#'*60)
    
    try:
        # Load config
        config = load_config(config_path)
        model_name = config['model']['name']
        
        results_config = config.get('results', {})
        figures_dir = Path(results_config.get('figures_dir', 'results/test_figures'))
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        print(f"\nCreating {model_name}...")
        model = create_model(config)
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {param_count:,}")
        print(f"  Trainable params: {trainable_count:,}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(config, num_workers=0)
        print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
        
        # Train
        print("\nTraining...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            save_dir=results_config.get('models_dir', 'results/test_models')
        )
        
        # Evaluate
        print("\nEvaluating...")
        predictions, labels, paths = evaluate_model(model, test_loader, device)
        metrics = calculate_metrics(predictions, labels)
        print_metrics(metrics, f"{model_name} (Test)")
        
        # Confusion matrix
        plot_confusion_matrix(
            predictions, labels,
            save_path=str(figures_dir / f"{model_name}_test_cm.png"),
            model_name=f"{model_name} (Test)"
        )
        
        # Explainability (simplified)
        explainability_config = config.get('explainability', {})
        if explainability_config.get('enabled', False):
            print("\nRunning explainability...")
            method = explainability_config.get('method', 'gradcam')
            normalization = config.get('data', {}).get('normalization', 'imagenet')
            
            selector = SampleSelector(
                num_samples_per_category=1,
                selection_strategy=['true_positive', 'false_positive']
            )
            selected = selector.select_samples(predictions, labels, paths)
            
            # Save logs for clinical analysis script
            logs_dir = Path(results_config.get('logs_dir', 'results/test_logs'))
            selector.log_selection(
                selected, 
                save_path=str(logs_dir / f"{model_name}_selected_samples.json")
            )
            # Just try one sample
            if len(paths) > 0:
                try:
                    save_path = str(figures_dir / f"{model_name}_test_xai.png")
                    visualize(
                        method=method,
                        model=model,
                        image_path=paths[0],
                        save_path=save_path,
                        device=device,
                        normalization=normalization
                    )
                    print(f"  ✓ Saved: {save_path}")
                except Exception as e:
                    print(f"  ✗ Explainability error: {e}")
        
        print(f"\n✓ {model_key} test completed successfully!")
        return True, metrics
        
    except Exception as e:
        print(f"\n✗ {model_key} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_test_pipeline():
    """Run the complete test pipeline for ALL models."""
    print("\n" + "="*60)
    print(" PNEUMONIA-XAI: TESTING ALL MODELS")
    print("="*60)
    
    # Create test data
    test_cnn, test_other = create_test_data()
    
    # Create test configs for all models
    test_configs = create_all_test_configs(test_cnn, test_other)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test all models
    results = {}
    for model_key, config_path in test_configs.items():
        success, metrics = test_single_model(model_key, config_path, device)
        results[model_key] = {'success': success, 'metrics': metrics}
    
    # Summary
    print("\n" + "#"*60)
    print("# TEST SUMMARY")
    print("#"*60)
    
    all_passed = True
    for model_key, result in results.items():
        status = "✓ PASSED" if result['success'] else "✗ FAILED"
        print(f"  {model_key}: {status}")
        if not result['success']:
            all_passed = False
    
    if all_passed:
        print("\n" + "#"*60)
        print("# ALL TESTS PASSED! main.py should work correctly.")
        print("#"*60)
    else:
        print("\n" + "#"*60)
        print("# SOME TESTS FAILED! Check errors above.")
        print("#"*60)
    
    print(f"\nTest outputs saved to:")
    print(f"  - results/test_logs/")
    print(f"  - results/test_figures/")
    print(f"  - results/test_models/")
    
    return all_passed


if __name__ == '__main__':
    run_test_pipeline()
