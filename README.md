# Pneumonia-XAI: Explainable Deep Learning Pipeline for Chest X-ray Analysis

This repository provides a fully automated, reproducible, and explainable deep learning pipeline for pneumonia detection from chest X-ray images.
The system integrates multiple vision models with post-hoc explainability methods and a medical vision–language model (MedGemma) for structured clinical interpretation.

The project is designed to support:

* Comparative evaluation of deep learning architectures
* Explainable AI (XAI) analysis
* Post-hoc clinical reasoning without diagnostic claims
* Reproducible experimentation suitable for academic publication

---

## Project Objectives

1. Train and evaluate multiple deep learning models for pneumonia detection:

   * Custom CNN
   * ResNet50
   * DenseNet121
   * Vision Transformer (ViT-B/16)

2. Provide model-specific explainability using:

   * Grad-CAM
   * Grad-CAM++
   * Attention Rollout (ViT)

3. Integrate MedGemma as a **post-hoc reasoning layer** to:

   * Summarize XAI outputs
   * Generate observational clinical reports
   * Assess cross-model consistency
   * Avoid diagnosis or treatment recommendations

4. Ensure reproducibility, transparency, and reviewer-friendly reporting.

---

## Repository Structure

```
pneumonia-xai/
│
├── configs/
│   ├── cnn.yaml
│   ├── resnet.yaml
│   ├── densenet.yaml
│   └── vit.yaml
│
├── models/
│   ├── cnn.py
│   ├── resnet.py
│   ├── densenet.py
│   ├── vit.py
│   └── __init__.py
│
├── data/
│   ├── cnn-normalized/
│   └── other-normalized/
│
├── training/
│   ├── train.py
│   ├── validate.py
│   └── data_loader.py
│
├── evaluation/
│   ├── metrics.py
│   ├── confusion_matrix.py
│   └── roc_curve.py
│
├── explainability/
│   ├── gradcam.py
│   ├── gradcam_pp.py
│   └── attention_rollout.py
│
├── medgemma/
│   ├── med_gemma_clinical_analysis.py
│   └── prompts/
│       ├── clinical_explanation.txt
│       ├── false_positive_analysis.txt
│       ├── false_negative_analysis.txt
│       ├── confidence_alignment.txt
│       └── cross_model_comparison.txt
│
├── results/
│   ├── logs/
│   ├── figures/
│   └── tables/
│
├── main.py
└── README.md
```

---

## Dataset and Preprocessing

* Input modality: Chest X-ray images (RGB)
* Input resolution: 224 × 224
* Class labels: Normal, Pneumonia
* Class distribution: Balanced (50% / 50%)

### Preprocessing Steps

1. Resize images to 224 × 224
2. Convert to float32
3. Normalize using ImageNet statistics:

   * Mean: [0.485, 0.456, 0.406]
   * Std:  [0.229, 0.224, 0.225]
4. Data augmentation :

   * Random horizontal flip
   * Random rotation 

Separate normalized datasets are maintained for:

* CNN (custom normalization)
* ImageNet-pretrained models (ResNet, DenseNet, ViT)

---

## Models and Training Configuration

### Common Training Settings

* Optimizer: Adam
* Loss function: Binary Cross Entropy
* Early stopping: patience = 5
* ReduceLROnPlateau: factor = 0.1, patience = 3
* Random seed: 42

### Model-Specific Details

| Model       | Pretrained   | Learning Rate | Batch Size | Explainability    |
| ----------- | ------------ | ------------- | ---------- | ----------------- |
| CNN         | No           | 1e-3          | 32         | Grad-CAM          |
| ResNet50    | ImageNet     | 1e-4          | 32         | Grad-CAM++        |
| DenseNet121 | ImageNet     | 1e-4          | 32         | Grad-CAM          |
| ViT-B/16    | ImageNet-21k | 3e-5          | 16         | Attention Rollout |

### Training Strategies

1. **Custom CNN**: Trained from scratch with a 5-block architecture. Uses progressive filters (32→256), BatchNorm, and Dropout to learn features directly from the pneumonia dataset.
2. **ResNet50 & DenseNet121**: Partial fine-tuning approach. The early layers (blocks 0-2 for ResNet) are frozen to retain general ImageNet features, while later blocks are fine-tuned to adapt to radiological patterns.
3. **Vision Transformer (ViT)**: Full fine-tuning from ImageNet-21k weights with a lower learning rate (3e-5) to maintain the stability of self-attention mechanisms during adaptation.
4. **Early Stopping & LR Scheduling**: All models employ `EarlyStopping` (patience=5) and `ReduceLROnPlateau` to ensure convergence and prevent overfitting.
---

## Pipeline Overview

<img width="8191" height="2567" alt="flow-chart" src="https://github.com/user-attachments/assets/88c8b8bd-e001-4b55-a8bc-55b50e0bd3e5" />


When running `main.py`, the following steps are executed sequentially for each model:

1. Load configuration file
2. Initialize model architecture
3. Train model on training set
4. Validate on validation set
5. Evaluate on test set
6. Save metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
7. Generate confusion matrix and ROC curve
8. Produce XAI visualizations
9. Trigger MedGemma-based clinical analysis
10. Store all outputs in `results/`

---

## Explainable AI (XAI) Results

The pipeline generates visual explanations to interpret model decisions. Below are examples of XAI outputs:

### Model Interpretability (Custom CNN)
<img width="1762" height="608" alt="CustomCNN_test_xai" src="https://github.com/user-attachments/assets/e748d2b5-8e3b-4b94-8ee4-15fe75f1aec6" />


### Grad-CAM Saliency Map
<img width="1762" height="608" alt="test_true_positive_gradcam" src="https://github.com/user-attachments/assets/7aa61e81-0abe-4c29-8183-d11d70b33626" />

---

## MedGemma Integration

MedGemma is used strictly as a **post-hoc interpretability and reporting module**.

### Inputs to MedGemma

1. Visual data:

   * Original chest X-ray
   * XAI heatmap (Grad-CAM / Attention Rollout)

2. Model outputs:

   * Prediction label
   * Confidence score
   * Model name

3. Structured prompt (task-specific)

### Outputs from MedGemma

* Textual explanation of visual patterns
* Radiological observations aligned with XAI
* Confidence–attention consistency analysis
* Cross-model comparison (optional)

### Safety Constraints

* No medical diagnosis
* No treatment recommendation
* Observational language only
* Model behavior explanation, not clinical decision-making

---

## Methodology Summary

This study follows a **model-agnostic explainable AI framework**:

* Train multiple architectures on the same dataset
* Apply architecture-appropriate XAI techniques
* Compare attention localization and confidence alignment
* Use a medical vision–language model to translate explanations into clinically interpretable narratives
* Validate interpretability through expert review (questionnaire-based)

---

## Reproducibility Checklist

* Fixed random seed
* Explicit train/val/test splits
* Versioned configuration files (YAML)
* Pretrained model sources documented
* Deterministic preprocessing pipeline
* Saved metrics, plots, and logs
* Prompt templates included
* No hidden manual steps

---

## How to Run

```bash
python main.py
```

All results will be saved automatically under the `results/` directory.

---

## Intended Use and Limitations

This project is intended for:

* Research and academic experimentation
* Explainable AI studies
* Model interpretability evaluation

This system is **not** intended for clinical diagnosis or deployment in real-world healthcare settings.

---

## Citation

If you use this repository in your research, please cite:

> Explainable Deep Learning with Vision–Language Reasoning for Pneumonia Detection from Chest X-rays
