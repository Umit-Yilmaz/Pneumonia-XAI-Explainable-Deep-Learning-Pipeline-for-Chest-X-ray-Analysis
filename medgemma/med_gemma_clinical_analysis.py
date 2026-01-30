"""
MedGemma Clinical Analysis Script

This script generates clinical explanations for model predictions and XAI outputs
using the MedGemma vision-language model.

It runs after main.py and processes the selected samples for each model.
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from transformers import pipeline

class MedGemmaAnalyzer:
    """
    Analyzer that uses MedGemma to generate clinical explanations.
    """
    
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", device: str = None, token: str = None):
        """
        Initialize the MedGemma pipeline.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use (cuda/cpu)
            token: HuggingFace API token (required for gated models)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Use token from argument or environment variable
        self.hf_token = token or os.environ.get("HF_TOKEN")
        
        print(f"Initializing MedGemma analyzer on {self.device}...")
        # No mock fallback - if it fails, it should raise an error
        self.pipe = pipeline(
            "image-text-to-text", 
            model=model_id, 
            device=self.device,
            token=self.hf_token
        )

    def load_prompt(self, category: str) -> str:
        """
        Load prompt text from the medgemma/prompts directory.
        """
        # Security/Policy constraint included in all prompts
        safety_notice = (
            "Important:\n"
            "You must not provide medical diagnoses, treatment advice, or clinical decisions.\n"
            "Your role is limited to explaining model behavior based on visual information.\n\n"
        )
        
        prompt_path = Path("medgemma/prompts") / f"{category}_analysis.txt"
        if not prompt_path.exists():
            # Fallback to general explanation if specific category analysis prompt is missing
            prompt_path = Path("medgemma/prompts/clinical_explanation.txt")
            
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                return f.read()
        return "Generate a clinical explanation for this model prediction."

    def analyze_sample(self, sample_info: Dict, prompt: str) -> str:
        """
        Generate explanation for a single sample.
        """
        # Construct message for MedGemma
        image_path = sample_info['path']
        
        # Metadata string
        metadata = f"Model: {sample_info.get('model', 'Deep Learning Model')}\n"
        metadata += f"Prediction: {'Pneumonia' if sample_info['pred_label'] == 1 else 'Normal'}\n"
        metadata += f"Confidence: {sample_info['prediction']:.4f}\n"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": f"{prompt}\n\nInput Metadata:\n{metadata}"}
                ]
            },
        ]
        
        try:
            output = self.pipe(text=messages)
            return output[0].get('generated_text', "No explanation generated.")
        except Exception as e:
            return f"Error during generation: {e}"

def run_clinical_analysis():
    """
    Main orchestrator for clinical analysis.
    """
    print("\n" + "="*60)
    print(" MEDGEMMA CLINICAL ANALYSIS PIPELINE")
    print("="*60)
    
    # Setup directories - check both production and test folders
    logs_dirs = [Path("results/logs"), Path("results/test_logs")]
    reports_dir = Path("results/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MedGemmaAnalyzer()
    
    # Find all selection logs across potential directories
    selection_files = []
    for d in logs_dirs:
        if d.exists():
            selection_files.extend(list(d.glob("*_selected_samples.json")))
    
    if not selection_files:
        print("No selected samples found in results/logs or results/test_logs.")
        print("Run main.py or test_pipeline.py first.")
        return

    all_reports = {}

    for log_file in selection_files:
        model_name = log_file.name.replace("_selected_samples.json", "")
        print(f"\nProcessing model: {model_name}")
        
        with open(log_file, 'r') as f:
            selected_samples = json.load(f)
            
        model_reports = []
        
        for category, samples in selected_samples.items():
            print(f"  Analyzing category: {category} ({len(samples)} samples)")
            
            # Load specific prompt for this category
            prompt = analyzer.load_prompt(category)
            
            for i, sample in enumerate(samples):
                # Update sample with model info
                sample['model'] = model_name
                
                # Generate explanation
                explanation = analyzer.analyze_sample(sample, prompt)
                
                report_item = {
                    "image": sample['path'],
                    "category": category,
                    "prediction": sample['prediction'],
                    "explanation": explanation
                }
                model_reports.append(report_item)
                
        # Save model reports
        report_file = reports_dir / f"{model_name}_clinical_report.json"
        with open(report_file, 'w') as f:
            json.dump(model_reports, f, indent=2)
        
        all_reports[model_name] = report_file
        print(f"✓ Saved clinical report for {model_name} to {report_file}")

    print("\n" + "#"*60)
    print("# CLINICAL ANALYSIS COMPLETED")
    print("#"*60)

if __name__ == "__main__":
    run_clinical_analysis()