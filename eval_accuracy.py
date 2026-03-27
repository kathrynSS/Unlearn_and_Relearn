"""
Accuracy evaluation script.

Evaluates model accuracy on a given dataset. Supports:
1. Original model (before unlearning)
2. Unlearned model
3. Relearned model

Usage:
    # Evaluate original model
    python eval_accuracy.py model_path=mistralai/Mistral-7B-Instruct-v0.3 \
        data_path=data_construct/data/data_ai_progressive split=forget_10 \
        use_pretrained=true

    # Evaluate unlearned model
    python eval_accuracy.py model_path=results/xxx/intervention/xxx \
        data_path=data_construct/data/data_ai_progressive split=forget_10

    # Evaluate relearned model
    python eval_accuracy.py model_path=results/relearn/xxx \
        data_path=data_construct/data/data_ai_progressive split=forget_10
"""

import os

# ============= Local resource path config (must be set before importing other libraries) =============
LOCAL_RESOURCES_DIR = "/root/autodl-tmp/local_resources"
os.makedirs(LOCAL_RESOURCES_DIR, exist_ok=True)

HF_LOCAL_DIR = os.path.join(LOCAL_RESOURCES_DIR, "huggingface_models")
os.makedirs(HF_LOCAL_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_LOCAL_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_LOCAL_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_LOCAL_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_LOCAL_DIR, "datasets"))

XDG_CACHE_DIR = os.path.join(LOCAL_RESOURCES_DIR, "cache")
os.makedirs(XDG_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_DIR)
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
import hydra
from omegaconf import DictConfig
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

from data_module import load_dataset_auto
from utils import get_model_identifiers_from_yaml


def load_model(model_path, model_id, use_pretrained=False):
    """Load model (supports both regular and LoRA models)."""
    config = AutoConfig.from_pretrained(model_id)
    
    if use_pretrained:
        print(f"Loading pretrained model from {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    elif os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print(f"Detected LoRA adapter at {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        print("LoRA adapter loaded and merged")
    else:
        print(f"Loading full model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    
    model.eval()
    return model


def evaluate_accuracy(model, tokenizer, dataset, model_configs, batch_size=8, max_samples=None):
    """Evaluate model accuracy on a dataset."""
    if max_samples and max_samples < len(dataset):
        indices = list(range(min(max_samples, len(dataset))))
        dataset = dataset.select(indices)
    
    question_start_tag = model_configs['question_start_tag']
    question_end_tag = model_configs['question_end_tag']
    answer_tag = model_configs['answer_tag']
    
    results = []
    total_loss = 0
    correct_exact = 0
    correct_contains = 0
    
    device = next(model.parameters()).device
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch_items = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        
        for item in batch_items:
            question = item['question']
            answer = item['answer']
            
            prompt = question_start_tag + question + question_end_tag + answer_tag
            
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            full_text = prompt + answer
            full_inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
            full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
            
            with torch.no_grad():
                loss_outputs = model(**full_inputs, labels=full_inputs['input_ids'])
                loss = loss_outputs.loss.item()
            
            total_loss += loss
            
            answer_clean = answer.strip()
            generated_clean = generated_text.strip()
            
            answer_letter = answer_clean.split('.')[0].strip() if '.' in answer_clean else answer_clean
            generated_letter = generated_clean.split('.')[0].strip() if '.' in generated_clean else generated_clean[:1]
            
            exact_match = answer_letter.upper() == generated_letter.upper()
            contains_match = answer_letter.upper() in generated_clean.upper()
            
            if exact_match:
                correct_exact += 1
            if contains_match:
                correct_contains += 1
            
            results.append({
                'question': question,
                'ground_truth': answer,
                'generated': generated_text,
                'exact_match': exact_match,
                'contains_match': contains_match,
                'loss': loss
            })
    
    n = len(results)
    return {
        'exact_match_accuracy': correct_exact / n if n > 0 else 0,
        'contains_accuracy': correct_contains / n if n > 0 else 0,
        'avg_loss': total_loss / n if n > 0 else 0,
        'num_samples': n,
        'correct_exact': correct_exact,
        'correct_contains': correct_contains,
        'results': results
    }


@hydra.main(version_base=None, config_path="config", config_name="eval_accuracy")
def main(cfg: DictConfig):
    print("=" * 60)
    print("Model Accuracy Evaluation")
    print("=" * 60)
    
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = load_model(cfg.model_path, model_id, cfg.get('use_pretrained', False))
    
    print(f"\nLoading dataset from {cfg.data_path}, split: {cfg.split}")
    dataset = load_dataset_auto(cfg.data_path, cfg.split)
    print(f"Dataset size: {len(dataset)}")
    
    print("\nStarting evaluation...")
    results = evaluate_accuracy(
        model, 
        tokenizer, 
        dataset, 
        model_cfg,
        batch_size=cfg.get('batch_size', 8),
        max_samples=cfg.get('max_samples', None)
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Dataset: {cfg.data_path}")
    print(f"Split: {cfg.split}")
    print(f"Samples: {results['num_samples']}")
    print(f"\nExact match accuracy: {results['exact_match_accuracy']:.4f} ({results['correct_exact']}/{results['num_samples']})")
    print(f"Contains match accuracy: {results['contains_accuracy']:.4f} ({results['correct_contains']}/{results['num_samples']})")
    print(f"Average loss: {results['avg_loss']:.4f}")
    
    if cfg.get('save_dir'):
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'model_path': cfg.model_path,
            'data_path': cfg.data_path,
            'split': cfg.split,
            'exact_match_accuracy': results['exact_match_accuracy'],
            'contains_accuracy': results['contains_accuracy'],
            'avg_loss': results['avg_loss'],
            'num_samples': results['num_samples'],
            'correct_exact': results['correct_exact'],
            'correct_contains': results['correct_contains'],
        }
        
        with open(save_dir / "eval_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        if cfg.get('save_details', True):
            with open(save_dir / "eval_details.json", 'w', encoding='utf-8') as f:
                json.dump(results['results'], f, indent=4, ensure_ascii=False)
        
        print(f"\nResults saved to: {save_dir}")
    
    return results


if __name__ == "__main__":
    main()

