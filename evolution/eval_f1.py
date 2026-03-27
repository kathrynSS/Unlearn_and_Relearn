import os

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

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_model_identifiers_from_yaml


def extract_choice(text: str) -> Optional[str]:
    """Extract the first A/B/C/D option letter from generated text."""
    if not text:
        return None
    text_upper = text.upper()
    m = re.search(r"\b([ABCD])\b", text_upper)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text_upper)
    return m.group(1) if m else None


def extract_gt_letter(raw: str) -> Optional[str]:
    """Extract the correct option letter from the answer_letter field."""
    s = str(raw).strip()
    if not s:
        return None
    m = re.match(r"([ABCDabcd])", s)
    if m:
        return m.group(1).upper()
    return None


def calculate_f1_score(
    predictions: List[Optional[str]], 
    ground_truth: List[Optional[str]],
    average: str = 'macro'
) -> Tuple[float, float, dict]:
    """Compute multi-class F1 score, accuracy, and per-class classification report."""
    labels = ['A', 'B', 'C', 'D']
    
    valid_pairs = [
        (pred if pred is not None else 'NONE', gt) 
        for pred, gt in zip(predictions, ground_truth) 
        if gt is not None
    ]
    
    if not valid_pairs:
        return 0.0, 0.0, {}
    
    preds_clean, gt_clean = zip(*valid_pairs)
    preds_clean = list(preds_clean)
    gt_clean = list(gt_clean)
    
    correct = sum(1 for p, g in zip(preds_clean, gt_clean) if p == g)
    accuracy = correct / len(gt_clean)
    
    f1 = f1_score(
        gt_clean, 
        preds_clean, 
        labels=labels, 
        average=average, 
        zero_division=0
    )
    
    report = classification_report(
        gt_clean, 
        preds_clean, 
        labels=labels, 
        output_dict=True, 
        zero_division=0
    )
    
    return f1, accuracy, report


def build_prompt(question: str, options: str, question_start_tag: str, question_end_tag: str) -> str:
    """Build a Mistral instruction-format multiple-choice prompt."""
    options_str = str(options).replace("|", "\n")
    instruction = (
        "你将看到一道单选题，请根据题目和选项选择唯一正确答案。\n\n"
        f"题目：{question}\n\n"
        f"选项：\n{options_str}\n\n"
        "请从 A、B、C、D 四个选项中选择唯一正确答案，"
        "并且**只输出选项字母本身（A/B/C/D）**，不要输出其他任何内容。"
    )
    return f"{question_start_tag}{instruction}{question_end_tag}"


def load_model(model_path: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load Mistral model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def load_model_with_adapter(
    base_model_path: str,
    adapter_path: str,
    device: torch.device,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_choice(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 8,
) -> Tuple[Optional[str], str]:
    """Generate and parse a single-sample answer as A/B/C/D."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    choice = extract_choice(text)
    return choice, text


def find_forget_model_path(forget_cfg_path: str) -> str:
    """Infer the forgotten model path from the training config file."""
    cfg = OmegaConf.load(forget_cfg_path)

    save_dir = Path(str(cfg.save_dir))
    if save_dir.exists():
        return str(save_dir)

    root = Path(str(getattr(cfg, "save_dir_root", "results"))) / str(cfg.model_path) / str(cfg.forget_loss)
    if not root.exists():
        raise FileNotFoundError(f"Forgotten model root directory does not exist: {root} (please confirm training has been completed)")

    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No forgotten model subdirectories found in {root}. Please complete training first.")

    candidates.sort(key=lambda p: p.stat().st_mtime)
    latest_dir = candidates[-1]

    return str(latest_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple-choice accuracy on questions_test.xlsx using Mistral-7B, "
        "comparing the original model with the forgotten model on A/B/C/D choices."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data_construct/raw_data/questions_test.xlsx",
        help="Path to the test set Excel file (must contain question, options, answer_letter columns).",
    )
    parser.add_argument(
        "--forget_cfg",
        type=str,
        default="config/forget_ai.yaml",
        help="Config file used for training the forgotten model (e.g. config/forget_ai.yaml). "
        "The script reads save_dir from it to load the forgotten model.",
    )
    parser.add_argument(
        "--forget_subdir",
        type=str,
        default=None,
        help="Optional: manually specify the subdirectory name of the forgotten model "
             "(e.g. '70_1e-06_forget'), overriding the automatic latest-subdirectory search.",
    )
    parser.add_argument(
        "--forget_adapter_dir",
        type=str,
        default=None,
        help="Optional: if the forgotten result only contains a LoRA adapter "
             "(with adapter_model.safetensors), specify that directory path directly.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional: only evaluate the first N samples for quick testing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional: directory to save results. If specified, saves eval_results_f1.json there. "
             "Otherwise saves to --forget_adapter_dir.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    test_path = Path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test set file does not exist: {test_path}")

    df = pd.read_excel(test_path)
    required_cols = {"question", "options", "answer_letter"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"questions_test.xlsx must contain columns: {required_cols}, got: {set(df.columns)}")

    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"Evaluating only the first {len(df)} samples.")
    else:
        print(f"Evaluating all {len(df)} samples.")

    model_cfg = get_model_identifiers_from_yaml("mistral-7b")
    q_start = model_cfg["question_start_tag"]
    q_end = model_cfg["question_end_tag"]

    forget_cfg = OmegaConf.load(args.forget_cfg)
    base_model_path = forget_cfg.model_path or model_cfg["hf_key"]
    print(f"Loading original (pre-unlearning) model: {base_model_path}")
    base_tokenizer, base_model = load_model(base_model_path, device)

    base_preds = []
    base_raw = []
    gt_full = df["answer_letter"].astype(str).str.strip().tolist()
    gt_letters = [extract_gt_letter(x) for x in gt_full]

    for idx, row in df.iterrows():
        question = str(row["question"])
        options = str(row["options"])
        prompt = build_prompt(question, options, q_start, q_end)
        choice, raw_text = generate_choice(base_tokenizer, base_model, prompt, device)
        base_preds.append(choice)
        base_raw.append(raw_text)
        print(f"[Original] Sample {idx + 1}/{len(df)}: pred={choice}, GT={gt_letters[idx]} (raw: {gt_full[idx]})")

    base_f1, base_acc, base_report = calculate_f1_score(base_preds, gt_letters, average='macro')
    base_correct = sum(1 for pred, gt in zip(base_preds, gt_letters) if pred is not None and gt is not None and pred == gt)

    del base_model, base_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    forget_model_path_str = None
    if args.forget_adapter_dir is not None:
        forget_path = Path(args.forget_adapter_dir).expanduser().resolve()
        if not forget_path.exists():
            raise FileNotFoundError(f"Specified forgotten adapter directory does not exist: {forget_path}")
        forget_model_path = forget_path
        forget_model_path_str = str(forget_model_path)
        is_adapter_only = True
        print(f"\nLoading forgotten model (LoRA adapter mode): {forget_model_path}")
        forget_tokenizer, forget_model = load_model_with_adapter(
            base_model_path=base_model_path,
            adapter_path=str(forget_model_path),
            device=device,
        )
    else:
        forget_model_path_str = None
        auto_dir = Path(find_forget_model_path(args.forget_cfg)).resolve()
        if args.forget_subdir is not None:
            root_dir = auto_dir.parent
            forget_model_path = root_dir / args.forget_subdir
        else:
            forget_model_path = auto_dir

        if not forget_model_path.exists():
            raise FileNotFoundError(f"Forgotten model directory does not exist: {forget_model_path}")

        adapter_file = forget_model_path / "adapter_model.safetensors"
        full_model_files = [
            forget_model_path / "pytorch_model.bin",
            forget_model_path / "model.safetensors",
            forget_model_path / "config.json",
        ]
        is_adapter_only = adapter_file.exists() and not any(p.exists() for p in full_model_files)

        if is_adapter_only:
            print(f"\nDetected LoRA adapter directory (no full weights found), loading base + adapter: {forget_model_path}")
            forget_model_path_str = str(forget_model_path)
            forget_tokenizer, forget_model = load_model_with_adapter(
                base_model_path=base_model_path,
                adapter_path=str(forget_model_path),
                device=device,
            )
        else:
            print(f"\nLoading full forgotten model directory: {forget_model_path}")
            forget_model_path_str = str(forget_model_path)
            forget_tokenizer, forget_model = load_model(
                model_path=str(forget_model_path),
                device=device,
            )

    forget_preds = []
    forget_raw = []

    for idx, row in df.iterrows():
        question = str(row["question"])
        options = str(row["options"])
        prompt = build_prompt(question, options, q_start, q_end)
        choice, raw_text = generate_choice(forget_tokenizer, forget_model, prompt, device)
        forget_preds.append(choice)
        forget_raw.append(raw_text)
        print(f"[Forgotten] Sample {idx + 1}/{len(df)}: pred={choice}, GT={gt_letters[idx]} (raw: {gt_full[idx]})")

    forget_f1, forget_acc, forget_report = calculate_f1_score(forget_preds, gt_letters, average='macro')
    forget_correct = sum(1 for pred, gt in zip(forget_preds, gt_letters) if pred is not None and gt is not None and pred == gt)

    diff_count = sum(
        1
        for base_p, forget_p in zip(base_preds, forget_preds)
        if base_p != forget_p
    )

    print("\n================= F1 Score Summary =================")
    print(f"Original Model Macro F1: {base_f1 * 100:.2f}%")
    print(f"Original Model Accuracy: {base_acc * 100:.2f}% ({base_correct}/{len(df)})")
    print(f"Forgotten Model Macro F1: {forget_f1 * 100:.2f}%")
    print(f"Forgotten Model Accuracy: {forget_acc * 100:.2f}% ({forget_correct}/{len(df)})")
    print(f"Samples with different predictions: {diff_count} / {len(df)}")
    
    print("\n================= Original Model Classification Details =================")
    for label in ['A', 'B', 'C', 'D']:
        if label in base_report:
            print(f"  Class {label}: Precision={base_report[label]['precision']:.3f}, "
                  f"Recall={base_report[label]['recall']:.3f}, F1={base_report[label]['f1-score']:.3f}")
    
    print("\n================= Forgotten Model Classification Details =================")
    for label in ['A', 'B', 'C', 'D']:
        if label in forget_report:
            print(f"  Class {label}: Precision={forget_report[label]['precision']:.3f}, "
                  f"Recall={forget_report[label]['recall']:.3f}, F1={forget_report[label]['f1-score']:.3f}")
    
    results = {
        "base_model_f1": base_f1,
        "base_model_f1_percent": f"{base_f1 * 100:.2f}%",
        "forget_model_f1": forget_f1,
        "forget_model_f1_percent": f"{forget_f1 * 100:.2f}%",
        "base_model_accuracy": base_acc,
        "base_model_accuracy_percent": f"{base_acc * 100:.2f}%",
        "base_model_correct": base_correct,
        "base_model_total": len(df),
        "forget_model_accuracy": forget_acc,
        "forget_model_accuracy_percent": f"{forget_acc * 100:.2f}%",
        "forget_model_correct": forget_correct,
        "forget_model_total": len(df),
        "different_predictions": diff_count,
        "total_samples": len(df),
        "model_path": forget_model_path_str,
        "test_path": str(test_path),
        "base_model_classification_report": base_report,
        "forget_model_classification_report": forget_report,
        "predictions": {
            "base_predictions": base_preds,
            "forget_predictions": forget_preds,
            "ground_truth": gt_letters,
            "base_raw_outputs": base_raw,
            "forget_raw_outputs": forget_raw,
        }
    }
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.forget_adapter_dir:
        output_dir = Path(args.forget_adapter_dir)
    else:
        output_dir = Path(".")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "eval_results_f1.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
