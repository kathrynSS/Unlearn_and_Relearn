# Simulating Novice Students Using Machine Unlearning and Relearning in Large Language Models

[![Paper](https://img.shields.io/badge/Paper-arXiv%202603.26142-red)](https://arxiv.org/abs/2603.26142)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## Overview

This project implements a framework for simulating novice student behavior in LLMs through three stages: Stage 1. Machine Unlearning, Stage 2. Machine Relearning, and Stage 3. Teachable-Agent Interaction.

<p align="center">
  <img src="pic/pic2.png" width="70%">
</p>

## Requirements

- Python 3.8+
- CUDA-compatible GPU

## Installation

```bash
pip install -r requirements.txt
```

For LLM-guided relearning (Step 3 Option B), set the following environment variables:

```bash
export OPENAI_API_KEY="your-deepseek-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```

## Usage

 The dataset supports progressive forget splits: `forget_10`, `forget_20`, ..., `forget_50` (representing 10%--50% of knowledge points).

### Step 1: Generate Teacher Distribution

Produce soft label distributions by feeding perturbed inputs (with incorrect answer options substituted) through the base model. This is a prerequisite for the unlearning step.

```bash
CUDA_VISIBLE_DEVICES=0 python teacher.py --config-name=forget_ai.yaml \
    data_path=data_construct/data/data_ai_progressive \
    split=forget_10
```

### Step 2: Unlearning

Train the model to forget targeted knowledge via intervention-based distillation:

```bash
CUDA_VISIBLE_DEVICES=0 python forget.py --config-name=forget_ai.yaml \
    data_path=data_construct/data/data_ai_progressive \
    split=forget_10 \
    lr=1e-4 \
    batch_size=8 \
    num_epochs=30 \
    teacher.N=3 \
    LoRA.r=32 \
    overwrite_dir=true \
    hydra.run.dir=.
```

The trained LoRA adapter is saved to `results/mistralai/Mistral-7B-Instruct-v0.3/intervention/{num_epochs}_{lr}_{split}/`.


### Step 3: Relearning

After unlearning, recover the forgotten knowledge through one of two methods.

**Option A : Standard fine-tuning:**

```bash
python relearn.py --config-name=relearn_ai \
    model_path=results/mistralai/Mistral-7B-Instruct-v0.3/intervention/30_0.0001_forget_10 \
    split=forget_10 \
    num_epochs=10
```

**Option B : LLM-guided multi-turn teaching (DeepSeek):**

```bash
python relearn_by_llm.py --config-name=relearn_ai_llm \
    model_path=results/mistralai/Mistral-7B-Instruct-v0.3/intervention/30_0.0001_forget_10 \
    split=forget_10
```

The LLM-guided approach uses a three-party interactive loop: a Coach Agent poses questions, the Teachable Agent (unlearned model) answers, and a Judge Agent evaluates. If incorrect, the model is immediately fine-tuned via LoRA before the next round (up to 4 rounds per question).

<p align="center">
  <img src="pic/pic1.png" width="70%">
</p>

### Step 4: Evaluation

Evaluate the model (either after unlearning or after relearning) on a held-out MCQ test set:

```bash
python evolution/eval_mistral_questions_test.py \
    --forget_adapter_dir ./results/mistralai/Mistral-7B-Instruct-v0.3/intervention/30_0.0001_forget_10
```

This compares the original base model and the target model on multiple-choice accuracy, and saves results to `eval_results.json` inside the adapter directory.

## Project Structure

```
.
├── forget.py                 # Unlearning training script
├── teacher.py                # Teacher distribution generation
├── relearn.py                # Standard relearning (LoRA fine-tuning)
├── relearn_by_llm.py         # LLM-guided relearning (DeepSeek)
├── eval_accuracy.py          # Accuracy evaluation on dataset splits
├── evaluate_util.py          # Evaluation utilities and metrics
├── data_module.py            # Dataset classes and data processing
├── dataloader.py             # Custom Trainer and data collators
├── utils.py                  # Utility functions
├── config/
│   ├── forget_ai.yaml        # Unlearning configuration
│   ├── relearn_ai.yaml       # Standard relearning configuration
│   ├── relearn_ai_llm.yaml   # LLM-guided relearning configuration
│   ├── eval_accuracy.yaml    # Accuracy evaluation configuration
│   └── model_config.yaml     # Model-specific prompt templates
├── evolution/
│   ├── eval_mistral_questions_test.py  # MCQ accuracy evaluation
│   └── eval_f1.py                      # F1-score evaluation
├── data_construct/
│   ├── data/data_ai_progressive/       # Progressive forget splits (10%-50%)
│   └── raw_data/                       # Original question bank (Excel)
└── chat_analysis/
    ├── chat_learning_loop_2.py         # 3-party interactive learning system
    └── prompt/                         # Prompt templates for DeepSeek agents
```

## Citation

If you find this work useful in your research or applications, please kindly cite:

```bibtex
@misc{song2026simulatingnovicestudentsusing,
      title={Simulating Novice Students Using Machine Unlearning and Relearning in Large Language Models}, 
      author={Jiajia Song and Zhihan Guo and Jionghao Lin},
      year={2026},
      eprint={2603.26142},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2603.26142}, 
}
```

## License

This project is licensed under the MIT License.
