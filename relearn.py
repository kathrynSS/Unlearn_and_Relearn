"""
Relearning script - retrains forgotten data on unlearned model.

Usage:
    python relearn.py --config-name=relearn_ai.yaml
"""

import os

LOCAL_RESOURCES_DIR = "/root/autodl-tmp/local_resources"
os.makedirs(LOCAL_RESOURCES_DIR, exist_ok=True)

HF_LOCAL_DIR = os.path.join(LOCAL_RESOURCES_DIR, "huggingface_models")
os.makedirs(HF_LOCAL_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_LOCAL_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_LOCAL_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_LOCAL_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_LOCAL_DIR, "datasets"))
TORCH_HUB_DIR = os.path.join(LOCAL_RESOURCES_DIR, "torch_hub")
os.makedirs(TORCH_HUB_DIR, exist_ok=True)
os.environ.setdefault("TORCH_HOME", TORCH_HUB_DIR)

XDG_CACHE_DIR = os.path.join(LOCAL_RESOURCES_DIR, "cache")
os.makedirs(XDG_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_DIR)

WANDB_DIR = os.path.join(LOCAL_RESOURCES_DIR, "wandb_logs")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ.setdefault("WANDB_DIR", WANDB_DIR)
os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(LOCAL_RESOURCES_DIR, "wandb_cache"))

from data_module import TextDatasetQA, custom_data_collator, load_dataset_auto
from dataloader import LossLoggingCallback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, Trainer, TrainingArguments

import hydra
import transformers
import shutil
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
import inspect


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class RelearnDataset(torch.utils.data.Dataset):
    """Relearning dataset - uses forget data for standard SFT."""
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget_10"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        
        self.data = load_dataset_auto(data_path, split)
        print(f'=' * 20 + f"Loaded {len(self.data)} samples from {split}" + '=' * 20)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        question_start_token = self.model_configs['question_start_tag']
        question_end_token = self.model_configs['question_end_tag']
        answer_token = self.model_configs['answer_tag']
        
        full_text = question_start_token + question + question_end_token + answer_token + answer
        
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        
        pad_length = self.max_length - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [self.tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
        
        if len(encoded.input_ids) == self.max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [self.tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        
        new_question = question_start_token + question + question_end_token
        num_question_tokens = len(self.tokenizer.encode(new_question, add_special_tokens=True))
        for i in range(num_question_tokens):
            label[i] = -100
        
        return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)


def relearn_data_collator(samples):
    """Data collator for relearning batches."""
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }


@hydra.main(version_base=None, config_path="config", config_name="relearn_ai")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")
    
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    set_seed(cfg.seed)
    print(f"seed: {cfg.seed}")
    
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["DEEPSPEED_DISABLE_MPI"] = "1"
    
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    save_dir_path = Path(cfg.save_dir)
    if save_dir_path.exists() and any(save_dir_path.iterdir()):
        print(f"Save dir already exists: {save_dir_path}")
        if not cfg.overwrite_dir:
            print("overwrite_dir is False, exiting to avoid overwriting existing results.")
            exit()
        else:
            print("overwrite_dir is True, removing existing save dir before training...")
            shutil.rmtree(save_dir_path)
    
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    if local_rank == 0:
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    max_length = 512
    print('=' * 20 + f"Max length: {max_length}" + '=' * 20)
    
    train_dataset = RelearnDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=max_length,
        split=cfg.split
    )
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
    max_steps = int(cfg.num_epochs * len(train_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    
    print("######################")
    print(f"Relearning model from: {cfg.model_path}")
    print(f"Saving to: {cfg.save_dir}")
    print("######################")
    
    print(f"batch_size per device: {batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"max_steps: {max_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Weight decay: {cfg.weight_decay}")
    
    training_kwargs = dict(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, max_steps // 10),
        report_to=["wandb"],
        run_name=os.path.basename(cfg.save_dir),
        save_strategy='no',
        save_only_model=True,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=f"{cfg.save_dir}/logs",
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
    )
    
    sig = inspect.signature(transformers.TrainingArguments.__init__)
    valid_keys = {k for k in sig.parameters.keys() if k not in {"self"}}
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in valid_keys}
    
    training_args = transformers.TrainingArguments(**filtered_kwargs)
    
    print('=' * 20 + f"Loading forgotten model from {cfg.model_path}" + '=' * 20)
    
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model path not found: {cfg.model_path}")
    
    config = AutoConfig.from_pretrained(model_id)
    
    if os.path.exists(os.path.join(cfg.model_path, "adapter_config.json")):
        print("Detected LoRA adapter, loading base model + adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, cfg.model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    
    use_full_finetune = cfg.get("full_finetune", False) or cfg.LoRA.r == 0
    
    if use_full_finetune:
        print("=" * 50)
        print("Using FULL FINE-TUNING mode (no LoRA)")
        print("=" * 50)
        for param in model.parameters():
            param.requires_grad = True
        print_trainable_parameters(model)
    else:
        print("=" * 50)
        print(f"Using LoRA mode with r={cfg.LoRA.r}")
        print("=" * 50)
        trainable_modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            target_modules=trainable_modules,
            lora_dropout=cfg.LoRA.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        print(f"LoRA target modules: {trainable_modules}")
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=relearn_data_collator,
        callbacks=[LossLoggingCallback()],
    )
    
    model.config.use_cache = False
    
    print("=" * 50)
    print("Starting relearning training...")
    print("=" * 50)
    trainer.train()
    
    trainer.save_model(cfg.save_dir)
    print(f"Model saved to {cfg.save_dir}")
    
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()

