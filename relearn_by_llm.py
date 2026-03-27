"""
Relearning script using ChatGPT teaching.

Pipeline (same as chat_learning_loop.py):
1. For each question, conduct multi-round dialogue (up to 4 rounds)
2. Each round: Teacher asks -> Student answers -> Judge correctness -> Train if wrong -> Teacher explains
3. Uses the same simple_relearn method as chat_learning_loop.py

Usage:
    python relearn_by_llm.py --config-name=relearn_ai
"""

import os

# ============= Local resource paths (must be set before other imports) =============
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
# =============================================================================

from data_module import load_dataset_auto
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, Trainer, TrainingArguments

import hydra
import transformers
import shutil
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
import json
import warnings
import logging
from openai import OpenAI
from datetime import datetime
import re

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

# ============= Configuration =============
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
MAX_ROUNDS = 4


# ============= ChatGPT Agent =============
class ChatGPTAgent:
    """ChatGPT Agent for evaluating answers and providing feedback."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=(base_url or OPENAI_BASE_URL))
        self.model = model
        self.request_timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "60"))
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
        self.retry_backoff_s = float(os.environ.get("OPENAI_RETRY_BACKOFF_S", "2"))
    
    def _chat_complete(self, system_prompt: str, user_prompt: str, max_completion_tokens: int) -> str:
        """Chat completion with retry/backoff."""
        import time
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=max_completion_tokens,
                    timeout=self.request_timeout_s,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
        raise last_err if last_err is not None else RuntimeError("OpenAI request failed")
    
    def judge_answer(self, question: str, options: dict, model_answer: str, 
                     model_explanation: str, correct_answer: str) -> dict:
        """Judge whether the answer is correct and provide feedback."""
        
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        system_prompt = """You are an expert teacher. Evaluate the student's answer and provide feedback.
Your response should be in JSON format with three fields:
{
  "is_correct": true/false,
  "feedback": "brief feedback message",
  "explanation": "detailed explanation of the correct answer"
}"""
        
        user_prompt = f"""Question: {question}

Options:
{options_text}

Student's answer: {model_answer}
Student's explanation: {model_explanation}
Correct answer: {correct_answer}

Please evaluate if the student is correct, and provide a clear explanation."""

        try:
            content = self._chat_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=1000,
            )
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            is_correct = model_answer.strip().upper() == correct_answer.strip().upper()
            correct_text = options.get(correct_answer.strip().upper(), "")
            if not is_correct:
                fallback_expl = f"The correct answer is {correct_answer}."
                if correct_text:
                    fallback_expl += f" {correct_text}"
            else:
                fallback_expl = ""
            return {
                "is_correct": is_correct,
                "feedback": "Correct." if is_correct else f"Incorrect. The correct answer is {correct_answer}.",
                "explanation": fallback_expl
            }


# ============= Dataset classes =============
class ChatFeedbackDataset(torch.utils.data.Dataset):
    """Dataset for storing ChatGPT feedback samples."""
    
    def __init__(self, questions: list, answers: list, explanations: list, 
                 options_list: list, tokenizer, model_configs: dict, 
                 max_length: int = 256):
        self.questions = questions
        self.answers = answers
        self.explanations = explanations
        self.options_list = options_list
        self.tokenizer = tokenizer
        self.model_configs = model_configs
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        explanation = self.explanations[idx]
        options = self.options_list[idx]
        
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        question_start_token = self.model_configs['question_start_tag']
        question_end_token = self.model_configs['question_end_tag']
        answer_token = self.model_configs['answer_tag']
        
        full_question = f"Question: {question}\nAnswer:"
        
        answer_text = f"{answer}"
        if explanation:
            answer_text += f". {explanation[:100]}"
        
        question_part = question_start_token + full_question + question_end_token + answer_token
        full_text = question_part + answer_text
        
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        
        pad_length = self.max_length - len(encoded.input_ids)
        if pad_length < 0:
            pad_length = 0
            encoded['input_ids'] = encoded['input_ids'][:self.max_length]
            encoded['attention_mask'] = encoded['attention_mask'][:self.max_length]
        
        pad_input_ids = encoded['input_ids'] + [self.tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
        
        # Labels: only compute loss on answer part
        if len(encoded['input_ids']) >= self.max_length:
            label = list(encoded['input_ids'])
        else:
            label = list(encoded['input_ids']) + [self.tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        
        question_tokens = self.tokenizer.encode(question_part, add_special_tokens=True)
        num_question_tokens = len(question_tokens)
        for i in range(min(num_question_tokens, len(label))):
            label[i] = -100
        
        return torch.tensor(pad_input_ids[:self.max_length]), torch.tensor(label[:self.max_length]), torch.tensor(pad_attention_mask[:self.max_length])


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


# ============= Model wrapper =============
class RelearnModel:
    """Trainable relearning model wrapper."""
    
    def __init__(self, model_path: str, base_model_id: str, model_configs: dict, 
                 tokenizer, lora_rank: int = 32):
        self.model_path = model_path
        self.base_model_id = base_model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_configs = model_configs
        self.tokenizer = tokenizer
        self.lora_rank = lora_rank
        
        print(f"Loading model from {model_path}...")
        
        config = AutoConfig.from_pretrained(base_model_id)
        
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
            self.is_lora = True
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path if os.path.exists(os.path.join(model_path, "config.json")) else base_model_id,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(base_model, lora_config)
            self.is_lora = True
        
        for name, param in self.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        try:
            embed_device = self.model.get_input_embeddings().weight.device
            self.device = embed_device
        except Exception:
            self.device = next(self.model.parameters()).device
        
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def answer_question(self, question: str, options: dict) -> tuple:
        """Generate model's answer to a question."""
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        prompt = f"Question: {question}\n\nOptions:\n{options_text}\n\nAnswer with the option letter and brief explanation:"
        
        start_tag = self.model_configs.get('question_start_tag', '')
        end_tag = self.model_configs.get('question_end_tag', '')
        if start_tag or end_tag:
            prompt = f"{start_tag}{prompt}{end_tag}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = 2
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    min_new_tokens=10,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=eos_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
            except Exception:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=150,
                    min_new_tokens=10,
                    do_sample=False,
                    pad_token_id=eos_token_id,
                )
        
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        answer, explanation = self._parse_response(response)
        
        return answer, explanation
    
    def _parse_response(self, response: str) -> tuple:
        """Parse model response into answer letter and explanation."""
        lines = response.strip().split('\n')
        answer = "A"
        explanation = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if any(skip in line.lower() for skip in ['question:', 'options:', 'previous', 'reply with']):
                continue
            
            match = re.match(r'^([A-Da-d])\s*[.\):]\s*(.*)$', line)
            if match:
                answer = match.group(1).upper()
                explanation = match.group(2).strip()
                break
            
            match = re.match(r'^([A-Da-d])$', line)
            if match:
                answer = match.group(1).upper()
                break
        
        if explanation and 'Question:' in explanation:
            explanation = explanation.split('Question:')[0].strip()
        
        if not explanation:
            explanation = "(No explanation)"
        
        return answer, explanation
    
    def simple_relearn(self, question: str, correct_answer: str, explanation: str,
                       options: dict, learning_rate: float = 1e-4, num_epochs: int = 1,
                       output_dir: str = "./temp_relearn"):
        """Fine-tune the model on a single question using ChatGPT feedback."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        train_dataset = ChatFeedbackDataset(
            questions=[question],
            answers=[correct_answer],
            explanations=[explanation],
            options_list=[options],
            tokenizer=self.tokenizer,
            model_configs=self.model_configs,
            max_length=256
        )
        
        safe_lr = min(learning_rate, 3e-5)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=safe_lr,
            warmup_steps=0,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            max_grad_norm=0.5,
            disable_tqdm=True,
            weight_decay=0.01,
            log_level="error",
            log_level_replica="error",
        )
        
        self.model.train()
        
        if self.is_lora:
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                args=training_args,
                data_collator=relearn_data_collator,
            )
        
        self.model.config.use_cache = False
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_result = trainer.train()
                
                final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None
                if final_loss is not None and (final_loss > 10.0 or not torch.isfinite(torch.tensor(final_loss))):
                    import sys
                    print(f"Warning: Training loss abnormal ({final_loss:.2f})", file=sys.stderr)
                    
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        finally:
            self.model.config.use_cache = True
            self.model.eval()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============= Data parsing =============
def parse_question_with_options(item: dict) -> tuple:
    """Parse question text and extract options."""
    question = item.get("question", "")
    answer_raw = item.get("answer", "")
    title = item.get("title", "")
    
    question_clean = question.strip()
    
    options = {}
    pattern_in_question = re.findall(r'([A-D])[.\)]\s*([^\n]+)', question)
    
    if pattern_in_question:
        for letter, text in pattern_in_question:
            options[letter] = text.strip()
        question_clean = re.split(r'\n\s*[A-D][.\)]', question)[0].strip()
        
        answer_match = re.match(r'^([A-D])', answer_raw.strip())
        if answer_match:
            correct_answer = answer_match.group(1)
        else:
            correct_answer = "A"
    else:
        answer_match = re.match(r'^([A-D])[.\)]\s*(.+)', answer_raw.strip())
        if answer_match:
            correct_letter = answer_match.group(1)
            correct_text = answer_match.group(2).strip()
            
            all_letters = ['A', 'B', 'C', 'D']
            for letter in all_letters:
                if letter == correct_letter:
                    options[letter] = correct_text
                else:
                    options[letter] = f"[Other option {letter}]"
            correct_answer = correct_letter
        else:
            options = {
                "A": answer_raw.strip() if answer_raw else "[No answer provided]",
                "B": "[Incorrect option B]",
                "C": "[Incorrect option C]",
                "D": "[Incorrect option D]"
            }
            correct_answer = "A"
    
    return question_clean, options, correct_answer, title


# ============= Main =============
@hydra.main(version_base=None, config_path="config", config_name="relearn_ai_llm")
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
    
    print("=" * 80)
    print("Relearning with ChatGPT Teaching")
    print("=" * 80)
    print(f"Model: {cfg.model_path}")
    print(f"Split: {cfg.split}")
    print(f"Pipeline: Teacher asks → Student answers → Judge → Train if wrong → Explain")
    print(f"Max rounds per question: {MAX_ROUNDS}")
    print("=" * 80)
    
    if not OPENAI_API_KEY:
        raise RuntimeError('Missing OpenAI API key. Please set env "OPENAI_API_KEY".')
    agent = ChatGPTAgent(OPENAI_API_KEY)
    
    dataset = load_dataset_auto(cfg.data_path, cfg.split)
    num_samples = getattr(cfg, 'num_samples', None)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} samples from {cfg.split}")
    
    lora_rank = cfg.LoRA.r if cfg.LoRA.r > 0 else 32
    relearn_model = RelearnModel(
        model_path=cfg.model_path,
        base_model_id=model_id,
        model_configs=model_cfg,
        tokenizer=tokenizer,
        lora_rank=lora_rank
    )
    
    relearn_lr = getattr(cfg, 'relearn_lr', cfg.lr)
    relearn_epochs = getattr(cfg, 'relearn_epochs', 2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir_path / f"relearn_log_{timestamp}.txt"
    results_file = save_dir_path / f"relearn_results_{timestamp}.json"
    
    results = []
    
    print("\n" + "=" * 80)
    print("Starting relearning process...")
    print("=" * 80)
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"Relearning with ChatGPT Teaching\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {cfg.model_path}\n")
        f.write(f"Split: {cfg.split}\n")
        f.write(f"Number of Questions: {len(dataset)}\n")
        f.write(f"=" * 80 + "\n\n")
        
        for q_idx, item in enumerate(dataset):
            question, options, correct_answer, title = parse_question_with_options(item)
            
            print(f"\n[{q_idx+1}/{len(dataset)}] {title}")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"Question {q_idx + 1}/{len(dataset)}\n")
            f.write(f"Topic: {title}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Question: {question}\n")
            options_text_log = "\n".join([f"   {k}. {v}" for k, v in options.items()])
            f.write(f"Options:\n{options_text_log}\n")
            f.write(f"Correct Answer: {correct_answer}\n\n")
            
            question_result = {
                "question_id": q_idx + 1,
                "title": title,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "attempts": []
            }
            
            for round_num in range(1, MAX_ROUNDS + 1):
                f.write(f"\n--- Round {round_num} ---\n")
                
                model_answer, model_explanation = relearn_model.answer_question(question, options)
                
                f.write(f"Student answer: {model_answer}. {model_explanation}\n")
                
                judgment = agent.judge_answer(
                    question, options, model_answer, 
                    model_explanation, correct_answer
                )
                
                is_correct = judgment.get("is_correct", False)
                feedback = judgment.get("feedback", "")
                explanation = judgment.get("explanation", "")
                
                attempt = {
                    "round": round_num,
                    "answer": model_answer,
                    "explanation": model_explanation,
                    "is_correct": is_correct,
                    "feedback": feedback,
                    "teacher_explanation": explanation
                }
                question_result["attempts"].append(attempt)
                
                if is_correct:
                    f.write(f"Teacher: {feedback}\n")
                    print(f"  ✓ Round {round_num}: Correct!")
                    
                    question_result["final_correct"] = True
                    question_result["rounds_needed"] = round_num
                    break
                else:
                    f.write(f"Teacher: {feedback}\n")
                    if explanation:
                        f.write(f"Explanation: {explanation}\n")
                    
                    print(f"  ✗ Round {round_num}: Wrong (expected {correct_answer}), training...")
                    
                    if round_num < MAX_ROUNDS:
                        trainer_out = save_dir_path / f"temp_q{q_idx+1}_round{round_num}"
                        os.makedirs(trainer_out, exist_ok=True)
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                relearn_model.simple_relearn(
                                    question=question,
                                    correct_answer=correct_answer,
                                    explanation=explanation,
                                    options=options,
                                    learning_rate=relearn_lr,
                                    num_epochs=relearn_epochs,
                                    output_dir=str(trainer_out),
                                )
                        except Exception as e:
                            f.write(f"Training failed: {e}\n")
            
            else:
                print(f"  ✗ Failed after {MAX_ROUNDS} rounds")
                question_result["final_correct"] = False
                question_result["rounds_needed"] = MAX_ROUNDS
            
            results.append(question_result)
            f.write(f"\n{'='*60}\n")
        
        total_correct = sum(1 for r in results if r.get("final_correct", False))
        avg_rounds = sum(r.get("rounds_needed", MAX_ROUNDS) for r in results) / len(results)
        
        summary = f"\n[Summary] {total_correct}/{len(results)} correct ({total_correct/len(results)*100:.1f}%), avg {avg_rounds:.1f} rounds"
        print(summary)
        f.write(f"\n{'='*40}\n{summary}\n")
    
    if local_rank == 0:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to {results_file}")
        print(f"Log saved to {log_file}")
    
    print("\nSaving final model...")
    if hasattr(relearn_model.model, 'save_pretrained'):
        relearn_model.model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"Model saved to {cfg.save_dir}")
    
    print("\n" + "=" * 80)
    print("Relearning completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
