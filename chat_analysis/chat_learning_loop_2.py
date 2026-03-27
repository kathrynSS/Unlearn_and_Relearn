"""
Chat Learning Loop — three-party interactive learning system.

Roles:
- Student (ForgetModel): answers questions and provides explanations.
- Scorer (ScoringAgent): objectively evaluates answer and explanation correctness.
- Teacher (ChatGPT Agent): provides teaching feedback based on scoring results.

Scoring:
- Answer correct = 1, wrong = 0; Explanation correct = 1, wrong/missing = 0.
- Total = (answer_score + explanation_score) / 2. Must reach 1.0 to pass.

Mode 1 — Single-question (default):
  Repeat ask/score/teach/fine-tune per question until perfect or MAX_ROUNDS.

Mode 2 — Multi-round (--multi_round):
  Iterate through different questions under one knowledge point, teaching each
  until mastered before moving on.

Usage:
    python chat_learning_loop_2.py --model_path ./30_0.0001_forget_50 --num_questions 5
    python chat_learning_loop_2.py --multi_round --num_rounds 30 --model_path ./30_0.0001_forget_50
    python chat_learning_loop_2.py --multi_round --knowledge_point "Closures" --model_path ./30_0.0001_forget_50
    python chat_learning_loop_2.py --demo --multi_round --num_rounds 10
"""

import os
import sys

# ============= Local resources path config (must be set before importing heavy libs) =============
# NOTE: Path is used in the fallback branch below, so import it before the try/except.
from pathlib import Path

# Prefer a writable path. On some environments (e.g., local macOS), /root is not writable.
_DEFAULT_LOCAL_RESOURCES_DIR = "/root/autodl-tmp/local_resources"
LOCAL_RESOURCES_DIR = os.environ.get("LOCAL_RESOURCES_DIR", _DEFAULT_LOCAL_RESOURCES_DIR)
try:
    os.makedirs(LOCAL_RESOURCES_DIR, exist_ok=True)
except PermissionError:
    _SCRIPT_DIR_FALLBACK = Path(__file__).parent.resolve()
    LOCAL_RESOURCES_DIR = str((_SCRIPT_DIR_FALLBACK / "local_resources").resolve())
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
# =============================================================================

import json
import torch
import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
import datasets

import transformers
transformers.logging.set_verbosity_error()

# ============= Configuration =============
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
MAX_ROUNDS = 4  # Max learning rounds per question
DEFAULT_CONVERSATION_ROUNDS = 30  # Default conversation rounds per knowledge point
_SCRIPT_DIR = Path(__file__).parent.resolve()
PROMPT_DIR = _SCRIPT_DIR /  "prompt"


def load_prompt(filename: str) -> str:
    """Load prompt template from file."""
    prompt_path = PROMPT_DIR / filename
    prompt_path = prompt_path.resolve()
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        script_dir = Path(__file__).parent.resolve()
        expected_prompt_dir = script_dir / "chat_histories" / "prompt"
        error_msg = (
            f"Prompt file not found: {prompt_path}\n"
            f"Script directory: {script_dir}\n"
            f"Expected prompt directory: {expected_prompt_dir}\n"
            f"Current PROMPT_DIR: {PROMPT_DIR.resolve()}\n"
            f"Please ensure the prompt files are in: {expected_prompt_dir}"
        )
        raise FileNotFoundError(error_msg)



class ScoringAgent:
    """Scoring agent that objectively evaluates learning performance."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=(base_url or OPENAI_BASE_URL))
        self.model = model
        self.request_timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "60"))
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
        self.retry_backoff_s = float(os.environ.get("OPENAI_RETRY_BACKOFF_S", "2"))
    
    def _chat_complete(self, system_prompt: str, user_prompt: str, max_completion_tokens: int) -> str:
        """Chat completion with retry/backoff"""
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
    
    def score_conversation_round(self, question: str, options: dict, correct_answer: str,
                                  student_answer: str, student_explanation: str,
                                  teacher_feedback: str, teacher_explanation: str,
                                  round_num: int, is_correct: bool) -> dict:
        """Score a single conversation round (answer + explanation)."""
        answer_score = 1 if is_correct else 0
        
        explanation_score = 0
        explanation_quality = "No explanation"
        
        if not student_explanation or student_explanation.strip() in ["", "(No explanation)", "(No clear explanation provided)"]:
            explanation_score = 0
            explanation_quality = "No explanation provided"
        else:
            system_prompt = """You are an objective evaluator. Judge whether the student's explanation demonstrates correct reasoning, regardless of whether they selected the correct answer.

Focus on:
- Is the reasoning logically sound?
- Does the explanation show understanding of the key concept?
- Is the explanation relevant to the question?

Return ONLY "1" if the explanation shows correct reasoning, or "0" if the reasoning is flawed or irrelevant."""

            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            
            user_prompt = f"""Question: {question}

Options:
{options_text}

Correct Answer: {correct_answer}

Student's Answer: {student_answer}
Student's Explanation: {student_explanation}

Evaluate the REASONING quality (not the answer choice). Reply with only "1" (correct reasoning) or "0" (incorrect reasoning)."""

            try:
                content = self._chat_complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_completion_tokens=50,
                )
                
                # Extract score
                content_clean = content.strip().lower()
                if "1" in content_clean[:5]:
                    explanation_score = 1
                    explanation_quality = "Correct reasoning"
                else:
                    explanation_score = 0
                    explanation_quality = "Incorrect reasoning"
                    
            except Exception as e:
                if is_correct and len(student_explanation) > 10:
                    explanation_score = 1
                    explanation_quality = "Likely correct (auto-scored)"
                else:
                    explanation_score = 0
                    explanation_quality = "Unable to verify"
        
        total_score = (answer_score + explanation_score) / 2.0
        
        return {
            "answer_score": answer_score,
            "explanation_score": explanation_score,
            "total_score": total_score,
            "explanation_quality": explanation_quality
        }
    
    def score_full_conversation(self, question: str, options: dict, correct_answer: str,
                                 conversation_rounds: list, final_correct: bool) -> dict:
        """Evaluate overall learning effectiveness of a full conversation."""
        num_rounds = len(conversation_rounds)
        
        system_prompt = """You are an educational assessment expert. Evaluate the overall learning outcome of a multi-round teacher-student dialogue.

Consider:
1. Final Mastery (0-10): Did the student achieve understanding?
2. Learning Efficiency (0-10): How quickly did the student learn?
3. Overall Effectiveness (0-10): Quality of the entire learning process.

Provide assessment in JSON format."""

        rounds_summary = "\n".join([
            f"Round {i+1}: Student answer={r.get('student_answer', 'N/A')}, Correct={r.get('is_correct', False)}"
            for i, r in enumerate(conversation_rounds)
        ])
        
        user_prompt = f"""Question: {question}

Conversation Summary:
{rounds_summary}

Total Rounds: {num_rounds}
Final Outcome: {"Correct" if final_correct else "Incorrect"}

Please evaluate the overall learning outcome in JSON format:
{{
    "final_mastery_score": <0-10>,
    "learning_efficiency": <0-10>,
    "overall_effectiveness": <0-10>,
    "summary": "<brief summary>"
}}"""

        try:
            content = self._chat_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=400,
            )
            
            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            # Fallback scoring
            mastery = 9 if final_correct else (4 if num_rounds <= 2 else 2)
            efficiency = max(0, 10 - num_rounds * 2)
            overall = (mastery + efficiency) // 2
            
            return {
                "final_mastery_score": mastery,
                "learning_efficiency": efficiency,
                "overall_effectiveness": overall,
                "summary": f"Completed in {num_rounds} rounds, {'successful' if final_correct else 'needs more work'}."
            }


class ChatGPTAgent:
    """ChatGPT Agent for evaluating answers and providing feedback"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str | None = None):
        # Keep the call format consistent with scripts/test_nus_llm_api.py:
        # client = OpenAI(api_key=..., base_url=...)
        self.client = OpenAI(api_key=api_key, base_url=(base_url or OPENAI_BASE_URL))
        self.model = model  # deepseek-chat for better reasoning and explanation
        # Robustness for timeouts/transient errors
        self.request_timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S", "60"))
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
        self.retry_backoff_s = float(os.environ.get("OPENAI_RETRY_BACKOFF_S", "2"))
        # Load prompts from files
        self.judge_system_prompt = load_prompt("judge_answer_system.txt")
        self.judge_user_prompt = load_prompt("judge_answer_user.txt")
        self.teaching_system_prompt = load_prompt("teaching_feedback_system.txt")
        self.teaching_user_prompt = load_prompt("teaching_feedback_user.txt")
        self.rewrite_system_prompt = load_prompt("rewrite_student_response_system.txt")
        self.rewrite_user_prompt = load_prompt("rewrite_student_response_user.txt")

    def _chat_complete(self, system_prompt: str, user_prompt: str, max_completion_tokens: int) -> str:
        """Chat completion with retry/backoff. Returns assistant content or raises on final failure."""
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
                # backoff then retry
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
        # give up
        raise last_err if last_err is not None else RuntimeError("OpenAI request failed")
    
    def judge_answer(self, question: str, options: dict, model_answer: str, 
                     model_explanation: str, correct_answer: str) -> dict:
        """
        Judge if the model's answer is correct and provide feedback
        
        Returns:
            dict: {
                "is_correct": bool,
                "feedback": str,
                "explanation": str
            }
        """
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        prompt = self.judge_user_prompt.format(
            question=question,
            options_text=options_text,
            model_answer=model_answer,
            model_explanation=model_explanation,
            correct_answer=correct_answer
        )

        try:
            content = self._chat_complete(
                system_prompt=self.judge_system_prompt,
                user_prompt=prompt,
                max_completion_tokens=300,
            )
            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            return result
            
        except Exception as e:
            # Silent fallback: keep dialogue clean even on timeouts/errors
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
    
    def provide_teaching_feedback(self, question: str, correct_answer: str, 
                                   options: dict, previous_attempts: list) -> str:
        """Provide teaching feedback to help the model learn"""
        
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        attempts_text = "\n".join([f"- Attempt {i+1}: {a['answer']} ({a['explanation'][:50]}...)" 
                                   for i, a in enumerate(previous_attempts)])
        
        prompt = self.teaching_user_prompt.format(
            question=question,
            options_text=options_text,
            correct_answer=correct_answer,
            attempts_text=attempts_text
        )

        try:
            return self._chat_complete(
                system_prompt=self.teaching_system_prompt,
                user_prompt=prompt,
                max_completion_tokens=400,
            )
        except Exception as e:
            # Silent fallback: keep dialogue clean
            return "Let's slow down and review the key concept step by step. " \
                   f"The correct answer is {correct_answer}. " \
                   f"{options.get(correct_answer.strip().upper(), '')}".strip()
    
    def rewrite_student_answer(self, answer: str, explanation: str) -> tuple:
        """
        Rewrite student answer to fix grammatical errors while preserving meaning
        
        Args:
            answer: The student's answer choice (e.g., "A", "B", "C", "D")
            explanation: The student's explanation text
        
        Returns:
            tuple: (rewritten_answer, rewritten_explanation)
        """
        prompt = self.rewrite_user_prompt.format(
            answer=answer,
            explanation=explanation
        )
        
        try:
            content = self._chat_complete(
                system_prompt=self.rewrite_system_prompt,
                user_prompt=prompt,
                max_completion_tokens=500,
            )
            
            # Parse the rewritten response
            lines = content.strip().split('\n')
            rewritten_answer = answer  # default to original
            rewritten_explanation = explanation  # default to original
            
            for line in lines:
                line = line.strip()
                if line.startswith("Answer:"):
                    rewritten_answer = line.replace("Answer:", "").strip()
                elif line.startswith("Explanation:"):
                    rewritten_explanation = line.replace("Explanation:", "").strip()
            
            return rewritten_answer, rewritten_explanation
            
        except Exception as e:
            # Silent fallback: return original if rewriting fails
            return answer, explanation
    
    def provide_teaching_explanation(self, question: str, options: dict, correct_answer: str,
                                      student_answer: str, student_explanation: str,
                                      answer_score: int, explanation_score: int) -> tuple:
        """Generate teaching feedback and explanation based on scoring results."""
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
        
        if answer_score == 1 and explanation_score == 1:
            system_prompt = """You are a supportive teacher. The student answered correctly with correct reasoning. 
Provide brief, encouraging feedback (1-2 sentences)."""
            user_prompt = f"""Question: {question}

Options:
{options_text}

Correct Answer: {correct_answer}
Student's Answer: {student_answer}
Student's Explanation: {student_explanation}

Provide brief encouraging feedback."""
            
        elif answer_score == 1 and explanation_score == 0:
            system_prompt = """You are a patient teacher. The student selected the correct answer but their reasoning is flawed or incomplete.

Your task:
1. Acknowledge they got the right answer
2. Explain WHY this answer is correct (focus on the key concept/knowledge point)
3. Point out what's wrong with their reasoning
4. Help them understand the correct reasoning

Be concise but educational (2-4 sentences)."""
            user_prompt = f"""Question: {question}

Options:
{options_text}

Correct Answer: {correct_answer}
Student's Answer: {student_answer} ✓ (correct)
Student's Explanation: {student_explanation} ✗ (reasoning flawed)

Provide teaching feedback that explains the correct concept and reasoning."""
            
        elif answer_score == 0 and explanation_score == 1:
            system_prompt = """You are a patient teacher. The student has correct reasoning but selected the wrong answer (perhaps a careless mistake).

Your task:
1. Point out the answer choice is wrong
2. Acknowledge their reasoning is sound
3. Guide them to apply their correct reasoning to select the right answer

Be concise (2-3 sentences)."""
            user_prompt = f"""Question: {question}

Options:
{options_text}

Correct Answer: {correct_answer}
Student's Answer: {student_answer} ✗ (wrong choice)
Student's Explanation: {student_explanation} ✓ (reasoning correct)

Help them connect their correct reasoning to the right answer."""
            
        else:
            system_prompt = """You are a patient teacher. The student answered incorrectly with flawed reasoning.

Your task:
1. Identify the misconception or gap in their understanding
2. Explain the KEY CONCEPT/KNOWLEDGE POINT this question tests
3. Explain WHY the correct answer is right (step by step if needed)
4. Connect the explanation to the specific question

Be clear and educational but concise (3-5 sentences)."""
            user_prompt = f"""Question: {question}

Options:
{options_text}

Correct Answer: {correct_answer}
Student's Answer: {student_answer} ✗ (wrong)
Student's Explanation: {student_explanation} ✗ (reasoning flawed)

Provide teaching feedback that addresses their misconception and explains the correct concept."""
        
        try:
            content = self._chat_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_completion_tokens=400,
            )
            
            feedback = content.strip()
            explanation = ""
            
            return feedback, explanation
            
        except Exception as e:
            # Fallback to simple template
            if answer_score == 1 and explanation_score == 1:
                return "Excellent! Your answer and reasoning are both correct.", ""
            elif answer_score == 1 and explanation_score == 0:
                return f"Your answer is correct, but your reasoning needs improvement. The correct answer is {correct_answer}. {options.get(correct_answer, '')}", ""
            elif answer_score == 0 and explanation_score == 1:
                return f"Your reasoning is sound, but you selected the wrong answer. The correct answer is {correct_answer}. {options.get(correct_answer, '')}", ""
            else:
                return f"Your answer is incorrect. The correct answer is {correct_answer}. {options.get(correct_answer, '')}", ""


class SimulatedForgetModel:
    """
    Simulated forget model (for demo mode)
    
    Simulates a model that has "forgotten" some knowledge:
    - Initial accuracy is low (simulating forgetting effect)
    - After seeing teacher feedback, accuracy improves (simulating conversation learning)
    """
    
    def __init__(self, initial_accuracy: float = 0.3):
        self.accuracy = initial_accuracy
    
    def answer_question(self, question: str, options: dict, correct_answer: str = None,
                        conversation_history: list = None) -> tuple:
        """
        Answer question - simulates student learning from conversation history
        
        Args:
            conversation_history: Previous conversation history, model improves based on teacher feedback
        """
        # Calculate accuracy based on conversation history
        # Each round of teacher feedback increases accuracy
        if conversation_history and len(conversation_history) > 0:
            # After seeing teacher feedback, accuracy improves significantly
            rounds_learned = len(conversation_history)
            # After round 1: 70%, round 2: 85%, round 3: 95%
            learned_accuracy = min(0.95, 0.5 + rounds_learned * 0.2)
            should_be_correct = random.random() < learned_accuracy
        else:
            # First answer, use initial accuracy
            should_be_correct = random.random() < self.accuracy
        
        if should_be_correct and correct_answer:
            answer = correct_answer
            if conversation_history:
                last_feedback = conversation_history[-1].get('teacher_explanation', '')
                explanation = f"Based on the teacher's explanation, I understand now. {last_feedback[:50]}... So the correct answer is {answer}."
            else:
                explanation = f"I think the answer is {answer}."
        else:
            wrong_options = [k for k in options.keys() if k != correct_answer]
            answer = random.choice(wrong_options) if wrong_options else "A"
            if conversation_history:
                explanation = f"I'm still not sure, let me think again... I guess it's {answer}."
            else:
                explanation = f"I'm not sure, but I guess it's {answer}."
        
        return answer, explanation


class ForgetModel:
    """Forget model wrapper with LoRA fine-tuning."""
    
    def __init__(self, model_path: str, base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 model_family: str = "mistral-7b", lora_rank: int = 32):
        self.model_path = model_path
        self.base_model_id = base_model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_family = model_family
        self.lora_rank = lora_rank
        
        from utils import get_model_identifiers_from_yaml
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        
        # Load prompts from files
        self.answer_prompt_template = load_prompt("model_answer_question.txt")
        self.answer_with_history_template = load_prompt("model_answer_with_history.txt")
        
        # Buffer to accumulate feedback for batch training
        self.feedback_buffer = {
            "questions": [],
            "answers": [],
            "explanations": [],
            "options_list": []
        }
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        config = AutoConfig.from_pretrained(base_model_id)

        # For round-by-round training persistence
        self.is_lora = False
        self._base_model_for_reload = None

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
            self._base_model_for_reload = base_model
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
                lora_alpha=lora_rank * 2,  # alpha = 2 * r
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(base_model, lora_config)
            self.is_lora = True
            self._base_model_for_reload = base_model
        
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

    def save_round_checkpoint(self, save_dir: str):
        """Save current (trained) LoRA weights so the next round can reload and continue training."""
        os.makedirs(save_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(save_dir)

    def reload_from_checkpoint(self, save_dir: str):
        """Reload LoRA weights from a saved checkpoint (currently keeps in-memory state)."""
        self.model.eval()
    
    def answer_question(self, question: str, options: dict, correct_answer: str = None,
                        conversation_history: list = None) -> tuple:
        """
        Answer question and explain the reason
        
        Args:
            question: The question text
            options: Dictionary of options
            correct_answer: The correct answer (optional, for reference)
            conversation_history: List of previous conversation turns, each is a dict with:
                - 'student_answer': str
                - 'student_explanation': str  
                - 'teacher_feedback': str
                - 'teacher_explanation': str
        
        Returns:
            tuple: (answer, explanation)
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        
        # Build prompt (conversation-aware)
        if conversation_history and len(conversation_history) > 0:
            prompt = self._build_conversational_prompt(question, options_text, conversation_history)
        else:
            prompt = self.answer_prompt_template.format(
                question=question,
                options_text=options_text
            )

        # Wrap with model-specific instruction tags if provided (e.g., Mistral uses [INST]...[/INST])
        start_tag = (self.model_configs or {}).get("question_start_tag", "")
        end_tag = (self.model_configs or {}).get("question_end_tag", "")
        if start_tag or end_tag:
            prompt = f"{start_tag}{prompt}{end_tag}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) if self.tokenizer.eos_token else 2
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=200,
                    min_new_tokens=10,
                    do_sample=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=eos_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
            except Exception as e:
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=150,
                    min_new_tokens=10,
                    do_sample=False,
                    pad_token_id=eos_token_id,
                )
        
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer, explanation = self._parse_response_simple(response, full_output)
        
        return answer, explanation
    
    def _build_conversational_prompt(self, question: str, options_text: str, 
                                      conversation_history: list) -> str:
        """
        Build prompt with conversation history so model can see teacher's feedback
        Simulates real teacher-student dialogue
        """
        # Build conversation history text
        history_text = ""
        for i, turn in enumerate(conversation_history):
            round_num = i + 1
            history_text += f"\n--- Round {round_num} ---\n"
            history_text += f"Your answer: {turn['student_answer']}\n"
            history_text += f"Your explanation: {turn['student_explanation']}\n"
            history_text += f"Teacher's feedback: {turn['teacher_feedback']}\n"
            if turn.get('teacher_explanation'):
                history_text += f"Teacher's explanation: {turn['teacher_explanation']}\n"
        
        # Use external prompt template
        prompt = self.answer_with_history_template.format(
            question=question,
            options_text=options_text,
            history_text=history_text
        )
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean model output by removing repetitions and garbled text."""
        import re
        
        # First remove prompt instruction echoes
        response = self._remove_prompt_echoes(response)
        
        lines = response.split('\n')
        cleaned_lines = []
        seen = set()
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                cleaned_lines.append(line)
                seen.add(line_stripped)
        
        response = '\n'.join(cleaned_lines)
        
        # # 截断过长的重复内容
        # if len(response) > 500:
        #     response = response[:500] + "..."
        
        return response.strip()
    
    def _parse_response_simple(self, response: str, full_output: str = None) -> tuple:
        """Parse model response flexibly across multiple output formats."""
        import re
        
        if response.strip().startswith('.'):
            explanation = response.strip()[1:].strip()
            
            answer = "A"
            if full_output:
                match = re.search(r'(?:answer|Answer):\s*([A-Da-d])\s*\.?\s*$', full_output[:full_output.rfind(response)])
                if match:
                    answer = match.group(1).upper()
                else:
                    match = re.search(r'\b([A-Da-d])\s*\.?\s*$', full_output[:full_output.rfind(response)])
                    if match:
                        answer = match.group(1).upper()
            
            if len(explanation) >= 5:
                return answer, explanation
        
        match = re.search(r'([A-Da-d])\s*[.\):]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            explanation = match.group(2).strip()
            
            if explanation:
                for skip in ['Question:', 'Options:', 'Previous', 'Reply with', 'Answer:']:
                    if skip in explanation:
                        explanation = explanation.split(skip)[0].strip()
                
                if len(explanation) >= 5:
                    return answer, explanation
        
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        for i, line in enumerate(lines):
            if any(skip in line.lower() for skip in ['question:', 'options:', 'previous', 'reply with']):
                continue
            
            match = re.match(r'^([A-Da-d])$', line)
            if match:
                answer = match.group(1).upper()
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not re.match(r'^[A-Da-d][\.\):]', next_line) and len(next_line) >= 5:
                        return answer, next_line
                return answer, "(No explanation)"
        
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response.upper():
                pattern = rf'{letter}\s*[.\):]\s*(.+?)(?:\n|$)'
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    explanation = match.group(1).strip()
                    if len(explanation) >= 5:
                        return letter, explanation
                return letter, "(No explanation)"
        
        if len(response.strip()) >= 10:
            return "A", response.strip()
        
        return "A", "(No explanation)"
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response"""
        import re
        
        # Pattern 1: "Answer: X" or "答案：X"
        match = re.search(r'(?:Answer|答案)[：:]\s*([A-Da-d])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Pattern 2: Single letter at the beginning
        match = re.search(r'^([A-Da-d])[.\s]', response)
        if match:
            return match.group(1).upper()
        
        # Pattern 3: Any single letter option in response
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response.upper():
                return letter
        
        return "A"  # Default return A

    def _parse_model_response(self, response: str) -> tuple:
        """
        Parse model output and keep only:
        - answer option letter (A/B/C/D)
        - reasoning (can be single or multi-line, without repeating question/options)
        """
        import re

        answer = self._extract_answer(response)
        reason_lines = []

        lines = [ln.strip() for ln in (response or "").splitlines() if ln.strip()]
        
        # Strategy: Find the first "X. reason" line, then collect subsequent non-echo lines
        found_answer_line = False
        answer_line_idx = -1
        
        for idx, line in enumerate(lines):
            low = line.lower()
            # Skip obvious prompt echoes
            if any(k in low for k in ("question:", "options:", "previous conversation", "reply with", "do not", "output one line")):
                continue

            # Match "A. reason" / "B) reason" / "C: reason"
            m = re.match(r"^([A-Da-d])\s*[\.\)\:]\s*(.+)$", line)
            if m and not found_answer_line:
                letter = m.group(1).upper()
                content = m.group(2).strip()
                
                # Check if this is followed by more "X. ..." lines (indicates option list echo)
                next_lines_are_options = False
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    # Only consider it an option list if next line also starts with option pattern
                    # AND has similar length (indicating it's part of the same list)
                    if re.match(r"^[A-Da-d]\s*[\.\)\:]\s*.{20,}", next_line):
                        next_lines_are_options = True
                
                # If not followed by more options, this is likely the answer
                if not next_lines_are_options:
                    answer = letter
                    reason_lines.append(content)
                    found_answer_line = True
                    answer_line_idx = idx
                    break

            # Match "Answer: A. reason" / "答案：B  reason"
            m = re.match(r"^(?:answer|答案)\s*[\:\：]\s*([A-Da-d])(?:\s*[\.\)\:])?\s*(.*)$", line, flags=re.IGNORECASE)
            if m and not found_answer_line:
                answer = m.group(1).upper()
                content = (m.group(2) or "").strip()
                if content:
                    reason_lines.append(content)
                found_answer_line = True
                answer_line_idx = idx
                break
        
        # Collect additional reasoning from subsequent lines (if any)
        if found_answer_line and answer_line_idx >= 0:
            for idx in range(answer_line_idx + 1, len(lines)):
                line = lines[idx]
                low = line.lower()
                
                # Stop if we hit another option pattern (likely echo)
                if re.match(r"^[A-Da-d]\s*[\.\)\:]\s*.{20,}", line):
                    break
                
                # Stop if we hit prompt echoes
                if any(k in low for k in ("question:", "options:", "previous conversation", "reply with", "do not")):
                    break
                
                # Otherwise, this is part of the explanation
                reason_lines.append(line)
        
        # Combine reason lines
        reason = " ".join(reason_lines).strip()
        
        # Clean up reason: remove trailing garbage
        if reason:
            # Truncate before option echoes
            truncate_match = re.search(r'\s+[A-D]\.\s+[A-Z][a-z]+\s', reason)
            if truncate_match:
                reason = reason[:truncate_match.start()].strip()
            
            # Remove trailing "/options" or similar garbage
            reason = re.sub(r'\s*/options[.\s]*$', '', reason, flags=re.IGNORECASE).strip()
            reason = re.sub(r'\s*\[[A-D]\]\s*$', '', reason).strip()
        
        if not reason:
            reason = "(No explanation)"

        return answer, reason
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation from response - only keep answer option and reasoning"""
        import re
        
        # Strategy: Find the actual answer line in the format "X. reasoning" 
        # where X is A/B/C/D, appearing after all prompt echoes
        
        lines = response.split('\n')
        
        # First pass: Look for answer lines that appear AFTER echoed content
        # Skip lines that look like option definitions (part of the echoed prompt)
        in_options_section = False
        found_instructions = False
        answer_candidates = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            lower = stripped.lower()
            
            # Detect echoed sections
            if lower.startswith('question:') or lower.startswith('options:'):
                in_options_section = 'options:' in lower
                continue
            
            if 'reply with one line' in lower or 'do not repeat' in lower or 'do not quote' in lower:
                found_instructions = True
                continue
            
            # Check if this is an option definition line (A. long text, B. long text, etc.)
            option_match = re.match(r'^([A-D])\.\s+(.+)$', stripped)
            if option_match:
                letter, content = option_match.groups()
                # If this looks like part of an options list (multiple options nearby), skip it
                # But if it appears after instructions, it's likely the answer
                if found_instructions or (len(content) < 100 and not any(
                    re.match(r'^[A-D]\.\s+', lines[j].strip()) 
                    for j in range(max(0, i-3), i) if lines[j].strip()
                )):
                    answer_candidates.append(stripped)
        
        # Return the last candidate (most likely the actual answer after all echoes)
        if answer_candidates:
            return answer_candidates[-1]
        
        # Fallback: Use _remove_prompt_echoes
        cleaned = self._remove_prompt_echoes(response)
        
        # If still has content, return it
        if cleaned and len(cleaned) > 2:
            return cleaned
        
        # Last resort: Try to find any "X. text" pattern
        match = re.search(r'([A-Da-d]\.\s+[^\n]+)', response)
        if match:
            return match.group(1).strip()
        
        return "(No explanation provided)"
    
    def _remove_prompt_echoes(self, text: str) -> str:
        """Remove prompt instruction echoes from model output"""
        import re
        
        # First, try to find the actual answer line (format: "X. reason" where X is A/B/C/D)
        # This should be after all the echoed prompt content
        lines = text.split('\n')
        
        # Look for the answer line pattern at the end of echoed content
        answer_line = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Match patterns like "B. The named groups..." or "A. reason here"
            if re.match(r'^[A-Da-d]\.\s+[^A-D]', stripped):
                # This looks like an answer line, not an option definition
                # Check it's not an option definition (which would be like "A. Some option text" as part of a list)
                if i > 0 or not any(re.match(r'^[A-Da-d]\.\s+', l.strip()) for l in lines[i+1:i+4] if l.strip()):
                    answer_line = stripped
                    break
        
        if answer_line:
            return answer_line
        
        # Common prompt instruction patterns to remove
        patterns_to_remove = [
            r'Question:\s*.*?(?=Options:|$)',  # Remove question text
            r'Options:\s*',  # Remove "Options:" header
            r'^[A-D]\.\s+[^\n]+\n?',  # Remove option lines at start of line
            r'Reply with ONE line only[^.]*\.?',
            r'in this exact format[^.]*\.?',
            r'\[A/B/C/D\]\s*\.?\s*',
            r'<brief reason[^>]*>',
            r'Do not repeat the question[^/]*\.?',
            r'Do not quote these instructions\.?',
            r'Based on the teacher\'s feedback[^.]*\.?',
            r'reconsider and answer again\.?',
            r'what you learned \(paraphrase\)',
            r'Previous conversation:.*?(?=Based on|$)',  # Remove conversation history echo
        ]
        
        cleaned = text
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up multiple spaces and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove leading punctuation after cleaning
        cleaned = re.sub(r'^[\s\.,;:]+', '', cleaned)
        
        return cleaned
    
    def simple_relearn(self, question: str, correct_answer: str, explanation: str,
                       options: dict, learning_rate: float = 1e-4, num_epochs: int = 1,
                       output_dir: str = "./temp_relearn"):
        """
        Relearning using the same method as relearn.py
        Training data is ChatGPT's feedback from the previous round
        
        Args:
            question: The question text
            correct_answer: The correct answer (e.g., "A")
            explanation: ChatGPT's explanation from previous round
            options: Dictionary of options
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
        """
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
        
        # Set to training mode
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
        
        # Disable cache for training
        self.model.config.use_cache = False
        
        try:
            # Train (silent)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_result = trainer.train()
                
                final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None
                if final_loss is not None and (final_loss > 10.0 or not torch.isfinite(torch.tensor(final_loss))):
                    import sys
                    print(f"Warning: Training loss abnormal ({final_loss:.2f}), model may be unstable", file=sys.stderr)
                    
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        finally:
            # Restore evaluation mode
            self.model.config.use_cache = True
            self.model.eval()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def add_feedback_to_buffer(self, question: str, correct_answer: str, 
                                explanation: str, options: dict):
        """Add ChatGPT feedback to buffer for batch training later."""
        self.feedback_buffer["questions"].append(question)
        self.feedback_buffer["answers"].append(correct_answer)
        self.feedback_buffer["explanations"].append(explanation)
        self.feedback_buffer["options_list"].append(options)
    
    def batch_relearn_from_buffer(self, learning_rate: float = 1e-4, num_epochs: int = 3):
        """Batch relearning using all accumulated ChatGPT feedback."""
        if len(self.feedback_buffer["questions"]) == 0:
            return
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        train_dataset = ChatFeedbackDataset(
            questions=self.feedback_buffer["questions"],
            answers=self.feedback_buffer["answers"],
            explanations=self.feedback_buffer["explanations"],
            options_list=self.feedback_buffer["options_list"],
            tokenizer=self.tokenizer,
            model_configs=self.model_configs,
            max_length=256
        )
        
        safe_lr = min(learning_rate, 5e-5)
        training_args = TrainingArguments(
            output_dir="./temp_batch_relearn",
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
            max_grad_norm=0.3,
            disable_tqdm=True,
            weight_decay=0.01,
            log_level="error",
            log_level_replica="error",
        )
        
        # Set to training mode
        self.model.train()
        
        if self.is_lora:
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=relearn_data_collator,
        )
        
        self.model.config.use_cache = False
        
        try:
            train_result = trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        finally:
            # Clear buffer
            self.feedback_buffer = {
                "questions": [],
                "answers": [],
                "explanations": [],
                "options_list": []
            }
            
            # Restore evaluation mode
            self.model.config.use_cache = True
            self.model.eval()
            
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class ChatFeedbackDataset(torch.utils.data.Dataset):
    """Dataset for relearning from ChatGPT feedback."""
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
        
        # Build a SHORT prompt for training so labels won't be fully masked by truncation.
        # (Including all options can easily exceed max_length and cause all labels = -100.)
        full_question = f"Question: {question}\nAnswer:"
        
        answer_text = f"{answer}"
        if explanation:
            answer_text += f". {explanation[:100]}"
        
        question_part = question_start_token + full_question + question_end_token + answer_token
        full_text = question_part + answer_text
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        
        # Padding
        pad_length = self.max_length - len(encoded.input_ids)
        if pad_length < 0:
            pad_length = 0
            encoded['input_ids'] = encoded['input_ids'][:self.max_length]
            encoded['attention_mask'] = encoded['attention_mask'][:self.max_length]
        
        pad_input_ids = encoded['input_ids'] + [self.tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
        
        # Labels: only compute loss on answer part (same as relearn.py)
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
    """Data collator for relearning (same as relearn.py)"""
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }


def load_forget_data(data_path: str, split: str = "forget_50"):
    """Load forget dataset in HuggingFace datasets format."""
    if os.path.exists(data_path):
        dataset_dict = datasets.load_from_disk(data_path)
        if split in dataset_dict:
            return dataset_dict[split]
    raise ValueError(f"Cannot load dataset: {data_path}/{split}")


def group_data_by_knowledge_point(data: list) -> dict:
    """Group data items by their knowledge point."""
    grouped = {}
    for item in data:
        kp = item.get("title", item.get("knowledge_point", "Unknown"))
        if kp not in grouped:
            grouped[kp] = []
        grouped[kp].append(item)
    return grouped


def load_forget_data_from_excel(excel_path: str) -> list:
    """Load forget data from an Excel file (questions_test.xlsx format)."""
    import re
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path)
    
    required_cols = {"question", "options", "answer_letter"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel file must contain columns: {required_cols}, found: {set(df.columns)}")
    
    data = []
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        options_raw = str(row["options"]).strip()
        answer_letter_raw = str(row["answer_letter"]).strip()
        title = str(row.get("question_type", row.get("knowledge_point", row.get("title", f"Question {idx+1}")))).strip()
        
        options = {}
        for opt in options_raw.split("|"):
            opt = opt.strip()
            match = re.match(r'^([A-D])[.\)]\s*(.+)$', opt)
            if match:
                letter, text = match.groups()
                options[letter] = text.strip()
        
        if not options:
            for opt in options_raw.split("\n"):
                opt = opt.strip()
                match = re.match(r'^([A-D])[.\)]\s*(.+)$', opt)
                if match:
                    letter, text = match.groups()
                    options[letter] = text.strip()
        
        answer_match = re.match(r'^([A-D])', answer_letter_raw)
        correct_answer = answer_match.group(1) if answer_match else "A"
        
        data.append({
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "title": title
        })
    
    return data


def parse_question_with_options(item: dict) -> tuple:
    """Parse question text and extract options from various data formats."""
    import re
    
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


def run_multi_round_conversation(
    model_path: str, 
    data_path: str = None, 
    split: str = "forget_50",
    num_rounds: int = 30, 
    knowledge_point: str = None,
    output_dir: str = None, 
    demo: bool = False,
    relearn_mode: str = "train_conversation", 
    relearn_lr: float = 1e-4,
    relearn_epochs: int = 1,
    round_ckpt_root: str = None,
    excel_path: str = None,
    lora_rank: int = 32
):
    """Run multi-round conversation learning loop on one knowledge point."""
    
    # Initialize ChatGPT Agent
    if not OPENAI_API_KEY:
        raise RuntimeError('Missing OpenAI API key. Please set env "OPENAI_API_KEY".')
    agent = ChatGPTAgent(OPENAI_API_KEY)
    
    scoring_agent = ScoringAgent(OPENAI_API_KEY)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if demo:
            forget_model = SimulatedForgetModel(initial_accuracy=0.3)
        else:
            forget_model = ForgetModel(model_path, lora_rank=lora_rank)
    
    # Load data
    data = None
    data_source = ""
    
    if excel_path and os.path.exists(excel_path):
        try:
            data = load_forget_data_from_excel(excel_path)
            data_source = f"excel:{excel_path}"
        except Exception as e:
            if demo:
                raise ValueError(f"Failed to load Excel data from {excel_path}: {e}")
    
    if data is None and data_path:
        try:
            data = load_forget_data(data_path, split)
            data_source = f"datasets:{data_path}/{split}"
        except Exception as e:
            if demo:
                raise ValueError(f"Failed to load dataset from {data_path}/{split}: {e}")
    
    if data is None:
        raise ValueError("No data source available. Please provide --excel_path or --data_path")
    
    grouped_data = group_data_by_knowledge_point(data)
    
    if knowledge_point is None:
        knowledge_point = random.choice(list(grouped_data.keys()))
        print(f"Randomly selected knowledge point: {knowledge_point}")
    
    if knowledge_point not in grouped_data:
        raise ValueError(f"Knowledge point '{knowledge_point}' not found in data. Available: {list(grouped_data.keys())}")
    
    kp_questions = grouped_data[knowledge_point]
    
    actual_rounds = min(num_rounds, len(kp_questions))
    if actual_rounds < num_rounds:
        print(f"\nNote: Requested {num_rounds} rounds, but only {len(kp_questions)} questions available for this knowledge point.")
        print(f"Will conduct {actual_rounds} rounds (one question per round, no repetition).\n")
    
    print(f"\nKnowledge Point: {knowledge_point}")
    print(f"Available questions for this knowledge point: {len(kp_questions)}")
    print(f"Will conduct {actual_rounds} rounds of conversation (no repeated questions)\n")
    print("="*80)
    
    script_dir = Path(__file__).parent.resolve()
    chat_histories_dir = script_dir / "chat_histories"
    chat_histories_dir.mkdir(parents=True, exist_ok=True)
    
    if round_ckpt_root is None:
        round_ckpt_root = str((chat_histories_dir / "round_checkpoints").resolve())
    os.makedirs(round_ckpt_root, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kp_safe_name = "".join(c if c.isalnum() else "_" for c in knowledge_point[:50])
    log_file = chat_histories_dir / f"conversation_{kp_safe_name}_{timestamp}.txt"
    results_file = chat_histories_dir / f"results_{kp_safe_name}_{timestamp}.json"
    
    conversation_history = []
    round_results = []
    
    selected_questions = kp_questions[:actual_rounds]
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Knowledge Point: {knowledge_point}\n")
        f.write(f"Total Rounds: {actual_rounds}\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{'='*80}\n\n")
        
        for question_num in range(1, actual_rounds + 1):
            item = selected_questions[question_num - 1]
            
            if data_source.startswith("excel:"):
                question = item["question"]
                options = item["options"]
                correct_answer = item["correct_answer"]
                title = item["title"]
            else:
                question, options, correct_answer, title = parse_question_with_options(item)
            
            if question_num > 1:
                print()
                f.write("\n")
            
            question_conversation_history = []
            
            for attempt_num in range(1, MAX_ROUNDS + 1):
                if attempt_num == 1:
                    options_inline = " ".join([f"{k}. {v}" for k, v in options.items()])
                    teacher_question = f"Teacher: {question} {options_inline}"
                    print(teacher_question)
                    f.write(f"{teacher_question}\n")
                
                model_answer, model_explanation = forget_model.answer_question(
                    question, options, correct_answer, 
                    conversation_history=question_conversation_history
                )
                
                if model_explanation and model_explanation.strip() and model_explanation.strip() != "(No explanation)":
                    model_answer, model_explanation = agent.rewrite_student_answer(model_answer, model_explanation)
                
                if model_explanation and model_explanation.strip() and model_explanation.strip() != "(No explanation)":
                    student_response = f"Student: {model_answer}. {model_explanation}"
                else:
                    student_response = f"Student: {model_answer}."
                
                print(student_response)
                f.write(f"{student_response}\n")
                
                if not model_explanation or model_explanation.strip() == "(No explanation)" or model_explanation.strip() == "":
                    option_text = options.get(model_answer, "")
                    teacher_prompt = f"Teacher: Why did you choose {model_answer}? (Option {model_answer}: {option_text})"
                    print(teacher_prompt)
                    f.write(f"{teacher_prompt}\n")
                    
                    explain_prompt = f"Explain why you chose option {model_answer}: {option_text}"
                    start_tag = (forget_model.model_configs or {}).get("question_start_tag", "")
                    end_tag = (forget_model.model_configs or {}).get("question_end_tag", "")
                    if start_tag or end_tag:
                        explain_prompt = f"{start_tag}{explain_prompt}{end_tag}"
                    
                    explain_inputs = forget_model.tokenizer(explain_prompt, return_tensors="pt", truncation=True, max_length=512)
                    explain_inputs = {k: v.to(forget_model.device) for k, v in explain_inputs.items()}
                    explain_input_length = explain_inputs["input_ids"].shape[1]
                    
                    with torch.no_grad():
                        explain_outputs = forget_model.model.generate(
                            **explain_inputs,
                            max_new_tokens=100,
                            min_new_tokens=5,
                            do_sample=False,
                            repetition_penalty=1.2,
                            pad_token_id=forget_model.tokenizer.eos_token_id,
                        )
                    
                    explain_tokens = explain_outputs[0][explain_input_length:]
                    explanation_only = forget_model.tokenizer.decode(explain_tokens, skip_special_tokens=True).strip()
                    
                    if explanation_only and len(explanation_only) > 3:
                        model_explanation = explanation_only
                        _, model_explanation = agent.rewrite_student_answer(model_answer, model_explanation)
                    else:
                        model_explanation = "(No clear explanation provided)"
                    
                    student_explain = f"Student: {model_explanation}"
                    print(student_explain)
                    f.write(f"{student_explain}\n")
                
                round_score = scoring_agent.score_conversation_round(
                    question=question,
                    options=options,
                    correct_answer=correct_answer,
                    student_answer=model_answer,
                    student_explanation=model_explanation,
                    teacher_feedback="",
                    teacher_explanation="",
                    round_num=attempt_num,
                    is_correct=(model_answer.strip().upper() == correct_answer.strip().upper())
                )
                
                answer_score = round_score.get('answer_score', 0)
                explanation_score = round_score.get('explanation_score', 0)
                total_score = round_score.get('total_score', 0)
                
                feedback, explanation = agent.provide_teaching_explanation(
                    question=question,
                    options=options,
                    correct_answer=correct_answer,
                    student_answer=model_answer,
                    student_explanation=model_explanation,
                    answer_score=answer_score,
                    explanation_score=explanation_score
                )
                
                teacher_response = f"Teacher: {feedback}"
                if explanation:
                    teacher_response += f" {explanation}"
                
                if total_score < 1.0 and attempt_num < MAX_ROUNDS:
                    rethink_phrases = [
                        " Please think again.",
                        " Can you reconsider?",
                        " Try once more.",
                        " Give it another thought.",
                        " Think carefully about this.",
                        " Let's try again.",
                        " Reconsider your answer."
                    ]
                    teacher_response += random.choice(rethink_phrases)
                
                print(teacher_response)
                f.write(f"{teacher_response}\n")
                
                round_score_text = f"[Round Score] Answer: {round_score.get('answer_score', 0)}/1, Explanation: {round_score.get('explanation_score', 0)}/1, Total: {round_score.get('total_score', 0):.2f}/1.0 ({round_score.get('explanation_quality', 'N/A')})"
                print(round_score_text)
                f.write(f"{round_score_text}\n")
                
                question_conversation_history.append({
                    "student_answer": model_answer,
                    "student_explanation": model_explanation,
                    "teacher_feedback": feedback,
                    "teacher_explanation": explanation,
                    "round_score": round_score
                })
                
                is_perfect_score = (round_score.get('total_score', 0) == 1.0)
                
                if is_perfect_score:
                    question_result = {
                        "question_num": question_num,
                        "question": question,
                        "options": options,
                        "correct_answer": correct_answer,
                        "attempts": len(question_conversation_history),
                        "final_correct": True,
                        "conversation": question_conversation_history
                    }
                    round_results.append(question_result)
                    break
                else:
                    if (not demo) and (relearn_mode == "train_conversation") and (attempt_num < MAX_ROUNDS):
                        round_dir = os.path.join(round_ckpt_root, f"{kp_safe_name}_{timestamp}", f"q{question_num}_attempt{attempt_num}")
                        trainer_out = os.path.join(round_dir, "trainer_out")
                        os.makedirs(trainer_out, exist_ok=True)
                        
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                forget_model.simple_relearn(
                                    question=question,
                                    correct_answer=correct_answer,
                                    explanation=explanation,
                                    options=options,
                                    learning_rate=relearn_lr,
                                    num_epochs=relearn_epochs,
                                    output_dir=trainer_out,
                                )
                                forget_model.save_round_checkpoint(round_dir)
                                forget_model.reload_from_checkpoint(round_dir)
                        except Exception:
                            pass
            
            else:
                attempts_for_feedback = [
                    {
                        "answer": conv["student_answer"],
                        "explanation": conv["student_explanation"]
                    }
                    for conv in question_conversation_history
                ]
                
                teaching = agent.provide_teaching_feedback(
                    question, correct_answer, options, 
                    attempts_for_feedback
                )
                
                teacher_final = f"Teacher: {teaching}"
                student_final = f"Student: I understand now. The correct answer is {correct_answer}."
                
                print(teacher_final)
                print(student_final)
                
                f.write(f"{teacher_final}\n")
                f.write(f"{student_final}\n")
                
                question_result = {
                    "question_num": question_num,
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "attempts": MAX_ROUNDS,
                    "final_correct": False,
                    "conversation": question_conversation_history,
                    "teaching_feedback": teaching
                }
                round_results.append(question_result)
        
        total_correct = sum(1 for r in round_results if r.get("final_correct", False))
        accuracy = total_correct / len(round_results) * 100 if len(round_results) > 0 else 0
        
        summary = f"\n\n{'='*80}\n"
        summary += f"Summary: {total_correct}/{actual_rounds} correct ({accuracy:.1f}%)\n"
        summary += f"{'='*80}\n"
        
        f.write(summary)
    
    results = {
        "knowledge_point": knowledge_point,
        "num_rounds": actual_rounds,
        "total_questions_available": len(kp_questions),
        "total_correct": total_correct,
        "accuracy": accuracy,
        "rounds": round_results
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nConversation log saved to: {log_file}")
    print(f"Results saved to: {results_file}\n")
    
    return results


def run_learning_loop(model_path: str, data_path: str = None, split: str = "forget_50",
                      num_questions: int = 5, output_dir: str = None, demo: bool = False,
                      relearn_mode: str = "train_conversation", relearn_lr: float = 1e-4,
                      relearn_epochs: int = 1,
                      round_ckpt_root: str = None,
                      excel_path: str = None,
                      lora_rank: int = 32):
    """Run single-question learning loop with teacher-student dialogue."""
    
    if not OPENAI_API_KEY:
        raise RuntimeError('Missing OpenAI API key. Please set env "OPENAI_API_KEY".')
    agent = ChatGPTAgent(OPENAI_API_KEY)
    
    scoring_agent = ScoringAgent(OPENAI_API_KEY)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if demo:
            forget_model = SimulatedForgetModel(initial_accuracy=0.3)
        else:
            forget_model = ForgetModel(model_path, lora_rank=lora_rank)
    
    data = None
    data_source = ""
    
    if excel_path and os.path.exists(excel_path):
        try:
            data = load_forget_data_from_excel(excel_path)
            data_source = f"excel:{excel_path}"
        except Exception as e:
            if demo:
                raise ValueError(f"Failed to load Excel data from {excel_path}: {e}")
    
    if data is None and data_path:
        try:
            data = load_forget_data(data_path, split)
            data_source = f"datasets:{data_path}/{split}"
        except Exception as e:
            if demo:
                raise ValueError(f"Failed to load dataset from {data_path}/{split}: {e}")
    
    if data is None:
        raise ValueError("No data source available. Please provide --excel_path or --data_path")
    
    selected_indices = random.sample(range(len(data)), min(num_questions, len(data)))
    
    script_dir = Path(__file__).parent.resolve()
    chat_histories_dir = script_dir / "chat_histories"
    
    chat_histories_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = chat_histories_dir

    if round_ckpt_root is None:
        round_ckpt_root = str((chat_histories_dir / "round_checkpoints").resolve())
    os.makedirs(round_ckpt_root, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = chat_histories_dir / f"learning_loop_{timestamp}.txt"
    results_file = chat_histories_dir / f"learning_results_{timestamp}.json"
    
    results = []
    results_by_kp = {}
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"Chat Learning Loop - Dialogue Record\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Data Source: {data_source}\n")
        f.write(f"Number of Questions: {num_questions}\n")
        f.write(f"LoRA Rank: {lora_rank}\n")
        f.write(f"=" * 80 + "\n\n")
        
        for q_idx, data_idx in enumerate(selected_indices):
            item = data[data_idx]
            
            if data_source.startswith("excel:"):
                question = item["question"]
                options = item["options"]
                correct_answer = item["correct_answer"]
                title = item["title"]
            else:
                question, options, correct_answer, title = parse_question_with_options(item)
            
            if q_idx > 0:
                print()  # 空行分隔
            
            kp_safe_name = "".join(c if c.isalnum() else "_" for c in title[:50])
            kp_dir = chat_histories_dir / kp_safe_name
            kp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each question's dialogue to a separate file in knowledge point directory
            question_chat_file = kp_dir / f"chat_q{q_idx+1}_{timestamp}.txt"
            question_chat_lines = []
            
            # Dialogue file header - only question and options
            options_text_header = "\n".join([f"{k}. {v}" for k, v in options.items()])
            question_chat_lines.append(f"Question: {question}\n\nOptions:\n{options_text_header}\n")
            
            # Main log file
            f.write(f"\n{'='*60}\n")
            f.write(f"Question {q_idx + 1}/{num_questions}\n")
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
            
            # Conversation history - allows model to "see" teacher's previous feedback
            conversation_history = []
            
            # Learning loop - simulates teacher-student dialogue
            for round_num in range(1, MAX_ROUNDS + 1):
                f.write(f"\n--- Round {round_num} ---\n")
                
                if round_num == 1:
                    options_display = "\n".join([f"   {k}. {v}" for k, v in options.items()])
                    teacher_question = f"Teacher: Please answer the following question.\n\nQuestion: {question}\n\nOptions:\n{options_display}"
                    print(teacher_question)
                    question_chat_lines.append(teacher_question)
                


                model_answer, model_explanation = forget_model.answer_question(
                    question, options, correct_answer, 
                    conversation_history=conversation_history
                )
                
                
                if model_explanation and model_explanation.strip() and model_explanation.strip() != "(No explanation)":
                    model_answer, model_explanation = agent.rewrite_student_answer(model_answer, model_explanation)
                if model_explanation and model_explanation.strip() and model_explanation.strip() != "(No explanation)":
                    student_response = f"Student: {model_answer}. {model_explanation}"
                else:
                    student_response = f"Student: {model_answer}."
                
                print(student_response)
                f.write(f"{student_response}\n")
                question_chat_lines.append(student_response)
                
                if not model_explanation or model_explanation.strip() == "(No explanation)" or model_explanation.strip() == "":
                    option_text = options.get(model_answer, "")
                    teacher_prompt = f"Teacher: Why did you choose {model_answer}? (Option {model_answer}: {option_text})"
                    print(teacher_prompt)
                    f.write(f"{teacher_prompt}\n")
                    question_chat_lines.append(teacher_prompt)
                    
                    explain_prompt = f"Question: {question}\n\nYou chose option {model_answer}: {option_text}\n\nExplain why you chose this option (in one sentence):\n"
                    
                    start_tag = (forget_model.model_configs or {}).get("question_start_tag", "")
                    end_tag = (forget_model.model_configs or {}).get("question_end_tag", "")
                    if start_tag or end_tag:
                        explain_prompt = f"{start_tag}{explain_prompt}{end_tag}"
                    
                    explain_inputs = forget_model.tokenizer(explain_prompt, return_tensors="pt", truncation=True, max_length=512)
                    explain_inputs = {k: v.to(forget_model.device) for k, v in explain_inputs.items()}
                    explain_input_length = explain_inputs["input_ids"].shape[1]
                    
                    with torch.no_grad():
                        explain_outputs = forget_model.model.generate(
                            **explain_inputs,
                            max_new_tokens=100,
                            min_new_tokens=5,
                            do_sample=False,
                            repetition_penalty=1.2,
                            pad_token_id=forget_model.tokenizer.eos_token_id,
                        )
                    
                    explain_tokens = explain_outputs[0][explain_input_length:]
                    explanation_only = forget_model.tokenizer.decode(explain_tokens, skip_special_tokens=True).strip()
                    
                    if explanation_only and len(explanation_only) > 3:
                        model_explanation = explanation_only
                    else:
                        model_explanation = "(No clear explanation provided)"
                    
                    if model_explanation and model_explanation != "(No clear explanation provided)":
                        _, model_explanation = agent.rewrite_student_answer(model_answer, model_explanation)
                    
                    student_explain = f"Student: {model_explanation}"
                    print(student_explain)
                    f.write(f"{student_explain}\n")
                    question_chat_lines.append(student_explain)
                
                round_score = scoring_agent.score_conversation_round(
                    question=question,
                    options=options,
                    correct_answer=correct_answer,
                    student_answer=model_answer,
                    student_explanation=model_explanation,
                    teacher_feedback="",
                    teacher_explanation="",
                    round_num=round_num,
                    is_correct=(model_answer.strip().upper() == correct_answer.strip().upper())
                )
                
                answer_score = round_score.get('answer_score', 0)
                explanation_score = round_score.get('explanation_score', 0)
                total_score = round_score.get('total_score', 0)
                
                feedback, explanation = agent.provide_teaching_explanation(
                    question=question,
                    options=options,
                    correct_answer=correct_answer,
                    student_answer=model_answer,
                    student_explanation=model_explanation,
                    answer_score=answer_score,
                    explanation_score=explanation_score
                )
                
                is_correct = (answer_score == 1)
                
                round_score_text = f"[Round Score] Answer: {round_score.get('answer_score', 0)}/1, Explanation: {round_score.get('explanation_score', 0)}/1, Total: {round_score.get('total_score', 0):.2f}/1.0 ({round_score.get('explanation_quality', 'N/A')})"
                print(round_score_text)
                f.write(f"{round_score_text}\n")
                question_chat_lines.append(round_score_text)
                
                attempt = {
                    "round": round_num,
                    "answer": model_answer,
                    "explanation": model_explanation,
                    "is_correct": is_correct,
                    "feedback": feedback,
                    "teacher_explanation": explanation,
                    "round_score": round_score
                }
                question_result["attempts"].append(attempt)
                
                is_perfect_score = (round_score.get('total_score', 0) == 1.0)
                
                if is_perfect_score:
                    # Natural teacher response for correct answer
                    teacher_response = f"Teacher: {feedback}"
                    print(teacher_response)
                    f.write(f"{teacher_response}\n")
                    question_chat_lines.append(teacher_response)
                    
                    question_result["final_correct"] = True
                    question_result["rounds_needed"] = round_num
                    break
                else:
                    num_attempts = len(question_result["attempts"])
                    
                    teacher_response = f"Teacher: {feedback}"
                    if explanation:
                        teacher_response += f" {explanation}"
                    
                    if round_num < MAX_ROUNDS:
                        if num_attempts == 2:
                            approach_phrases = [
                                f" Let's try a different approach. Think about the key concept: what makes option {correct_answer} different from the others?",
                                f" Consider this from another angle. What distinguishes option {correct_answer} from the rest?",
                                f" Let me help you think differently. Focus on why option {correct_answer} stands out.",
                                f" Try approaching this differently. What's unique about option {correct_answer}?"
                            ]
                            teacher_response += random.choice(approach_phrases)
                        elif num_attempts >= 3:
                            hint_phrases = [
                                f" Let me give you a hint: The answer is {correct_answer}. Focus on understanding WHY this is correct.",
                                f" Here's a hint: It's {correct_answer}. Try to understand the reasoning behind it.",
                                f" I'll help you: The correct answer is {correct_answer}. Think about why this makes sense.",
                                f" A clue for you: {correct_answer} is correct. Can you see why now?"
                            ]
                            teacher_response += random.choice(hint_phrases)
                        else:
                            rethink_phrases = [
                                " Please think again.",
                                " Can you reconsider?",
                                " Try once more.",
                                " Give it another thought.",
                                " Think carefully about this.",
                                " Let's try again.",
                                " Reconsider your answer."
                            ]
                            teacher_response += random.choice(rethink_phrases)
                    
                    print(teacher_response)
                    f.write(f"{teacher_response}\n")
                    question_chat_lines.append(teacher_response)
                    
                    conversation_history.append({
                        "student_answer": model_answer,
                        "student_explanation": model_explanation,
                        "teacher_feedback": feedback,
                        "teacher_explanation": explanation
                    })

                    if (not demo) and (relearn_mode == "train_conversation") and (round_num < MAX_ROUNDS):
                        q_round_dir = os.path.join(round_ckpt_root, f"q{q_idx+1}_{timestamp}", f"round_{round_num}")
                        trainer_out = os.path.join(q_round_dir, "trainer_out")
                        os.makedirs(trainer_out, exist_ok=True)
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                forget_model.simple_relearn(
                                    question=question,
                                    correct_answer=correct_answer,
                                    explanation=explanation,
                                    options=options,
                                    learning_rate=relearn_lr,
                                    num_epochs=relearn_epochs,
                                    output_dir=trainer_out,
                                )
                                forget_model.save_round_checkpoint(q_round_dir)
                                forget_model.reload_from_checkpoint(q_round_dir)
                        except Exception:
                            pass
            
            else:
                # Reached max rounds without correct answer - teacher gives final detailed explanation
                teaching = agent.provide_teaching_feedback(
                    question, correct_answer, options, 
                    question_result["attempts"]
                )
                
                teacher_final = f"Teacher: {teaching}"
                student_final = f"Student: I understand now. The correct answer is {correct_answer}."
                
                print(teacher_final)
                print(student_final)
                
                f.write(f"{teacher_final}\n")
                f.write(f"{student_final}\n")
                
                question_chat_lines.append(teacher_final)
                question_chat_lines.append(student_final)
                
                question_result["final_correct"] = False
                question_result["rounds_needed"] = MAX_ROUNDS
                question_result["teaching_feedback"] = teaching
            
            results.append(question_result)
            
            with open(question_chat_file, "w", encoding="utf-8") as chat_f:
                chat_f.write("\n".join(question_chat_lines))
            
            if kp_safe_name not in results_by_kp:
                results_by_kp[kp_safe_name] = {
                    "knowledge_point": title,
                    "questions": []
                }
            results_by_kp[kp_safe_name]["questions"].append(question_result)
            
            f.write(f"\n{'='*60}\n")
        
        # Batch relearning if in batch mode (after processing all questions)
        if relearn_mode == "batch" and not demo and hasattr(forget_model, 'batch_relearn_from_buffer'):
            f.write(f"\n{'='*60}\n")
            f.write("Executing Batch Relearning...\n")
            try:
                forget_model.batch_relearn_from_buffer(
                    learning_rate=relearn_lr, num_epochs=relearn_epochs
                )
                f.write("Batch relearning completed\n")
            except Exception as e:
                f.write(f"Batch relearning failed: {e}\n")
        
        # Summary statistics
        total_correct = sum(1 for r in results if r.get("final_correct", False))
        avg_rounds = sum(r.get("rounds_needed", MAX_ROUNDS) for r in results) / len(results)
        
        summary = f"\n[Summary] {total_correct}/{len(results)} correct ({total_correct/len(results)*100:.1f}%), avg {avg_rounds:.1f} rounds"
        f.write(f"\n{'='*40}\n{summary}\n")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    for kp_safe_name, kp_data in results_by_kp.items():
        kp_dir = chat_histories_dir / kp_safe_name
        kp_results_file = kp_dir / f"results_{timestamp}.json"
        with open(kp_results_file, "w", encoding="utf-8") as f:
            json.dump(kp_data, f, ensure_ascii=False, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Chat Learning Loop - Teacher-Student Dialogue Learning (LoRA)")
    parser.add_argument("--model_path", type=str, 
                        default="./results/mistralai/Mistral-7B-Instruct-v0.3/intervention/20_0.0001_forget_10",
                        help="Forget model path (relative to project root)")
    parser.add_argument("--data_path", type=str,
                        default=None,
                        help="Dataset path (HuggingFace datasets format, optional)")
    parser.add_argument("--excel_path", type=str,
                        default="./data_construct/raw_data/questions_test.xlsx",
                        help="Excel file path for forget data (preferred over --data_path)")
    parser.add_argument("--split", type=str, default="forget_10",
                        help="Dataset split (for HuggingFace datasets)")
    parser.add_argument("--num_questions", type=int, default=1,
                        help="Number of questions to test (default: 1, for single-question mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode (uses simulated model, no GPU required)")
    
    # Multi-round conversation mode
    parser.add_argument("--multi_round", action="store_true",
                        help="Enable multi-round conversation mode (30 rounds on same knowledge point)")
    parser.add_argument("--num_rounds", type=int, default=30,
                        help="Number of conversation rounds for multi-round mode (default: 30)")
    parser.add_argument("--knowledge_point", type=str, default=None,
                        help="Specific knowledge point to test (if None, randomly select one)")
    
    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank for fine-tuning (default: 32)")
    
    # Learning mode configuration
    parser.add_argument("--relearn_mode", type=str, default="train_conversation",
                        choices=["conversation", "train_conversation", "batch"],
                        help="Learning mode: conversation=dialogue-only (no training), train_conversation=train 1 round after each teacher correction (default), batch=batch training after collecting feedback")
    parser.add_argument("--relearn_lr", type=float, default=3e-5,
                        help="Training learning rate (default: 3e-5, balanced for faster learning)")
    parser.add_argument("--relearn_epochs", type=int, default=2,
                        help="Training epochs per round (default: 2 for effective learning, use 3 for stronger effect)")
    parser.add_argument("--round_ckpt_root", type=str, default=None,
                        help="Where to save per-round checkpoints (default: chat_histories/round_checkpoints)")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    if not os.path.isabs(args.model_path):
        candidate_path = project_root / args.model_path
        if candidate_path.exists():
            args.model_path = str(candidate_path)
        else:
            candidate_path = script_dir / args.model_path
            if candidate_path.exists():
                args.model_path = str(candidate_path)
            else:
                args.model_path = str(project_root / args.model_path)
    
    if args.excel_path and not os.path.isabs(args.excel_path):
        candidate_path = project_root / args.excel_path
        if candidate_path.exists():
            args.excel_path = str(candidate_path)
        else:
            candidate_path = script_dir / args.excel_path
            if candidate_path.exists():
                args.excel_path = str(candidate_path)
            else:
                args.excel_path = str(project_root / args.excel_path)
    
    if args.data_path and not os.path.isabs(args.data_path):
        candidate_path = project_root / args.data_path
        if candidate_path.exists():
            args.data_path = str(candidate_path)
        else:
            candidate_path = script_dir / args.data_path
            if candidate_path.exists():
                args.data_path = str(candidate_path)
            else:
                args.data_path = str(project_root / args.data_path)
    
    if args.multi_round:
        print(f"\n{'='*80}")
        print("Multi-Round Conversation Mode")
        print(f"{'='*80}\n")
        
        run_multi_round_conversation(
            model_path=args.model_path,
            data_path=args.data_path,
            split=args.split,
            num_rounds=args.num_rounds,
            knowledge_point=args.knowledge_point,
            output_dir=args.output_dir,
            demo=args.demo,
            relearn_mode=args.relearn_mode,
            relearn_lr=args.relearn_lr,
            relearn_epochs=args.relearn_epochs,
            round_ckpt_root=args.round_ckpt_root,
            excel_path=args.excel_path,
            lora_rank=args.lora_rank,
        )
    else:
        print(f"\n{'='*80}")
        print("Single Question Mode")
        print(f"{'='*80}\n")
        
        run_learning_loop(
            model_path=args.model_path,
            data_path=args.data_path,
            split=args.split,
            num_questions=args.num_questions,
            output_dir=args.output_dir,
            demo=args.demo,
            relearn_mode=args.relearn_mode,
            relearn_lr=args.relearn_lr,
            relearn_epochs=args.relearn_epochs,
            round_ckpt_root=args.round_ckpt_root,
            excel_path=args.excel_path,
            lora_rank=args.lora_rank,
        )


if __name__ == "__main__":
    main()
