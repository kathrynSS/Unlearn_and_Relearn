import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 确保可以从项目根目录导入本仓库的 utils.py，而不是环境中的第三方 utils 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_model_identifiers_from_yaml


def extract_choice(text: str) -> Optional[str]:
    """
    从模型生成的文本中抽取第一个 A/B/C/D 选项字母。
    """
    if not text:
        return None
    text_upper = text.upper()
    # 优先匹配独立的字母（避免误匹配到单词内部）
    m = re.search(r"\b([ABCD])\b", text_upper)
    if m:
        return m.group(1)
    # 退而求其次，匹配任意位置出现的 A/B/C/D
    m = re.search(r"([ABCD])", text_upper)
    return m.group(1) if m else None


def extract_gt_letter(raw: str) -> Optional[str]:
    """
    从 Excel 中的 answer_letter 字段中抽取正确选项字母。
    例如: \"B. open(\\\"data.txt\\\", \\\"r\\\")\" -> \"B\"。
    """
    s = str(raw).strip()
    if not s:
        return None
    m = re.match(r"([ABCDabcd])", s)
    if m:
        return m.group(1).upper()
    return None


def build_prompt(question: str, options: str, question_start_tag: str, question_end_tag: str) -> str:
    """
    构造 Mistral 指令格式的多选题提示词。
    `options` 在 Excel 中是形如 "A. ... | B. ... | C. ... | D. ..." 的字符串。
    """
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
    """
    加载 Mistral 模型和 tokenizer。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    """
    加载带有 LoRA adapter 的模型：
    - base_model_path: HuggingFace 上的基座模型（例如 mistralai/Mistral-7B-Instruct-v0.3）
    - adapter_path: 本地的 LoRA adapter 目录（例如 50_2e-06_forget）
    """
    # tokenizer 一律从基座模型加载
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    # 加载 LoRA adapter
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
    """
    用模型对单条样本生成答案，并解析为 A/B/C/D。
    返回 (解析后的选项字母, 原始生成文本)。
    """
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
    """
    根据用于训练的配置文件，推断遗忘后模型的路径。

    注意：训练时你可能用了 Hydra 的命令行 override（例如修改 lr、num_epochs），
    这会导致实际的 save_dir（例如 `50_5e-06_forget`）和原始 YAML 里写的默认 save_dir
    （例如 `10_0.0005_forget`）不一致。

    这里的策略是：
    1. 如果 cfg.save_dir 这个目录本身存在，就直接用它；
    2. 否则，在 results/<model_path>/<forget_loss>/ 下面自动寻找最新修改的子目录，
       作为遗忘模型目录。
    """
    cfg = OmegaConf.load(forget_cfg_path)

    # 1) 优先使用 cfg.save_dir（如果这个目录真实存在）
    save_dir = Path(str(cfg.save_dir))
    if save_dir.exists():
        return str(save_dir)

    # 2) 如果 cfg.save_dir 不存在，说明训练时可能使用了 lr / num_epochs 等 override，
    #    那么就在 results/<model_path>/<forget_loss>/ 下自动寻找最新的一个子目录
    root = Path(str(getattr(cfg, "save_dir_root", "results"))) / str(cfg.model_path) / str(cfg.forget_loss)
    if not root.exists():
        raise FileNotFoundError(f"遗忘模型根目录不存在：{root}（请确认已经完成过训练）")

    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"在目录 {root} 下没有找到任何遗忘模型子目录，请先完成训练。")

    # 按修改时间排序，选择最近一次的训练结果
    candidates.sort(key=lambda p: p.stat().st_mtime)
    latest_dir = candidates[-1]

    return str(latest_dir)


def main():
    parser = argparse.ArgumentParser(
        description="使用 Mistral-7B 对 questions_test.xlsx 进行多选题评估，"
        "对比原始模型与遗忘后模型在 A/B/C/D 选择上的准确率。"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data_construct/raw_data/questions_test.xlsx",
        help="测试集 Excel 文件路径（包含 question, options, answer_letter 等列）。",
    )
    parser.add_argument(
        "--forget_cfg",
        type=str,
        default="config/forget_ai.yaml",
        help="用于训练遗忘模型的配置文件（例如 config/forget_ai.yaml），"
        "脚本会从中读取 save_dir 以加载遗忘后的模型。",
    )
    parser.add_argument(
        "--forget_subdir",
        type=str,
        default=None,
        help="可选：手动指定遗忘模型所在的子目录名（如 '70_1e-06_forget'），"
             "将覆盖自动搜索最新子目录的逻辑。",
    )
    parser.add_argument(
        "--forget_adapter_dir",
        type=str,
        default=None,
        help="可选：如果遗忘结果只有 LoRA 适配器（包含 adapter_model.safetensors），"
             "直接指定该目录路径（绝对或相对）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda 或 cpu。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="可选，若指定则只评估前 N 条样本，便于快速测试。",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 1. 读取测试集
    test_path = Path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"测试集文件不存在：{test_path}")

    df = pd.read_excel(test_path)
    required_cols = {"question", "options", "answer_letter"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"questions_test.xlsx 需要包含列: {required_cols}，当前为: {set(df.columns)}")

    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"仅评估前 {len(df)} 条样本。")
    else:
        print(f"评估全部 {len(df)} 条样本。")

    # 2. 读取 Mistral 在 config/model_config.yaml 中的配置（主要是提示词标签）
    model_cfg = get_model_identifiers_from_yaml("mistral-7b")
    q_start = model_cfg["question_start_tag"]
    q_end = model_cfg["question_end_tag"]

    # 3. 加载原始（未遗忘）模型：使用 forget_ai.yaml 中的 model_path（通常是预训练 Mistral）
    forget_cfg = OmegaConf.load(args.forget_cfg)
    base_model_path = forget_cfg.model_path or model_cfg["hf_key"]
    print(f"加载原始模型（未遗忘）: {base_model_path}")
    base_tokenizer, base_model = load_model(base_model_path, device)

    # 4. 对原始模型进行评估
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
        print(f"[原始模型] 样本 {idx + 1}/{len(df)}: 预测={choice}, GT={gt_letters[idx]} (原始: {gt_full[idx]})")

    # 计算原始模型准确率（先保存结果，稍后与遗忘模型一起打印）
    base_correct = sum(
        1 for pred, gt in zip(base_preds, gt_letters) if pred is not None and gt is not None and pred == gt
    )
    base_acc = base_correct / len(df)

    # 释放显存（如果使用 GPU）
    del base_model, base_tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 5. 加载遗忘后的模型（自动识别：完整模型目录 or 仅 LoRA 适配器目录）
    # 优先：用户显式提供适配器目录
    if args.forget_adapter_dir is not None:
        forget_path = Path(args.forget_adapter_dir).expanduser().resolve()
        if not forget_path.exists():
            raise FileNotFoundError(f"指定的遗忘适配器目录不存在：{forget_path}")
        forget_model_path = forget_path
        is_adapter_only = True
        print(f"\n加载遗忘后模型（LoRA 适配器模式）: {forget_model_path}")
        forget_tokenizer, forget_model = load_model_with_adapter(
            base_model_path=base_model_path,
            adapter_path=str(forget_model_path),
            device=device,
        )
    else:
        # 默认：基于用于训练的配置文件自动推断遗忘模型目录；
        # 若只含 LoRA 适配器则自动切换到「基座 + 适配器」模式。
        #
        # - 若指定了 forget_subdir，则在推断出的根目录下使用该子目录名；
        # - 否则，使用 find_forget_model_path() 返回的目录（通常是最近一次训练结果）。
        auto_dir = Path(find_forget_model_path(args.forget_cfg)).resolve()
        if args.forget_subdir is not None:
            root_dir = auto_dir.parent  # results/<model_path>/<forget_loss>/
            forget_model_path = root_dir / args.forget_subdir
        else:
            forget_model_path = auto_dir

        if not forget_model_path.exists():
            raise FileNotFoundError(f"遗忘模型目录不存在：{forget_model_path}")

        # 判定是否为“仅 LoRA 适配器”目录
        adapter_file = forget_model_path / "adapter_model.safetensors"
        full_model_files = [
            forget_model_path / "pytorch_model.bin",
            forget_model_path / "model.safetensors",
            forget_model_path / "config.json",
        ]
        is_adapter_only = adapter_file.exists() and not any(p.exists() for p in full_model_files)

        if is_adapter_only:
            print(f"\n检测到 LoRA 适配器目录（未找到完整权重），使用基座+适配器加载: {forget_model_path}")
            forget_tokenizer, forget_model = load_model_with_adapter(
                base_model_path=base_model_path,
                adapter_path=str(forget_model_path),
                device=device,
            )
        else:
            print(f"\n加载遗忘后完整模型目录: {forget_model_path}")
            forget_tokenizer, forget_model = load_model(
                model_path=str(forget_model_path),
                device=device,
            )

    # 6. 对遗忘模型进行评估
    forget_preds = []
    forget_raw = []

    for idx, row in df.iterrows():
        question = str(row["question"])
        options = str(row["options"])
        prompt = build_prompt(question, options, q_start, q_end)
        choice, raw_text = generate_choice(forget_tokenizer, forget_model, prompt, device)
        forget_preds.append(choice)
        forget_raw.append(raw_text)
        print(f"[遗忘模型] 样本 {idx + 1}/{len(df)}: 预测={choice}, GT={gt_letters[idx]} (原始: {gt_full[idx]})")

    forget_correct = sum(
        1 for pred, gt in zip(forget_preds, gt_letters) if pred is not None and gt is not None and pred == gt
    )
    forget_acc = forget_correct / len(df)

    # 7. 统计两个模型在多少条样本上的预测不同
    diff_count = sum(
        1
        for base_p, forget_p in zip(base_preds, forget_preds)
        if base_p != forget_p
    )

    # 8. 打印最终结果
    print("\n================= 准确率总结 =================")
    print(f"原始模型准确率: {base_acc * 100:.2f}% ({base_correct}/{len(df)})")
    print(f"遗忘模型准确率: {forget_acc * 100:.2f}% ({forget_correct}/{len(df)})")
    print(f"两模型预测不同的样本数: {diff_count} / {len(df)}")


if __name__ == "__main__":
    main()


