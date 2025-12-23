import os
import sys
import json
import pickle
import string
import re
from functools import reduce
from contextlib import contextmanager
import copy

# ============= 本地资源路径配置（必须在导入其他库之前设置） =============
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_RESOURCES_DIR = os.path.join(PROJECT_ROOT, "local_resources")

# 创建本地资源目录
os.makedirs(LOCAL_RESOURCES_DIR, exist_ok=True)

# 配置 NLTK 本地数据路径（下载也会保存到这里）
NLTK_LOCAL_DIR = os.path.join(LOCAL_RESOURCES_DIR, "nltk_data")
os.makedirs(NLTK_LOCAL_DIR, exist_ok=True)
os.environ["NLTK_DATA"] = NLTK_LOCAL_DIR

# 配置 HuggingFace 缓存路径（所有模型下载都会保存到这里）
HF_LOCAL_DIR = os.path.join(LOCAL_RESOURCES_DIR, "huggingface_models")
os.makedirs(HF_LOCAL_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_LOCAL_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_LOCAL_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_LOCAL_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_LOCAL_DIR, "datasets")

# 配置 Stanza 本地资源路径
STANZA_LOCAL_DIR = os.path.join(LOCAL_RESOURCES_DIR, "stanza_resources")
os.makedirs(STANZA_LOCAL_DIR, exist_ok=True)
os.environ["STANZA_RESOURCES_DIR"] = STANZA_LOCAL_DIR

# 配置 spaCy 本地模型路径
SPACY_LOCAL_DIR = os.path.join(LOCAL_RESOURCES_DIR, "spacy_models", "en_core_web_sm")

# 配置 TensorFlow/TensorBoard 日志路径
TF_LOGS_DIR = os.path.join(LOCAL_RESOURCES_DIR, "tf_logs")
os.makedirs(TF_LOGS_DIR, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = os.path.join(LOCAL_RESOURCES_DIR, "tfhub_cache")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少 TF 日志输出

# 配置 Torch Hub 缓存路径
TORCH_HUB_DIR = os.path.join(LOCAL_RESOURCES_DIR, "torch_hub")
os.makedirs(TORCH_HUB_DIR, exist_ok=True)
os.environ["TORCH_HOME"] = TORCH_HUB_DIR

# 配置 XDG 缓存目录（许多库会使用这个）
XDG_CACHE_DIR = os.path.join(LOCAL_RESOURCES_DIR, "cache")
os.makedirs(XDG_CACHE_DIR, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = XDG_CACHE_DIR

# 配置 wandb 目录
WANDB_DIR = os.path.join(PROJECT_ROOT, "wandb_logs")
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ["WANDB_DIR"] = WANDB_DIR
os.environ["WANDB_CACHE_DIR"] = os.path.join(LOCAL_RESOURCES_DIR, "wandb_cache")

print(f"========== 资源目录配置 ==========")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"本地资源目录: {LOCAL_RESOURCES_DIR}")
print(f"NLTK 数据: {NLTK_LOCAL_DIR}")
print(f"HuggingFace 模型: {HF_LOCAL_DIR}")
print(f"Stanza 资源: {STANZA_LOCAL_DIR}")
print(f"Torch Hub: {TORCH_HUB_DIR}")
print(f"TF 日志: {TF_LOGS_DIR}")
print(f"XDG 缓存: {XDG_CACHE_DIR}")
print(f"Wandb 日志: {WANDB_DIR}")
print(f"==================================")

# 现在导入其他库（环境变量已设置，下载会保存到项目目录）
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
import datasets
import numpy as np
import difflib
import stanza
from tqdm import tqdm
import spacy
import nltk

# ============= Patch Stanza MD5 校验（解决服务器文件与元数据不匹配的问题）=============
# 这是 Stanza 的一个已知问题：服务器上的模型文件已更新，但 resources.json 中的 MD5 是旧的
try:
    from stanza.resources import common as stanza_common
    _original_assert_file_exists = stanza_common.assert_file_exists
    
    def _patched_assert_file_exists(path, md5=None, alternate_md5=None):
        """跳过 MD5 校验，只验证文件存在"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required Stanza resource not found: {path}")
        # 文件存在就认为 OK，跳过 MD5 校验
    
    stanza_common.assert_file_exists = _patched_assert_file_exists
    print("已禁用 Stanza MD5 校验（解决版本不匹配问题）")
except Exception as e:
    print(f"警告: 无法禁用 Stanza MD5 校验: {e}")
# =============================================================================

# 设置 NLTK 数据路径
nltk.data.path.insert(0, NLTK_LOCAL_DIR)

from utils import get_model_identifiers_from_yaml, split_document, replace_name, replace_name_only_first_n

# 导入 NLTK 模块（在设置路径后）
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stanza's DownloadMethod enum is not available in some versions.
# Provide a lightweight fallback so the rest of the code can run even
# when the enum or the download_method argument is missing.
try:
    from stanza.resources.common import DownloadMethod
except ImportError:  # pragma: no cover - version-dependent compatibility
    class DownloadMethod:  # type: ignore
        REUSE_RESOURCES = None
        NONE = None

# 注：不再使用 coref，无需 PyTorch 2.6 兼容性修复


@contextmanager
def _stanza_weights_only_compat():
    """Temporarily force torch.load to allow pickled objects (weights_only=False)."""
    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


# Stanza 资源目录：始终使用项目目录下的 local_resources/stanza_resources
# （环境变量 STANZA_RESOURCES_DIR 已在文件开头设置）
STANZA_RESOURCES_DIR = STANZA_LOCAL_DIR
STANZA_RESOURCES_JSON = os.path.join(STANZA_RESOURCES_DIR, "resources.json")


def _instantiate_stanza_pipeline(processors: str, download_method: DownloadMethod):
    """Instantiate a Stanza pipeline, handling older Stanza versions gracefully.

    Some Stanza releases do not expose DownloadMethod and/or the
    `download_method` keyword on `Pipeline`. In those cases, we simply
    omit the argument and fall back to Stanza's default behaviour.
    """
    with _stanza_weights_only_compat():
        pipeline_kwargs = dict(
            lang="en",
            processors=processors,
            dir=STANZA_RESOURCES_DIR,
        )
        # Only pass download_method if we have a non-None value; this
        # keeps the code working when using the fallback DownloadMethod
        # or older Stanza versions that don't accept the argument.
        if download_method is not None:
            pipeline_kwargs["download_method"] = download_method

        return stanza.Pipeline(**pipeline_kwargs)


def _load_stanza_pipeline(processors: str):
    # 如果本地资源存在，优先使用 DownloadMethod.NONE 跳过 MD5 校验
    # 这样可以避免因 Stanza 版本不同导致的 MD5 不匹配错误
    if os.path.exists(STANZA_RESOURCES_JSON):
        try:
            return _instantiate_stanza_pipeline(processors, DownloadMethod.NONE)
        except Exception as e:
            tqdm.write(f"使用本地 Stanza 资源失败: {e}，尝试重新下载...")
    
    # 本地资源不存在或加载失败，尝试下载
    try:
        return _instantiate_stanza_pipeline(
            processors, DownloadMethod.REUSE_RESOURCES
        )
    except requests.exceptions.RequestException as err:
        raise RuntimeError(
            f"Stanza resources missing in {STANZA_RESOURCES_DIR}. "
            "Please run `python download_resources.py --skip_nltk --skip_spacy --skip_model` "
            "on a machine with internet access once and retry."
        ) from err
    except ValueError as err:
        # MD5 校验失败，尝试强制使用本地资源
        if "md5" in str(err).lower() and os.path.exists(STANZA_RESOURCES_JSON):
            tqdm.write(
                "Stanza MD5 校验失败（可能是版本不匹配），强制使用本地资源..."
            )
            return _instantiate_stanza_pipeline(processors, DownloadMethod.NONE)
        raise

counterfact_prompt = "Complete the following passage about {replace_ent}."

# 不再使用 coref（共指消解），原因：
# 1. AI 知识点数据集不需要（概念词不存在代词指代）
# 2. TOFU 数据集也有 fallback 机制
# 3. 加快加载速度，减少依赖问题
nlp = _load_stanza_pipeline("tokenize,ner")
nlp_nocoref = nlp  # 统一使用同一个 pipeline

# 加载 spaCy 模型：优先从本地目录加载
if os.path.exists(SPACY_LOCAL_DIR):
    spacynlp = spacy.load(SPACY_LOCAL_DIR)
    print(f"使用本地 spaCy 模型: {SPACY_LOCAL_DIR}")
else:
    spacynlp = spacy.load("en_core_web_sm")
    print("使用系统安装的 spaCy 模型: en_core_web_sm")

stop_word_list = set(stopwords.words("english")).union(
    spacynlp.Defaults.stop_words
)


def find_non_ascii(s):
    indices = []
    start = None
    for i, c in enumerate(s):
        if not c.isascii():
            if start is None:
                start = i
        else:
            if start is not None and not c.isspace():
                indices.append((start, i))
                start = None
    if start is not None:
        indices.append((start, len(s)))
    non_ascii_spans = [s[i:j].strip() for i, j in indices if j - i > 3]
    # use regex to find spans like [...]
    spans = re.findall(r'\[(.*)\]', s)
    spans = [i.strip() for i in spans if not i.isascii() and len(i) > 3]
    spans += non_ascii_spans
    return sorted(spans, key=lambda x: len(x), reverse=True)


def get_target_ent_mentions(document, title, target_type='person'):
    """Find mentions of the target entity in the document.
    
    简化版本：不使用 coref（共指消解），仅使用 NER 和字符串匹配。
    对于 AI 知识点数据集完全足够，对于 TOFU 也能正常工作。
    """
    # clean format for stanza
    if '==\n' in document:
        stanza_content = document.replace('==\n', '==\n\n')
    else:
        stanza_content = document.replace('\n', '\n\n').replace('\n\n\n\n', '\n\n')
    
    # 确保无论下面 try 是否成功，doc 至少被定义
    doc = None
    try:
        doc = nlp(stanza_content)
    except:
        # stanza error
        stanza_content = document.split('\n\n\n')
        if len(stanza_content) > 1:
            stanza_content = stanza_content[:-1]
        stanza_content = '\n\n\n'.join(stanza_content)
        try:
            doc = nlp(stanza_content)
        except:
            pass
    
    if doc is None:
        # 如果 Stanza 完全解析失败，退化为简单的字符串匹配
        print(f"Error in processing {title} with Stanza, fallback to simple string search.")
        for word in title.split():
            if word in document:
                return [{'mention': title,
                         'position': document.index(word),
                         'prefix': document[:document.index(word)]}], [title]
        return [{'mention': title, 'position': -1, 'prefix': document}], [title]
    
    # 直接使用 NER 实体和字符串匹配（不使用 coref）
    target_ents = []
    ents = [ent for ent in doc.ents if len(ent.text) > 1]
    
    # 1. 检查 NER 实体是否与 title 匹配
    for ent in ents:
        ent_text_clean = ent.text.replace("'s", "").replace("'s", "").lower().strip()
        if ent_text_clean in title.lower() or title.lower() in ent_text_clean:
            target_ents.append({'mention': ent.text, 'position': ent.start_char, 'prefix': stanza_content[:ent.start_char]})
    
    # 2. 如果是 person 类型，也检查 PERSON 类型的实体
    if target_type == 'person':
        person_ents = [ent for ent in doc.ents if ent.type == 'PERSON']
        for ent in person_ents:
            # 检查是否与 title 的任何部分匹配
            if any(word.lower() in ent.text.lower() for word in title.split()):
                if not any(ent.start_char == e['position'] for e in target_ents):
                    target_ents.append({'mention': ent.text, 'position': ent.start_char, 'prefix': stanza_content[:ent.start_char]})
    
    # 3. 如果没有找到任何实体，用简单字符串匹配
    if len(target_ents) == 0:
        for word in title.split():
            if word in document:
                return [{'mention': title, 'position': document.index(word), 'prefix': document[:document.index(word)]}], [title]
        return [{'mention': title, 'position': -1, 'prefix': document}], [title]
    
    # 收集所有提及文本
    target_ent_mention_texts = [i['mention'] for i in target_ents]
    target_ent_mention_texts = list(set(target_ent_mention_texts))
    target_ent_mention_texts = sorted(target_ent_mention_texts, key=lambda x: len(x), reverse=True)
    
    return target_ents, target_ent_mention_texts


def get_book_names(document, additional_target_type, title, replace_book_names):
    work_of_art_ent_positions = []
    work_of_art_replace_name = dict()
    if len(additional_target_type) > 0:
        doc = nlp_nocoref(document)
        additional_targets = [ent for ent in doc.ents if ent.type in additional_target_type and len(ent.text.split()) > 1]
        
        # clean results where "XX's" is recognized as WORK_OF_ART
        additional_targets = [ent for ent in additional_targets if not ((ent.text.endswith("'s") or ent.text.endswith("’s")) and any([w in ent.text for w in title.split()]))]
        additional_targets = [ent for ent in additional_targets if all([i not in word_tokenize(ent.text.lower()) for i in ['award', 'prize', 'ph.d.']])]
        for each_ent in additional_targets:
            # clean results where person names are included in WORK_OF_ART entities
            each_ent_text = each_ent.text.strip()
            for strip_name in [f"{title}'s", f"{title}’s", title]:
                if each_ent_text.endswith(strip_name):
                    each_ent_text = each_ent_text.rstrip(strip_name).strip()
                if each_ent_text.startswith(strip_name):
                    each_ent_text = each_ent_text.lstrip(strip_name).strip()
            # strip all punctuations
            while each_ent_text[0] in string.punctuation or each_ent_text[-1] in string.punctuation:
                each_ent_text = each_ent_text.strip(string.punctuation)
            each_ent_text = each_ent_text.strip()
            # sometimes stanza only extracts part of the entity
            longest_ent_text = []
            for i in replace_book_names:
                if each_ent_text.lower() in i.lower() and len(i) >= len(each_ent_text) and i.lower() in document.lower():
                    within_index = i.lower().find(each_ent_text.lower())
                    if document.lower()[each_ent.start_char-within_index:each_ent.start_char-within_index+len(i)] == i.lower():
                        longest_ent_text.append(i)
            replace_key = each_ent_text
            if len(longest_ent_text) > 0:
                longest_ent_text = sorted(longest_ent_text, key=lambda x: len(x), reverse=True)
                start_position = document.lower().find(longest_ent_text[0].lower())
                each_ent_text = document[start_position: start_position + len(longest_ent_text[0])]
                replace_key = longest_ent_text[0]
            if replace_key not in replace_book_names:
                print(f"========== Cannot find {replace_key} in replace_book_names ==========")
                continue
            work_of_art_replace_name[each_ent_text] = replace_key
            start_position = 0
            while each_ent_text in document[start_position:]:
                start_position = document.find(each_ent_text, start_position)
                if all([start_position != i['position'] for i in work_of_art_ent_positions]):
                    work_of_art_ent_positions.append({'mention': each_ent_text, 'position': start_position, 'prefix': document[:start_position], 'all_occurrences': [each_ent_text]})
                start_position += 1
        # add titles that are not detected by stanza
        for i in replace_book_names:
            if i.lower() in document.lower() and all([i.lower() not in j['mention'].lower() for j in work_of_art_ent_positions]) and len(i.split()) > 2:
                start_position = 0
                while i.lower() in document[start_position:].lower():
                    start_position = document.lower().find(i.lower(), start_position)
                    each_ent_text = document[start_position: start_position + len(i)]
                    work_of_art_ent_positions.append({'mention': each_ent_text, 'position': start_position, 'prefix': document[:start_position], 'all_occurrences': [each_ent_text]})
                    work_of_art_replace_name[each_ent_text] = i
                    start_position += 1
                    print(f"========== Added {each_ent_text} ==========")
    return work_of_art_ent_positions, work_of_art_replace_name


def get_target_ent_text_indices(tokenizer, original_ids, target_ent_mention_texts):
    # all anchor mentions, including pronouns
    target_text_starts = []
    if len(target_ent_mention_texts) > 0:
        for i in range(1, len(original_ids)):
            tmp_text = tokenizer.decode(original_ids[i:i+len(target_ent_mention_texts[0])], skip_special_tokens=True)
            if any([tmp_text.startswith(j) for j in target_ent_mention_texts]):
                target_text_starts.append(i)
    
    return target_text_starts


def get_ent_indices(named_ents, original_ids, tokenizer):
    ent_spans = []
    if len(named_ents) > 0:
        for i in range(1, len(original_ids)):
            tmp_text = tokenizer.decode(original_ids[i:i+len(named_ents[0])], skip_special_tokens=True).lower()
            for ent in named_ents:
                if tmp_text.startswith(ent.lower()):
                    # find the end of the entity
                    ent_len = len(tokenizer.encode(ent, add_special_tokens=False))
                    for j in range(max(i+1, i+ent_len-2), len(original_ids)+1):
                        if tokenizer.decode(original_ids[i:j], skip_special_tokens=True).lower() not in ent.lower():
                            break
                    ent_spans.append((i, j-1))
                    break
    return ent_spans


def move_probability_to_original_name(match_probs, modify_index, gt_token_id, adjust_ent_first_token_ids):
    for id_to_reduce in adjust_ent_first_token_ids:
        if id_to_reduce == gt_token_id:
            continue
        if match_probs[modify_index, id_to_reduce] > 1e-8:
            match_probs[modify_index, gt_token_id] += match_probs[modify_index, id_to_reduce] - 1e-8
            match_probs[modify_index, id_to_reduce] = 1e-8


@hydra.main(version_base=None, config_path="config", config_name="forget_wpu")
def main(cfg):
    teacher_cfg = cfg.teacher
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]
    
    save_dir = f"{cfg.save_dir_root}/{cfg.model_path}/{cfg.forget_loss}"
    os.makedirs(save_dir, exist_ok=True)
    if teacher_cfg.whp_baseline:
        save_name = f"{save_dir}/{cfg.split}.pkl"
    else:
        save_name = f"{save_dir}/{cfg.split}_{teacher_cfg.N}_{teacher_cfg.counter_fact_prompt}_{teacher_cfg.change_name_back}.pkl"
    if os.path.exists(save_name):
        print(f"========== {save_name} already exists ==========")
        return

    # 检查本地 HuggingFace 模型路径
    local_model_path = os.path.join(HF_LOCAL_DIR, model_id.replace("/", "_"))
    if os.path.exists(local_model_path):
        tokenizer_path = local_model_path
        print(f"使用本地 Tokenizer: {local_model_path}")
    else:
        tokenizer_path = model_id
        print(f"从 HuggingFace Hub 加载 Tokenizer: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    target_type = 'person'
    
    # 自动检测本地还是远程数据集
    if os.path.exists(cfg.data_path):
        # 本地数据集，使用 load_from_disk
        print(f"========== 从本地加载数据集: {cfg.data_path} ==========")
        dataset = datasets.load_from_disk(cfg.data_path)[cfg.split]
    else:
        # 远程数据集，使用 load_dataset
        print(f"========== 从HuggingFace加载数据集: {cfg.data_path} ==========")
        dataset = datasets.load_dataset(cfg.data_path, cfg.split)['train']
    
    if 'data_ai' in cfg.data_path:
        # AI知识点遗忘数据集
        forget_target_name = False
        additional_target_type = []
        tofu = True  # 使用QA格式
        replace_name_file = 'data/replace_knowledge_points.json'
        # 加载知识点列表
        with open('data/ai_knowledge_points.txt', 'r', encoding='utf-8') as f:
            forget_knowledge_points = [line.strip() for line in f if line.strip()]
        # 使用title字段作为知识点
        question_to_title = {item['question']: item['title'] for item in dataset}
        titles = {item['title'] for item in dataset}
        print(f"========== AI数据集: {len(titles)} 个知识点 ==========")
    elif 'TOFU' in cfg.data_path:
        forget_target_name = True
        additional_target_type = ['WORK_OF_ART']
        tofu = True
        replace_name_file = 'data/replace_name_forget10.json'
        # get forget targets
        with open('data/tofu_author.txt', 'r') as f:
            forget_people = f.readlines()
            forget_people = [i.strip() for i in forget_people]
        question_to_title = dict()
        for item in dataset:
            title = [i for i in forget_people if i in item['question'] + item['answer']]
            if len(title) != 1:
                title = [i for i in forget_people if any([w in item['question'] + item['answer'] for w in i.split()])]
                assert len(title) == 1
            question_to_title[item['question']] = title[0]
        titles = {question_to_title[i['question']] for i in dataset}
    else:
        forget_target_name = False
        additional_target_type = []
        tofu = False
        titles = set(dataset['title'])
        replace_name_file = 'data/replace_name_forget_100.json'
    
    # get replacement names
    with open(replace_name_file, 'r') as f:
        replace_person_names = json.load(f)
        print(f"========== Loaded replace names from {replace_name_file} ==========")
    replace_person_names = {k: v for k, v in replace_person_names.items() if k in titles}
    for k in replace_person_names:
        # 灵活处理：如果可用替换少于N，使用所有可用的
        available_replacements = len(replace_person_names[k])
        if available_replacements >= teacher_cfg.N:
            replace_person_names[k] = replace_person_names[k][:teacher_cfg.N]
        else:
            print(f"Warning: {k} has only {available_replacements} replacements (requested N={teacher_cfg.N})")
        # 确保至少有一些替换
        assert len(replace_person_names[k]) > 0, f"No replacements available for {k}"
    # load replace book names
    with open('data/replace_book_names.json', 'r') as f:
        replace_book_names = json.load(f)
    for k, v in replace_book_names.items():
        for i in v:
            assert len(k.split()) == len(i.split())

    # 检查模型路径：优先使用本地下载的模型
    actual_model_path = cfg.model_path
    if not os.path.exists(cfg.model_path):
        # 如果配置的路径不存在，检查本地资源目录
        local_model_path = os.path.join(HF_LOCAL_DIR, cfg.model_path.replace("/", "_"))
        if os.path.exists(local_model_path):
            actual_model_path = local_model_path
            print(f"使用本地模型: {local_model_path}")
        else:
            print(f"将从 HuggingFace Hub 加载模型: {cfg.model_path}")
    
    # Load model; avoid passing unsupported arguments like `use_flash_attention_2`
    # which some architectures (e.g., MistralForCausalLM) do not accept.
    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model.to("cuda")
    print(f"========== Loading from checkpoint: {actual_model_path} ==========")
    
    results = {i: dict() for i in titles}
    processed = set()

    for data_ind in tqdm(range(len(dataset))):
        data_item = dataset[data_ind]
        if tofu:
            document = model_cfg['question_start_tag'] + data_item['question'] + model_cfg['question_end_tag'] + model_cfg['answer_tag'] + data_item['answer']
            title = question_to_title[data_item['question']]
            result_key = data_item['question']
        else:
            if data_item['title'] in processed:
                continue
            document = data_item['wikipage']
            title = data_item['title']
            processed.add(title)
            result_key = title
        
        # get target entities
        target_ent_positions, target_ent_mention_texts = get_target_ent_mentions(document, title, target_type=target_type)
        target_ents = set([i['mention'] for i in target_ent_positions] + [title])
        target_ents = sorted(target_ents, key=lambda x: len(x), reverse=True)
        for anchor_ent_i in target_ent_positions:
            anchor_ent_i['all_occurrences'] = target_ents
        work_of_art_ent_positions, work_of_art_replace_name = get_book_names(document, additional_target_type, title, replace_book_names)
        
        book_ents = {i['mention'] for i in work_of_art_ent_positions}
        print(f"Title: {title}")
        print(f"Person names: {target_ents}")
        print(f"Book names: {book_ents}")
        target_ent_positions = copy.deepcopy(target_ent_positions + work_of_art_ent_positions)
        target_ent_positions = sorted(target_ent_positions, key=lambda x: x['position'])
            
        replace_persons = replace_person_names[title]
        # prepare for name change
        replace_person_first_token_ids = []
        for person in replace_persons:
            words = person.split()
            person_first_token_ids = {tokenizer(tok, add_special_tokens=False)['input_ids'][0] for tok in words}
            words2 = [f"\n{tok}" for tok in words]
            person_first_token_ids_2 = {tokenizer(tok, add_special_tokens=False)['input_ids'][2] for tok in words2}
            for tok_id in person_first_token_ids_2:
                assert any([w.startswith(tokenizer.decode(tok_id, skip_special_tokens=True)) for w in words])
            replace_person_first_token_ids.append({'w_prefix': person_first_token_ids, 'wo_prefix': person_first_token_ids_2, 'all': person_first_token_ids.union(person_first_token_ids_2)})
        
        if cfg.sentence_chunk == -1:
            # whole document
            contents = [document]
        else:
            doc = nlp_nocoref(document)
            if teacher_cfg.whp_baseline:
                contents, _ = split_document(doc.sentences, fix_chunk_token=256, tokenizer=tokenizer)
            else:
                contents, _ = split_document(doc.sentences, chunk_size=cfg.sentence_chunk)
        
        # process each content
        item_results = []
        for content_ind, content in enumerate(contents):
            adjust_list = [[(target_ents, p)] for p in replace_persons]
            for work_i in book_ents:
                replace_key = work_of_art_replace_name[work_i]
                for i, adjust_set in enumerate(adjust_list):
                    adjust_set.append(([work_i], replace_book_names[replace_key][i]))
            for i, adjust_set in enumerate(adjust_list):
                adjust_list[i] = adjust_set[:1] + sorted(adjust_set[1:], key=lambda x: len(x[0][0]), reverse=True)
            
            teacher_dist = dict()
            perturbed_contents, num_added_tokens = [], []
            for adjust_set in adjust_list:
                perturbed_content = copy.deepcopy(content)
                # replace target entity with adjust_ent
                perturbed_content = replace_name(target_ents, title, adjust_set[0][1], perturbed_content, consistent_person_name=target_type=='person')
                for replace_item in adjust_set[1:]:
                    perturbed_content = replace_name(replace_item[0], None, replace_item[1], perturbed_content, consistent_person_name=False)
                if teacher_cfg.counter_fact_prompt:
                    perturbed_content = f'[INST] {counterfact_prompt.format(forget_ent=title, replace_ent=adjust_set[0][1])} [/INST] {perturbed_content}'
                perturbed_contents.append(perturbed_content)
                if model_cfg['question_end_tag'] in perturbed_content:
                    added_content = perturbed_content.split(model_cfg['question_end_tag'])[0] + model_cfg['question_end_tag']
                    num_added_tok = len(tokenizer(added_content, add_special_tokens=True)['input_ids'])
                else:
                    num_added_tok = 0
                num_added_tokens.append(num_added_tok)
            intervened_inputs = tokenizer(perturbed_contents, add_special_tokens=True, return_tensors='pt', padding=True)
            with torch.no_grad():
                outs = model(**intervened_inputs.to('cuda'))
            # logits are in bfloat16; convert to float32 before softmax / numpy
            logits = outs.logits.float()
            if teacher_cfg.whp_baseline:
                # reduce probability of anchor terms
                for adjust_ind, adjust_set in enumerate(adjust_list):
                    for i in range(logits.shape[1]):
                        # do not move probability for the first appearance
                        prefix = tokenizer.decode(intervened_inputs['input_ids'][adjust_ind, :i], skip_special_tokens=True)
                        if adjust_set[0][1] not in prefix:
                            continue
                        # assume only 1 replacement
                        assert adjust_ind == 0
                        logits[adjust_ind, i, list(replace_person_first_token_ids[0]['all'])] -= 5
            
            probs_blv = torch.nn.functional.softmax(logits, dim=-1)
            for i, adjust_set in enumerate(adjust_list):
                teacher_dist[adjust_set[0][1]] = {'probs': probs_blv[i].cpu().numpy(), 'num_added_tokens': num_added_tokens[i], 'replace_ids': intervened_inputs['input_ids'][i].tolist()}
            # 使用实际的替换数量（可能与teacher_cfg.N不同）
            actual_n = len(replace_persons)
            assert len(teacher_dist) == actual_n
            
            # get training indices and probabilities
            original_ids = tokenizer(content, add_special_tokens=True)['input_ids']
            # find named entity indices
            target_text_starts = get_target_ent_text_indices(tokenizer, original_ids, target_ent_mention_texts)
            target_ent_mention_spans = get_ent_indices(target_ents, original_ids, tokenizer)
            # find target indices
            forget_target_indices = set()
            if teacher_cfg.change_name_back:
                for name in target_ents + [name_i['mention'] for name_i in work_of_art_ent_positions]:
                    name_span = get_ent_indices([name], original_ids, tokenizer)
                    forget_target_indices.update({i for s, e in name_span for i in range(s, e)})
            tmp_dict = dict()
            for adjust_set_ind, adjust_set in enumerate(adjust_list):
                adjust_ent = adjust_set[0][1]
                num_added_tok = teacher_dist[adjust_ent]['num_added_tokens']
                replace_ids = teacher_dist[adjust_ent]['replace_ids']
                matcher = difflib.SequenceMatcher(None, original_ids, replace_ids)
                blocks = matcher.get_matching_blocks()
                blocks = [b for b in blocks if b.size > 1]
                ori_match_ids, rep_match_ids = [], []
                for block in blocks:
                    if block.b + block.size <= num_added_tok:
                        continue
                    if any([block.a-1 >= s and block.a-1 < e for s, e in target_ent_mention_spans]) and not any([block.a >= s and block.a < e for s, e in target_ent_mention_spans]):
                        assert tokenizer.decode(replace_ids[block.b-1:block.b], skip_special_tokens=True) in adjust_ent or tokenizer.decode(replace_ids[block.b-1:block.b], skip_special_tokens=True)[-1] == "'"
                        a_start = block.a-1
                        b_start = block.b-1
                    else:
                        a_start = block.a
                        b_start = block.b
                    if b_start < num_added_tok:
                        offset = num_added_tok - 1 - b_start
                        b_start += offset
                        a_start += offset
                    ori_match_ids.extend(list(range(a_start, block.a + block.size)))
                    rep_match_ids.extend(list(range(b_start, block.b + block.size)))
                keep_indices = [i for i, v in enumerate(ori_match_ids) if v not in forget_target_indices]
                ori_match_ids = [ori_match_ids[i] for i in keep_indices]
                rep_match_ids = [rep_match_ids[i] for i in keep_indices]
                # get the matched probabilities
                assert len(rep_match_ids) == len(set(rep_match_ids))
                match_probs = teacher_dist[adjust_ent]['probs'][rep_match_ids, :] # (L, V)
                if not teacher_cfg.whp_baseline and teacher_cfg.change_name_back:
                    anchor_positions = target_text_starts
                    for s, e in target_ent_mention_spans:
                        anchor_positions.extend(list(range(s, e)))
                    anchor_positions = sorted(list(set(anchor_positions)))
                    for anchor_first_ind in anchor_positions:
                        if anchor_first_ind - 1 in ori_match_ids:
                            # do not reduce probability for 1st appearance
                            if tofu and all([aem[0] >= anchor_first_ind for aem in target_ent_mention_spans]):
                                continue
                            modify_index = ori_match_ids.index(anchor_first_ind - 1)
                            move_probability_to_original_name(match_probs, modify_index, original_ids[anchor_first_ind], replace_person_first_token_ids[adjust_set_ind]['all'])
                assert np.allclose(match_probs.sum(axis=-1), 1)
                tmp_dict[adjust_ent] = {'matched_original_ids_index': np.array(ori_match_ids), 'match_probs': match_probs}
            
            # get the intersection of matched_original_ids_index
            reduced_original_ids_index = reduce(np.intersect1d, [v['matched_original_ids_index'] for v in tmp_dict.values()])
            if len(reduced_original_ids_index) == 0:
                # 不让整个程序退出，只跳过当前 content
                print(f"Error in processing {title}: empty reduced_original_ids_index, skip this content chunk.")
                continue
            reduced_original_ids = np.array(original_ids)[reduced_original_ids_index]
            # 使用实际的替换数量
            actual_n = len(replace_persons)
            assert len(tmp_dict) == actual_n
            for k, v in tmp_dict.items():
                # reduce probs to the matched indices
                matched_original_ids_index = v['matched_original_ids_index']
                probs_i = v['match_probs']
                # get the index of reduced_original_ids_index in matched_original_ids_index
                idx = np.searchsorted(matched_original_ids_index, reduced_original_ids_index)
                assert np.all(matched_original_ids_index[idx] == reduced_original_ids_index)
                tmp_dict[k]['match_probs'] = probs_i[idx]
            
            if not teacher_cfg.whp_baseline and teacher_cfg.change_name_back:
                # check if there are positions where most of teachers predict the replaced entity
                original_w_prefix = tokenizer(title, add_special_tokens=False)['input_ids'][0]
                original_wo_prefix = tokenizer(f"\n{title}", add_special_tokens=False)['input_ids'][2]
                assert title.startswith(tokenizer.decode(original_w_prefix, skip_special_tokens=True)) and title.startswith(tokenizer.decode(original_wo_prefix, skip_special_tokens=True))
                for i in range(len(reduced_original_ids_index)):
                    move_threshold = 0.1
                    n_w_prefix = sum([any([tmp_dict[name]['match_probs'][i, first_tok] > move_threshold for first_tok in replace_person_first_token_ids[adjust_ind]['w_prefix']]) for adjust_ind, name in enumerate(replace_persons)])
                    n_wo_prefix = sum([any([tmp_dict[name]['match_probs'][i, first_tok] > move_threshold for first_tok in replace_person_first_token_ids[adjust_ind]['wo_prefix']]) for adjust_ind, name in enumerate(replace_persons)])
                    actual_n = len(replace_persons)
                    if n_w_prefix > 0.5 * actual_n:
                        gt_token_id = original_w_prefix
                    elif n_wo_prefix > 0.5 * actual_n:
                        gt_token_id = original_wo_prefix
                    else:
                        continue
                    # do not move probability for the first appearance
                    prefix = tokenizer.decode(original_ids[:reduced_original_ids_index[i]], skip_special_tokens=True)
                    if not teacher_cfg.counter_fact_prompt and not any([ae in prefix for ae in target_ents]):
                        continue
                    for adjust_ind, name in enumerate(replace_persons):
                        move_probability_to_original_name(tmp_dict[name]['match_probs'], i, gt_token_id, replace_person_first_token_ids[adjust_ind]['all'])
            # get the weighted average of match_probs
            weighted_avg_probs = np.stack([v['match_probs'] for v in tmp_dict.values()], axis=0)
            actual_n = len(replace_persons)
            if actual_n == 1:
                weighted_avg_probs = weighted_avg_probs[0]
            else:
                weighted_avg_probs = weighted_avg_probs.mean(axis=0)
            
            if not tofu and content_ind == 0:
                # do not train on pronounciations
                non_ascii_spans = find_non_ascii(content)
                for i in range(1, len(original_ids)):
                    tmp_text = tokenizer.decode(original_ids[i:], skip_special_tokens=True)
                    for span in non_ascii_spans:
                        if tmp_text.startswith(span):
                            for j in range(i+1, len(original_ids)):
                                if tokenizer.decode(original_ids[i:j], skip_special_tokens=True) not in span:
                                    break
                            # remove indices i-1 to j-2
                            indices_to_keep = [keep_i for keep_i, v in enumerate(reduced_original_ids_index) if v < i-1 or v >= j-2]
                            reduced_original_ids_index = reduced_original_ids_index[indices_to_keep]
                            reduced_original_ids = reduced_original_ids[indices_to_keep]
                            weighted_avg_probs = weighted_avg_probs[indices_to_keep]
                            break
            
            if forget_target_name:
                forget_target_ent_positions = [i for i in target_ent_positions if all([occ.lower() not in i['prefix'].lower() for occ in i['all_occurrences']]) and '[/INST]' in i['prefix']]
                for target_i in forget_target_ent_positions:
                    original_ids_index_i, weighted_avg_probs_i, original_ids_i = get_target_name_teacher(content, adjust_list, model, tokenizer, teacher_cfg.N, target_i['mention'], replace_anchor_person=target_i['mention'] not in target_ents)
                    reduced_original_ids_index = np.concatenate([reduced_original_ids_index, original_ids_index_i])
                    weighted_avg_probs = np.concatenate([weighted_avg_probs, weighted_avg_probs_i], axis=0)
                    reduced_original_ids = np.concatenate([reduced_original_ids, original_ids_i])
            
            if teacher_cfg.whp_baseline:
                # convert to one-hot probability
                predict_token = np.argmax(weighted_avg_probs, axis=-1)
                weighted_avg_probs = np.zeros_like(weighted_avg_probs)
                weighted_avg_probs[np.arange(predict_token.shape[0]), predict_token] = 1
            assert len(reduced_original_ids_index) == len(np.unique(reduced_original_ids_index))
            assert np.allclose(weighted_avg_probs.sum(axis=-1), 1)
            assert len(reduced_original_ids_index) == len(reduced_original_ids) == len(weighted_avg_probs)
            item_results.append({'original_ids_index': reduced_original_ids_index, 'weighted_avg_probs': weighted_avg_probs, 'original_ids': reduced_original_ids, 'anchor_entities': target_ents, 'anchor_ent_mentions': target_ent_mention_spans})
            
            if teacher_cfg.verbose:
                predicted = np.argsort(weighted_avg_probs, axis=-1)[:, ::-1]
                total_nll, tok_cnt = 0, 0
                for i in range(len(reduced_original_ids_index)):
                    print(f"\nOriginal input: {tokenizer.decode(original_ids[:reduced_original_ids_index[i]+2], skip_special_tokens=True)}\n")
                    print(f"Top 5 at position {i}: {[(tokenizer.decode([predicted[i, j].item()]), round(weighted_avg_probs[i, predicted[i, j]].item(), 4)) for j in range(5)]}\n")
                    if reduced_original_ids_index[i]+1 < len(original_ids):
                        nll_i = -np.log(weighted_avg_probs[i, original_ids[reduced_original_ids_index[i]+1]])
                        total_nll += nll_i
                        tok_cnt += 1
                        print(f"Neg log prob of GT token: {nll_i}")
            else:
                print(f"\nOriginal input: {tokenizer.decode(original_ids, skip_special_tokens=True)}\n")
                print(f"\nMatched original input: {tokenizer.decode([original_ids[i] for i in reduced_original_ids_index], skip_special_tokens=True)}\n")
                # weighted_avg_probs 是 numpy 数组，这里用 numpy 的 argmax 即可
                predicted = np.argmax(weighted_avg_probs, axis=-1)
                print(f"\nPredicted: {tokenizer.decode(predicted.tolist(), skip_special_tokens=True)}\n")
        
        results[title][result_key] = item_results
    
    with open(save_name, "wb") as f:
        pickle.dump(results, f)


def get_target_name_teacher(content, adjust_list, model, tokenizer, N, target_forget_name, replace_anchor_person=True):
    original_ids = tokenizer(content, add_special_tokens=True)['input_ids']
    target_forget_name_word = target_forget_name.split()
    target_name_num_word = len(target_forget_name_word)
    occur_index = content.find(target_forget_name)
    if content[occur_index-1] != ' ':
        prepend_char = content[occur_index-1]
    else:
        prepend_char = ''
    only_consider_spans = get_ent_indices([target_forget_name], original_ids, tokenizer)
    assert len(only_consider_spans) == 1
    only_consider_indices = list(range(only_consider_spans[0][0], only_consider_spans[0][1]))
    # get indices for each word
    target_name_tokenized = tokenizer((prepend_char+target_forget_name).split(), add_special_tokens=False, is_split_into_words=True)
    word_ids = target_name_tokenized.word_ids()
    start_ind = [i for i in range(len(word_ids)) if target_name_tokenized.input_ids[i:] == [original_ids[j] for j in only_consider_indices]]
    assert len(start_ind) == 1
    word_ids = word_ids[start_ind[0]:]
    assert len(set(word_ids)) == target_name_num_word
    word_ids_len = [sum([i == w_id for i in word_ids]) for w_id in range(target_name_num_word)]
    only_consider_word_indices = [only_consider_indices[:word_ids_len[0]-1]]
    cum_sum = word_ids_len[0] - 1
    for i in range(1, target_name_num_word):
        only_consider_word_indices.append(only_consider_indices[cum_sum:cum_sum+word_ids_len[i]])
        cum_sum += word_ids_len[i]
    # no context for the target name
    target_start_index = [i for i in range(len(original_ids)) if tokenizer.decode(original_ids[i:], skip_special_tokens=True) == content[occur_index:]]
    assert len(target_start_index) == 1
    target_start_index = target_start_index[0]
    target_start_indices = [target_start_index + sum(word_ids_len[:i]) for i in range(target_name_num_word)]
    original_ids_by_word = [original_ids[target_start_indices[i]:target_start_indices[i]+word_ids_len[i]] for i in range(target_name_num_word)]
    
    tmp_dict = dict()
    perturbed_contents, num_added_tokens = [], []
    for adjust_set in adjust_list:
        for replace_first_n in range(target_name_num_word):
            perturbed_content = copy.deepcopy(content)
            # replace anchor entity with adjust_ent
            # assume 1st adjust_ent is the anchor entity
            if replace_anchor_person:
                replace_item = [adjust_i for adjust_i in adjust_set if target_forget_name in adjust_i[0]]
                assert len(replace_item) == 1
                perturbed_content = replace_name_only_first_n(replace_item[0][0][0], replace_item[0][1], perturbed_content, replace_first_n, strict=True, no_context=True)
            else:
                perturbed_content = replace_name_only_first_n(target_forget_name, adjust_set[0][1], perturbed_content, replace_first_n, strict=False, no_context=True)
            
            perturbed_content = f'[INST] Complete the following name. [/INST] {prepend_char}{perturbed_content}'
            perturbed_contents.append(perturbed_content)
            if '[/INST]' in perturbed_content:
                added_content = perturbed_content.split('[/INST]')[0] + '[/INST]'
                num_added_tok = len(tokenizer(added_content, add_special_tokens=True)['input_ids'])
            else:
                num_added_tok = 1
            num_added_tokens.append(num_added_tok)
    intervened_inputs = tokenizer(perturbed_contents, add_special_tokens=True, return_tensors='pt', padding=True)
    with torch.no_grad():
        outs = model(**intervened_inputs.to('cuda'))
    for adjust_set_ind, adjust_set in enumerate(adjust_list):
        all_origin_match_ids, all_match_probs = [], []
        for target_name_word_ind, input_idx in enumerate(range(adjust_set_ind*target_name_num_word, (adjust_set_ind+1)*target_name_num_word)):
            check_inds = only_consider_word_indices[target_name_word_ind]
            prediction_word = perturbed_contents[input_idx].split()[-1].replace('[/INST]', '').strip(prepend_char).strip()
            if len(check_inds) == 0:
                assert target_name_word_ind == 0
                continue
            replace_ids = intervened_inputs['input_ids'][input_idx].tolist()[num_added_tokens[input_idx]:]
            matcher = difflib.SequenceMatcher(None, original_ids_by_word[target_name_word_ind], replace_ids)
            blocks = matcher.get_matching_blocks()
            blocks = [b for b in blocks if b.size > 0]
            ori_match_ids, rep_match_ids = [], []
            for block in blocks:
                if prediction_word not in tokenizer.decode(replace_ids[block.b: block.b+block.size], skip_special_tokens=True).split()[-1]:
                    continue
                if block.a - 1 + target_start_indices[target_name_word_ind] in check_inds:
                    # still want to train on 1st token of the word, the 1st token of the target name has been included outside
                    a_start = block.a - 1
                    b_start = block.b - 1
                else:
                    a_start = block.a
                    b_start = block.b
                ori_match_ids.extend(list(range(a_start + target_start_indices[target_name_word_ind], block.a + block.size + target_start_indices[target_name_word_ind])))
                rep_match_ids.extend(list(range(b_start + num_added_tokens[input_idx], num_added_tokens[input_idx] + block.b + block.size)))
            # get the matched probabilities
            indices = [idx for idx, v in enumerate(ori_match_ids) if v in check_inds]
            if len(indices) == 0:
                assert len(check_inds) == 1
                assert tokenizer.decode(original_ids[check_inds[0]+1:check_inds[0]+2], skip_special_tokens=True).lower() in stop_word_list
                continue
            ori_match_ids = [ori_match_ids[idx] for idx in indices]
            rep_match_ids = [rep_match_ids[idx] for idx in indices]
            assert len(rep_match_ids) == len(set(rep_match_ids))
            # logits are bfloat16; cast to float32 before softmax / numpy
            match_logits = outs.logits[input_idx][rep_match_ids, :].float()  # (L, V)
            match_probs = torch.nn.functional.softmax(match_logits, dim=-1).cpu().numpy()
            all_origin_match_ids += ori_match_ids
            all_match_probs.append(match_probs)
        assert len(all_origin_match_ids) == len(set(all_origin_match_ids))
        all_match_probs = np.concatenate(all_match_probs, axis=0)
        assert np.allclose(all_match_probs.sum(axis=-1), 1)
        tmp_dict[adjust_set[0][1]] = {'matched_original_ids_index': np.array(all_origin_match_ids), 'match_probs': all_match_probs}
    
    assert len(tmp_dict) == N
    # get the intersection of matched_original_ids_index
    reduced_original_ids_index = reduce(np.intersect1d, [v['matched_original_ids_index'] for v in tmp_dict.values()])
    if len(reduced_original_ids_index) == 0:
        # 如果这里没有交集，说明这个 target 名字没法稳定对齐，返回空结果而不是直接退出程序
        print(f"Error in processing {content} in get_target_name_teacher; skip this target.")
        vocab_size = outs.logits.shape[-1]
        return np.array([], dtype=int), np.zeros((0, vocab_size), dtype=float), np.array([], dtype=int)
    reduced_original_ids = np.array(original_ids)[reduced_original_ids_index]
    for k, v in tmp_dict.items():
        # reduce probs to the matched indices
        matched_original_ids_index = v['matched_original_ids_index']
        probs_i = v['match_probs']
        # get the index of reduced_original_ids_index in matched_original_ids_index
        idx = np.searchsorted(matched_original_ids_index, reduced_original_ids_index)
        assert np.all(matched_original_ids_index[idx] == reduced_original_ids_index)
        tmp_dict[k]['match_probs'] = probs_i[idx]
    # get the weighted average of match_probs
    weighted_avg_probs = np.stack([v['match_probs'] for v in tmp_dict.values()], axis=0)
    if N == 1:
        weighted_avg_probs = weighted_avg_probs[0]
    else:
        weighted_avg_probs = weighted_avg_probs.mean(axis=0)
    return reduced_original_ids_index, weighted_avg_probs, reduced_original_ids


if __name__ == "__main__":
    print("###############################")
    print("Constructing teacher distribution")
    print("###############################")
    main()