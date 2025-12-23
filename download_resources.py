#!/usr/bin/env python
"""
下载 teacher.py 所需的所有离线资源到项目目录下的 local_resources 文件夹。

使用方法（在有网络的机器上运行）:
    python download_resources.py --model_id <模型名称>

示例:
    python download_resources.py --model_id mistralai/Mistral-7B-Instruct-v0.3
    python download_resources.py --model_id meta-llama/Llama-2-7b-chat-hf

下载完成后，将整个 local_resources/ 目录上传到服务器的项目根目录下。

资源会下载到以下位置（与 teacher.py 中的路径配置一致）:
    <项目目录>/local_resources/
        ├── nltk_data/          # NLTK 数据
        ├── spacy_models/       # spaCy 模型  
        │   └── en_core_web_sm/
        ├── stanza_resources/   # Stanza 资源
        └── huggingface_models/ # HuggingFace 模型
"""

import os
import argparse

# 获取项目根目录（download_resources.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 默认资源保存目录
DEFAULT_RESOURCES_DIR = os.path.join(PROJECT_ROOT, "local_resources")


def download_nltk_data(save_dir):
    """下载 NLTK 数据"""
    import nltk
    nltk_dir = os.path.join(save_dir, "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    
    # 检查各个资源是否已存在
    stopwords_path = os.path.join(nltk_dir, "corpora", "stopwords")
    punkt_path = os.path.join(nltk_dir, "tokenizers", "punkt")
    punkt_tab_path = os.path.join(nltk_dir, "tokenizers", "punkt_tab")
    
    all_exist = all([
        os.path.exists(stopwords_path),
        os.path.exists(punkt_path),
        os.path.exists(punkt_tab_path)
    ])
    
    if all_exist:
        print(f"NLTK 数据已存在于 {nltk_dir}，跳过下载。")
        return
    
    print(f"正在下载 NLTK 数据到 {nltk_dir}...")
    if not os.path.exists(stopwords_path):
        nltk.download('stopwords', download_dir=nltk_dir)
    else:
        print("  stopwords 已存在，跳过。")
    
    if not os.path.exists(punkt_path):
        nltk.download('punkt', download_dir=nltk_dir)
    else:
        print("  punkt 已存在，跳过。")
    
    if not os.path.exists(punkt_tab_path):
        nltk.download('punkt_tab', download_dir=nltk_dir)
    else:
        print("  punkt_tab 已存在，跳过。")
    
    print("NLTK 数据下载完成！")


def download_spacy_model(save_dir):
    """下载 spaCy 模型"""
    spacy_dir = os.path.join(save_dir, "spacy_models")
    target_path = os.path.join(spacy_dir, "en_core_web_sm")
    
    # 检查模型是否已存在（通过检查 meta.json 文件）
    meta_file = os.path.join(target_path, "meta.json")
    if os.path.exists(meta_file):
        print(f"spaCy 模型已存在于 {target_path}，跳过下载。")
        return
    
    import spacy
    from spacy.cli import download
    import shutil
    
    os.makedirs(spacy_dir, exist_ok=True)
    
    print(f"正在下载 spaCy 模型...")
    # 先下载模型
    download("en_core_web_sm")
    
    # 获取安装位置并复制到本地目录
    import en_core_web_sm
    
    model_path = os.path.dirname(en_core_web_sm.__file__)
    
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(model_path, target_path)
    print(f"spaCy 模型已保存到 {target_path}")


def download_stanza_resources(save_dir, force_redownload=False, skip_md5_check=False):
    """下载 Stanza 资源"""
    import shutil
    stanza_dir = os.path.join(save_dir, "stanza_resources")
    
    # 如果强制重新下载，先清理目录
    if force_redownload and os.path.exists(stanza_dir):
        print(f"强制重新下载，清理 {stanza_dir}...")
        shutil.rmtree(stanza_dir)
    
    # 检查资源是否已存在（通过检查 resources.json 和 en 目录）
    resources_file = os.path.join(stanza_dir, "resources.json")
    en_dir = os.path.join(stanza_dir, "en")
    if os.path.exists(resources_file) and os.path.exists(en_dir) and not force_redownload:
        print(f"Stanza 资源已存在于 {stanza_dir}，跳过下载。")
        return
    
    import stanza
    from stanza.resources import common as stanza_common
    
    os.makedirs(stanza_dir, exist_ok=True)
    
    print(f"正在下载 Stanza 资源到 {stanza_dir}...")
    
    if skip_md5_check:
        print("注意: 已启用跳过 MD5 校验模式")
        # 保存原始的 assert_file_exists 函数
        original_assert_file_exists = stanza_common.assert_file_exists
        
        # 创建一个跳过 MD5 检查的替代函数
        def patched_assert_file_exists(path, md5=None, alternate_md5=None):
            """跳过 MD5 检查，只验证文件存在"""
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
            print(f"  已下载: {os.path.basename(path)} (跳过 MD5 校验)")
        
        # 临时替换函数
        stanza_common.assert_file_exists = patched_assert_file_exists
        
        try:
            stanza.download('en', model_dir=stanza_dir, processors='tokenize,mwt,ner,coref')
            print("Stanza 资源下载完成！")
        finally:
            # 恢复原始函数
            stanza_common.assert_file_exists = original_assert_file_exists
    else:
        try:
            stanza.download('en', model_dir=stanza_dir, processors='tokenize,mwt,ner,coref')
            print("Stanza 资源下载完成！")
        except ValueError as e:
            if "md5" in str(e).lower():
                print(f"\nMD5 校验失败: {e}")
                print("\n这是 Stanza 的一个已知问题，服务器上的文件与元数据不匹配。")
                print("请使用 --skip_stanza_md5 参数跳过 MD5 校验:")
                print("  python download_resources.py --skip_nltk --skip_spacy --skip_model --skip_stanza_md5")
                raise
            else:
                raise


def download_huggingface_model(model_id, save_dir):
    """下载 Hugging Face 模型和 Tokenizer"""
    # 处理模型名称中的斜杠
    model_name = model_id.replace("/", "_")
    model_dir = os.path.join(save_dir, "huggingface_models", model_name)
    
    # 检查模型是否已存在（通过检查 config.json 文件）
    config_file = os.path.join(model_dir, "config.json")
    if os.path.exists(config_file):
        print(f"Hugging Face 模型已存在于 {model_dir}，跳过下载。")
        return
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"正在下载 Hugging Face 模型: {model_id}...")
    print(f"保存路径: {model_dir}")
    print("（这可能需要较长时间，取决于模型大小和网络速度）")
    
    # 下载 tokenizer
    print("下载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(model_dir)
    
    # 下载模型
    print("下载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.save_pretrained(model_dir)
    
    print(f"模型已保存到 {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载 teacher.py 所需的离线资源")
    parser.add_argument("--model_id", type=str, default=None,
                        help="要下载的 Hugging Face 模型 ID，例如 'mistralai/Mistral-7B-Instruct-v0.3'")
    parser.add_argument("--save_dir", type=str, default=None,
                        help=f"资源保存目录（默认: {DEFAULT_RESOURCES_DIR}）")
    parser.add_argument("--skip_nltk", action="store_true", help="跳过 NLTK 数据下载")
    parser.add_argument("--skip_spacy", action="store_true", help="跳过 spaCy 模型下载")
    parser.add_argument("--skip_stanza", action="store_true", help="跳过 Stanza 资源下载")
    parser.add_argument("--skip_model", action="store_true", help="跳过 Hugging Face 模型下载")
    parser.add_argument("--force_stanza", action="store_true", help="强制重新下载 Stanza 资源（清理现有文件）")
    parser.add_argument("--skip_stanza_md5", action="store_true", help="跳过 Stanza 资源的 MD5 校验（解决服务器文件与元数据不匹配的问题）")
    
    args = parser.parse_args()
    
    # 使用默认项目目录下的 local_resources，或用户指定的目录
    if args.save_dir is None:
        save_dir = DEFAULT_RESOURCES_DIR
    else:
        save_dir = os.path.abspath(args.save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"资源将保存到: {save_dir}")
    
    # 下载各类资源
    if not args.skip_nltk:
        download_nltk_data(save_dir)
    
    if not args.skip_spacy:
        download_spacy_model(save_dir)
    
    if not args.skip_stanza:
        download_stanza_resources(save_dir, force_redownload=args.force_stanza, skip_md5_check=args.skip_stanza_md5)
    
    if not args.skip_model and args.model_id:
        download_huggingface_model(args.model_id, save_dir)
    elif not args.skip_model and not args.model_id:
        print("\n未指定 --model_id，跳过 Hugging Face 模型下载。")
        print("如需下载模型，请使用: python download_resources.py --model_id <模型名称>")
    
    print("\n" + "="*60)
    print("下载完成！")
    print("="*60)
    print(f"\n资源目录结构:")
    print(f"  {save_dir}/")
    print(f"    ├── nltk_data/          # NLTK 数据")
    print(f"    ├── spacy_models/       # spaCy 模型")
    print(f"    │   └── en_core_web_sm/")
    print(f"    ├── stanza_resources/   # Stanza 资源")
    if args.model_id:
        model_name = args.model_id.replace("/", "_")
        print(f"    └── huggingface_models/")
        print(f"        └── {model_name}/  # HF 模型")
    else:
        print(f"    └── huggingface_models/  # (未下载模型)")
    
    print(f"\n路径配置（与 teacher.py 一致）:")
    print(f"  NLTK 数据:     {os.path.join(save_dir, 'nltk_data')}")
    print(f"  spaCy 模型:    {os.path.join(save_dir, 'spacy_models', 'en_core_web_sm')}")
    print(f"  Stanza 资源:   {os.path.join(save_dir, 'stanza_resources')}")
    print(f"  HF 模型目录:   {os.path.join(save_dir, 'huggingface_models')}")
    
    print(f"\n如需上传到服务器，将整个 local_resources/ 目录复制到服务器的项目根目录下即可。")


if __name__ == "__main__":
    main()

