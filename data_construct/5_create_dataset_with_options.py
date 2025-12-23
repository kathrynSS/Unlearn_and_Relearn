"""
使用错误选项作为替换的数据集构造脚本（N=3）
这个方案更符合多选题的特点
"""
import pandas as pd
import json
import random
from datasets import Dataset, DatasetDict
from pathlib import Path

random_seed = 42
random.seed(random_seed)

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.resolve()

# 输入文件（相对于脚本目录）
FORGET_PATH = SCRIPT_DIR / "raw_data/questions_forget.xlsx"
TRAIN_PATH = SCRIPT_DIR / "raw_data/questions_train.xlsx"
TEST_PATH = SCRIPT_DIR / "raw_data/questions_test.xlsx"
FORGOTTEN_KP_PATH = SCRIPT_DIR / "raw_data/forgotten_knowledge_points.txt"

# 输出路径（相对于脚本目录）
OUTPUT_DIR = SCRIPT_DIR / "data/data_ai_v2"
REPLACE_KP_FILE = SCRIPT_DIR / "data/replace_knowledge_points_v2.json"
KP_LIST_FILE = SCRIPT_DIR / "data/ai_knowledge_points.txt"


def load_data():
    """加载所有数据集"""
    forget_df = pd.read_excel(FORGET_PATH)
    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)
    
    with open(FORGOTTEN_KP_PATH, 'r', encoding='utf-8') as f:
        forgotten_kps = [line.strip() for line in f if line.strip()]
    
    return forget_df, train_df, test_df, forgotten_kps


def extract_wrong_options(row):
    """
    从一道题中提取错误选项
    返回3个错误答案
    """
    correct_letter = row['answer_letter'].strip()
    options_str = row['options']
    
    # 解析选项字符串 "A. xxx | B. yyy | C. zzz | D. www"
    options = {}
    for opt in options_str.split('|'):
        opt = opt.strip()
        if opt and len(opt) > 2:
            letter = opt[0]
            text = opt[3:].strip()  # 去掉 "A. "
            options[letter] = text
    
    # 获取错误选项
    wrong_options = []
    for letter in ['A', 'B', 'C', 'D']:
        if letter != correct_letter and letter in options:
            wrong_options.append(f"{letter}. {options[letter]}")
    
    return wrong_options[:3]  # 确保只返回3个


def create_knowledge_point_replacements_from_options(forget_df, forgotten_kps):
    """
    为每个知识点创建替换映射
    使用该知识点下所有题目的错误选项作为替换池
    
    返回格式: {知识点: [错误答案1, 错误答案2, 错误答案3, ...]}
    """
    replace_kp_dict = {}
    
    for kp in forgotten_kps:
        # 获取该知识点的所有题目
        kp_questions = forget_df[forget_df['question_type'] == kp]
        
        # 收集所有错误选项
        all_wrong_options = []
        for _, row in kp_questions.iterrows():
            wrong_opts = extract_wrong_options(row)
            all_wrong_options.extend(wrong_opts)
        
        # 去重并打乱
        all_wrong_options = list(set(all_wrong_options))
        random.shuffle(all_wrong_options)
        
        # 至少保留3个不同的错误答案
        if len(all_wrong_options) >= 3:
            replace_kp_dict[kp] = all_wrong_options
        else:
            # 如果不够3个，重复使用
            replace_kp_dict[kp] = all_wrong_options * 2
            replace_kp_dict[kp] = replace_kp_dict[kp][:3]
        
        print(f"  {kp}: 收集了 {len(all_wrong_options)} 个错误选项")
    
    return replace_kp_dict


def convert_to_qa_format(df, add_title=False):
    """将DataFrame转换为question-answer格式"""
    df = df.copy()
    df['answer'] = df['answer_letter'].astype(str).str.strip() + ". " + df['answer_text'].astype(str).str.strip()
    
    if add_title:
        df['title'] = df['question_type']
        df_converted = df[['question', 'answer', 'title']].copy()
    else:
        df_converted = df[['question', 'answer']].copy()
    
    return df_converted


def create_perturbed_dataset(df, replace_kp_dict, is_forget=True):
    """创建perturbed数据集"""
    df = df.copy()
    df['paraphrased_answer'] = df['answer']
    
    if is_forget:
        perturbed_answers = []
        for idx, row in df.iterrows():
            kp = row['title']
            if kp in replace_kp_dict and len(replace_kp_dict[kp]) > 0:
                # 随机选择一个错误答案作为扰动
                perturbed_answer = random.choice(replace_kp_dict[kp])
                perturbed_answers.append(perturbed_answer)
            else:
                perturbed_answers.append(row['answer'])
        df['perturbed_answer'] = perturbed_answers
    else:
        df['perturbed_answer'] = df['answer']
    
    return df


def main():
    print("="*60)
    print("创建基于错误选项的数据集（N由实际选项数量决定）")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1/6] 加载原始数据...")
    forget_df, train_df, test_df, forgotten_kps = load_data()
    print(f"  ✓ forget数据: {len(forget_df)} 条")
    print(f"  ✓ train数据: {len(train_df)} 条")
    print(f"  ✓ test数据: {len(test_df)} 条")
    print(f"  ✓ 遗忘知识点: {len(forgotten_kps)} 个")
    
    # 2. 从错误选项创建替换映射
    print("\n[2/6] 从错误选项创建替换映射...")
    replace_kp_dict = create_knowledge_point_replacements_from_options(
        forget_df, forgotten_kps
    )
    
    # 统计每个知识点的替换数量
    replacement_counts = {kp: len(v) for kp, v in replace_kp_dict.items()}
    avg_n = sum(replacement_counts.values()) / len(replacement_counts)
    print(f"\n  平均每个知识点有 {avg_n:.1f} 个替换")
    print(f"  最少: {min(replacement_counts.values())}，最多: {max(replacement_counts.values())}")
    
    # 3. 保存替换映射
    print("\n[3/6] 保存配置文件...")
    (SCRIPT_DIR / "data").mkdir(exist_ok=True)
    
    with open(REPLACE_KP_FILE, 'w', encoding='utf-8') as f:
        json.dump(replace_kp_dict, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 保存到: {REPLACE_KP_FILE}")
    
    # 知识点列表保持不变
    with open(KP_LIST_FILE, 'w', encoding='utf-8') as f:
        for kp in forgotten_kps:
            f.write(kp + '\n')
    print(f"  ✓ 保存到: {KP_LIST_FILE}")
    
    # 4. 创建数据集splits
    print("\n[4/6] 创建数据集splits...")
    forget_data = convert_to_qa_format(forget_df, add_title=True)
    train_converted = convert_to_qa_format(train_df, add_title=True)
    retain_data = train_converted[~train_converted['title'].isin(forgotten_kps)].copy()
    
    forget_perturbed = create_perturbed_dataset(forget_data, replace_kp_dict, is_forget=True)
    
    test_converted = convert_to_qa_format(test_df, add_title=True)
    retain_test = test_converted[~test_converted['title'].isin(forgotten_kps)].copy()
    retain_perturbed = create_perturbed_dataset(retain_test, replace_kp_dict, is_forget=False)
    
    print(f"  forget: {len(forget_data)} 条")
    print(f"  retain: {len(retain_data)} 条")
    print(f"  forget_perturbed: {len(forget_perturbed)} 条")
    print(f"  retain_perturbed: {len(retain_perturbed)} 条")
    
    # 5. 保存数据集
    print("\n[5/6] 保存数据集...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets_to_save = {
        'forget': Dataset.from_pandas(forget_data.reset_index(drop=True)),
        'retain': Dataset.from_pandas(retain_data.reset_index(drop=True)),
        'forget_perturbed': Dataset.from_pandas(forget_perturbed.reset_index(drop=True)),
        'retain_perturbed': Dataset.from_pandas(retain_perturbed.reset_index(drop=True)),
    }
    
    dataset_dict = DatasetDict(datasets_to_save)
    dataset_dict.save_to_disk(str(OUTPUT_DIR))
    print(f"  ✓ 保存到: {OUTPUT_DIR}")
    
    # 6. 显示示例
    print("\n[6/6] 验证数据集...")
    from datasets import load_from_disk
    loaded = load_from_disk(str(OUTPUT_DIR))
    
    print("\n" + "="*60)
    print("替换映射示例:")
    print("="*60)
    first_kp = list(replace_kp_dict.keys())[0]
    print(f"\n知识点: {first_kp}")
    print(f"错误选项替换 (共{len(replace_kp_dict[first_kp])}个):")
    for i, opt in enumerate(replace_kp_dict[first_kp][:5], 1):
        print(f"  {i}. {opt}")
    
    print("\n" + "="*60)
    print("✅ 数据集创建完成！")
    print("="*60)
    print(f"\n📁 数据集目录: {OUTPUT_DIR}")
    print(f"📄 替换映射: {REPLACE_KP_FILE}")
    print(f"\n💡 这个版本使用了实际的错误选项作为替换")
    print(f"   每个知识点的N值由该知识点的错误选项数量决定")
    print(f"   平均N ≈ {avg_n:.0f}")
    
    print("\n下一步:")
    print("  1. 更新配置文件中的 data_path 为 data/data_ai_v2")
    print("  2. 运行 teacher.py 时，N值会根据实际替换数量自动确定")


if __name__ == "__main__":
    main()

