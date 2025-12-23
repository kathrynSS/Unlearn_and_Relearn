import pandas as pd
import random
import math
from pathlib import Path

# ============ 参数设置 ============
base_dir = Path(__file__).parent
input_path = base_dir / "raw_data" /"all_questions_updated.xlsx"  # 原始数据
group_col = "question_type"               # 知识点列名

# 选取“题目数最多”的前 60% 知识点作为 A，其余 40% 作为 B
top_ratio_kp = 0.6

# 在 A 内部，对每个知识点按题目顺序做 10% 训练(A1) / 50% 遗忘(A2) / 40% 测试(A3)
train_ratio_q = 0.1
forget_ratio_q = 0.5
# 剩余约 40% 题目做测试(A3)

# ============ 读数据 ============
df = pd.read_excel(input_path)
df = df.reset_index(drop=True)

# ============ 1. 统计每个知识点的题目数量，并按数量从多到少排序 ============
kp_counts = df[group_col].value_counts()  # index: 知识点; value: 题目数
sorted_kps = list(kp_counts.index)        # 已按数量从多到少排序
num_kps = len(sorted_kps)

# 选取“题目数最多”的前 60% 知识点作为 A，其余作为 B
num_top_kps = max(1, round(num_kps * top_ratio_kp))
top_kps = set(sorted_kps[:num_top_kps])          # A 中的知识点
other_kps = set(sorted_kps[num_top_kps:])        # B 中的知识点

# A：用于按 10/50/40 再划分
selected_df = df[df[group_col].isin(top_kps)].copy()

# B：直接全部并入训练集
other_df = df[df[group_col].isin(other_kps)].copy()

# ============ 2. 在 A 中，每个知识点内部按 10% 训练(A1) / 50% 遗忘(A2) / 40% 测试(A3) ============
selected_df["idx_in_kp"] = selected_df.groupby(group_col).cumcount()
selected_df["kp_size"] = selected_df.groupby(group_col)[group_col].transform("size")
selected_df["ratio_in_kp"] = (selected_df["idx_in_kp"] + 1) / selected_df["kp_size"]

train_mask = selected_df["ratio_in_kp"] <= train_ratio_q
forget_mask = (selected_df["ratio_in_kp"] > train_ratio_q) & (
    selected_df["ratio_in_kp"] <= (train_ratio_q + forget_ratio_q)
)
test_mask = selected_df["ratio_in_kp"] > (train_ratio_q + forget_ratio_q)

train_A1 = selected_df[train_mask].copy()
forget_A2 = selected_df[forget_mask].copy()
test_A3 = selected_df[test_mask].copy()

# 去掉辅助列（只存在于 A 中）
for col in ["idx_in_kp", "kp_size", "ratio_in_kp"]:
    for df_part in (train_A1, forget_A2, test_A3):
        if col in df_part.columns:
            df_part.drop(columns=[col], inplace=True)

# ============ 3. 组合最终的训练/遗忘/测试集 ============
# 训练集 = A1 + B；遗忘集 = A2；测试集 = A3
train_df = pd.concat([train_A1, other_df], ignore_index=True)
forget_df = forget_A2
test_df = test_A3

# ============ 4. 保存结果 ============

test_df.to_excel(base_dir / "raw_data" / "questions_test.xlsx", index=False)
forget_df.to_excel(base_dir / "raw_data" / "questions_forget.xlsx", index=False)
train_df.to_excel( base_dir / "raw_data" / "questions_train.xlsx", index=False)

# ============ 4. 保存遗忘知识点到一个 TXT ============
forget_kps = sorted(forget_df[group_col].unique())
with open(base_dir / "raw_data" / "forgotten_knowledge_points.txt", "w", encoding="utf-8") as f:
    for kp in forget_kps:
        f.write(str(kp) + "\n")

print("数据划分完成!")
print("知识点总数：", num_kps)
print("被选中（题目数最多）用于划分的知识点数：", len(top_kps))
print("训练集题目数：", len(train_df))
print("遗忘集题目数：", len(forget_df))
print("测试集题目数：", len(test_df))
print("遗忘知识点已保存到 forgotten_knowledge_points.txt")
