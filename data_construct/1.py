import pandas as pd
from pathlib import Path

# 1. 读入原始文件（使用脚本所在目录作为相对路径基准，避免工作目录不同导致找不到文件）
base_dir = Path(__file__).parent
file_path = base_dir / "raw_data" / "Python_Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

# 2. 将 answer_letter + answer_text 拼成完整答案
#    去掉两边空格，并处理缺失值
df["answer_letter"] = (
    df["answer_letter"].astype(str).str.strip()  # A/B/C/D
    + ". "
    + df["answer_text"].astype(str).str.strip()  # Guido van Rossum
)

# 如果你想保留原来的字母列，可以先拷贝一份：
# df["answer_letter_original"] = df["answer_letter"]
# 然后再生成新的完整答案列：
# df["answer_full"] = (
#     df["answer_letter_original"].astype(str).str.strip()
#     + ". "
#     + df["answer_text"].astype(str).str.strip()
# )

# 3. 保存为新文件（同样保存到脚本所在目录下的 raw_data）
output_path = base_dir / "raw_data" / "all_questions_updated.xlsx"
df.to_excel(output_path, index=False)

print("处理完成，已保存为：", str(output_path))
