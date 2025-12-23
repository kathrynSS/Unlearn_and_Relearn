# AI知识点遗忘数据集 - 使用说明

## 📋 概述

本文档说明如何使用 `data/data_ai` 数据集进行AI知识点的机器遗忘训练。数据集已经构造为类似TOFU的格式，可以直接使用原有的 `teacher.py` 和 `forget.py` 进行训练。

## 📁 数据集结构

### 生成的文件

```
data/
├── data_ai/                          # 主数据集目录
│   ├── forget/                       # 遗忘集 (78条)
│   ├── retain/                       # 保留集 (564条)
│   ├── forget_perturbed/             # 遗忘集评估版本 (78条)
│   └── retain_perturbed/             # 保留集评估版本 (171条)
├── replace_knowledge_points.json     # 知识点替换映射 (18个知识点 × 20个替换)
└── ai_knowledge_points.txt           # 遗忘的知识点列表 (18个)
```

### 数据集详情

- **forget**: 包含18个要遗忘的知识点的问题（每个知识点5题）
- **retain**: 从训练集中选择的其他知识点（不在遗忘列表中）
- **forget_perturbed**: 用于评估遗忘效果，包含扰动答案
- **retain_perturbed**: 用于评估模型在其他知识上的保留能力

### 遗忘的18个知识点

```
Functions, String, Variable Names, Input & Output, 
Shallow copy vs Deep copy, Argument Parsing, Function, Lists,
File read and write, Bitwise, Namespaces and Scope, output,
Variables, Precedence and Associativity, Data Type, Files,
Basic Operators, Exception
```

## 🚀 快速开始

### 步骤 1: 验证数据集

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/data_ai')
print('可用splits:', list(ds.keys()))
print('forget数据量:', len(ds['forget']))
print('示例:', ds['forget'][0])
"
```

### 步骤 2: 生成教师分布（Causal Intervention方法）

```bash
# 使用默认配置（N=20, 即每个知识点使用20个替换）
python teacher.py --config-name=forget_ai.yaml

# 或者自定义N值
python teacher.py --config-name=forget_ai.yaml teacher.N=10
```

**参数说明:**
- `teacher.N`: 每个知识点生成的替换数量（默认20）
- `teacher.change_name_back`: 是否在答案中将替换知识点改回原知识点（默认true）
- `teacher.verbose`: 是否打印详细信息（默认false）

**预期输出:**
- 生成文件: `results/locuslab/tofu_ft_llama2-7b/intervention/forget_20_false_true.pkl`
- 这个文件包含了每个知识点的教师概率分布

### 步骤 3: 训练遗忘模型

```bash
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python forget.py --config-name=forget_ai.yaml

# 多GPU训练（推荐）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=18765 \
    forget.py --config-name=forget_ai.yaml
```

**训练参数（可在命令行覆盖）:**
```bash
python forget.py --config-name=forget_ai.yaml \
    lr=1e-5 \
    batch_size=8 \
    num_epochs=10 \
    teacher.N=20
```

### 步骤 4: 评估模型

训练完成后会自动进行评估，结果保存在：
```
results/locuslab/tofu_ft_llama2-7b/intervention/10_1e-5_forget/
└── checkpoint-{step}/
    ├── eval_log.json                 # 保留集评估
    ├── eval_log_forget.json          # 遗忘集评估
    ├── eval_log_aggregated.json      # 汇总结果
    └── aggregate_stat.csv            # 统计指标
```

## ⚙️ 配置文件说明

`config/forget_ai.yaml` 中的关键配置：

```yaml
# 模型配置
model_family: llama2-7b
model_path: "locuslab/tofu_ft_llama2-7b"  # 可替换为你的模型

# 数据配置
split: forget                  # 使用forget split
data_path: data/data_ai        # 数据集路径
input_type: question           # 使用QA格式

# 训练配置
lr: 1e-5
batch_size: 8
gradient_accumulation_steps: 2
num_epochs: 10
forget_loss: intervention      # 使用intervention方法
retain_strength: 0.0           # intervention方法不需要retain数据

# 教师配置
teacher:
  N: 20                        # 每个知识点使用20个替换
  change_name_back: true       # 将替换的知识点改回原知识点
  counter_fact_prompt: false   # 不使用counterfactual prompt
```

## 🔧 核心修改说明

为了支持AI知识点数据集，我们对原代码做了以下修改：

### 1. 数据加载（`data_module.py`）

添加了 `load_dataset_auto()` 函数，自动识别本地或远程数据集：

```python
def load_dataset_auto(data_path, split=None):
    if os.path.exists(data_path):
        # 本地数据集
        dataset_dict = datasets.load_from_disk(data_path)
        return dataset_dict[split] if split else dataset_dict
    else:
        # 远程数据集（HuggingFace）
        return datasets.load_dataset(data_path, split)["train"]
```

### 2. 知识点识别（`teacher.py` 和 `data_module.py`）

添加了对 `data_ai` 数据集的支持：

```python
if 'data_ai' in cfg.data_path:
    # AI知识点遗忘
    replace_name_file = 'data/replace_knowledge_points.json'
    # 使用title字段作为知识点
    question_to_title = {item['question']: item['title'] for item in dataset}
    titles = {item['title'] for item in dataset}
```

### 3. 数据集格式

数据集包含以下字段：
- `question`: 问题文本
- `answer`: 答案文本（格式："选项字母. 答案内容"）
- `title`: 知识点名称（用于识别和替换）

## 📊 实验流程

```
原始数据 (all_questions.xlsx)
    ↓
划分数据集 (2.py)
    ├── questions_train.xlsx  (训练集)
    ├── questions_forget.xlsx (遗忘集)
    └── questions_test.xlsx   (测试集)
    ↓
构造TOFU格式 (4_create_hf_dataset.py)
    ├── data/data_ai/
    ├── replace_knowledge_points.json
    └── ai_knowledge_points.txt
    ↓
生成教师分布 (teacher.py)
    └── results/.../forget_20_false_true.pkl
    ↓
训练遗忘模型 (forget.py)
    └── results/.../checkpoint-{step}/
    ↓
评估效果
    ├── 遗忘质量：模型是否忘记了目标知识点
    └── 模型效用：模型在其他知识上的表现
```

## 🐛 常见问题

### Q1: 报错 "KeyError: 'train'"
**原因**: 本地数据集不需要访问'train'键
**解决**: 已通过 `load_dataset_auto()` 函数自动处理

### Q2: 找不到知识点替换映射
**原因**: `replace_knowledge_points.json` 不存在
**解决**: 运行 `python data_construct/4_create_hf_dataset.py` 重新生成

### Q3: 训练时显存不足
**解决方案**:
```yaml
# 在配置文件中减小batch_size
batch_size: 4
gradient_accumulation_steps: 4  # 增加梯度累积以保持有效batch_size
```

### Q4: 教师分布文件已存在
**原因**: 避免重复计算，teacher.py会跳过已存在的文件
**解决**: 
```bash
# 方法1: 删除旧文件
rm results/locuslab/tofu_ft_llama2-7b/intervention/forget_*.pkl

# 方法2: 修改配置使用不同的N值
python teacher.py --config-name=forget_ai.yaml teacher.N=15
```

## 📈 评估指标

模型评估会计算以下指标：

1. **遗忘质量 (Forget Quality)**
   - 模型在遗忘集上的困惑度↑ = 遗忘效果好
   - 生成答案与正确答案的差异↑ = 遗忘效果好

2. **模型效用 (Model Utility)**
   - 模型在保留集上的准确率↑ = 保留能力强
   - 模型在保留集上的困惑度↓ = 保留能力强

## 🔍 下一步

1. **调整超参数**: 尝试不同的学习率、epoch数等
2. **对比实验**: 与其他遗忘方法（grad_diff, npo）对比
3. **扩展数据集**: 添加更多知识点或问题

## 📚 参考资料

- 原始TOFU论文: [TOFU: A Task of Fictitious Unlearning](https://arxiv.org/abs/2401.06121)
- Causal Intervention方法: 见项目README

## 💡 提示

- 第一次运行teacher.py会比较慢（需要计算所有替换版本的概率分布）
- 建议使用多GPU训练以加快速度
- 可以先用小的N值（如N=5）快速测试流程
- 训练前建议先在小batch上验证数据加载正常

---

如有问题，请检查：
1. 数据集是否正确生成（`data/data_ai` 目录存在且包含4个split）
2. 知识点文件是否存在（`data/ai_knowledge_points.txt`）
3. 替换映射是否正确（`data/replace_knowledge_points.json`）

