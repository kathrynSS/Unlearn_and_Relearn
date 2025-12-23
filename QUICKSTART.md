# AI知识点遗忘 - 快速开始指南

## ✅ 已完成的准备工作

### 1. 数据集（基于错误选项的方案）
```
data/data_ai/
├── forget/          # 78条遗忘数据（18个知识点）
├── retain/          # 564条保留数据
├── forget_perturbed/
└── retain_perturbed/
```

### 2. 核心文件
- ✅ `config/forget_ai.yaml` - Mistral-7B配置文件
- ✅ `data/replace_knowledge_points.json` - 知识点替换映射（用错误选项替换）
- ✅ `data/ai_knowledge_points.txt` - 18个遗忘知识点列表
- ✅ `teacher.py` - 已适配支持本地数据集
- ✅ `data_module.py` - 已适配自动加载本地数据

### 3. 模型配置
- **模型**: `mistralai/Mistral-7B-Instruct-v0.3`
- **方法**: Causal Intervention
- **N值**: 3（使用每道题的3个错误选项）

## 🚀 运行步骤

### 步骤1: 验证环境

```bash
cd /Users/songjiajia/Desktop/causal_unlearn-main

# 检查数据集
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/data_ai')
print('数据集splits:', list(ds.keys()))
print('forget数据量:', len(ds['forget']))
print('示例:', ds['forget'][0])
"
```

### 步骤2: 生成教师分布

```bash
# 使用Mistral-7B生成教师分布（N=3，使用错误选项）
python teacher.py --config-name=forget_ai.yaml

# 预期输出文件:
# results/mistralai/Mistral-7B-Instruct-v0.3/intervention/forget_3_false_true.pkl
```

**说明**：
- 这一步会对每个知识点的每道题，用其他3个错误选项生成替换版本
- 例如：正确答案是"C. Guido van Rossum"
  - 替换1: 用选项A的答案
  - 替换2: 用选项B的答案
  - 替换3: 用选项D的答案
- 模型会学习这3个替换版本的概率分布的平均值

**时间估计**：
- GPU: 1-2小时（取决于GPU性能）
- 进度: 78条数据 × 3个替换版本 = 234次前向传播

### 步骤3: 训练遗忘模型

```bash
# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python forget.py --config-name=forget_ai.yaml

# 多GPU训练（推荐）
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=18765 \
    forget.py --config-name=forget_ai.yaml
```

**训练参数**：
- 学习率: 1e-5
- Batch size: 8
- Epochs: 10
- 优化器: AdamW
- Weight decay: 0.01

**时间估计**：
- 单GPU: 3-4小时
- 双GPU: 2-3小时

### 步骤4: 查看结果

训练完成后，结果保存在：
```
results/mistralai/Mistral-7B-Instruct-v0.3/intervention/10_1e-5_forget/
├── checkpoint-{step}/
│   ├── eval_log.json                 # 保留集评估
│   ├── eval_log_forget.json          # 遗忘集评估
│   ├── eval_log_aggregated.json      # 汇总结果
│   └── aggregate_stat.csv            # 关键指标
└── pytorch_model.bin                  # 模型权重
```

## 📊 核心配置说明

### N=3 的含义

```yaml
teacher:
  N: 3  # 每道题用其他3个错误选项代替正确选项
```

**举例说明**：
```
原题:
Q: Who developed Python?
A: C. Guido van Rossum  ← 正确答案
其他选项: 
  A. Wick van Rossum
  B. Rasmus Lerdorf
  D. Niene Stom

生成3个替换版本:
版本1: Q: Who developed Python? → A. Wick van Rossum
版本2: Q: Who developed Python? → B. Rasmus Lerdorf
版本3: Q: Who developed Python? → D. Niene Stom

教师分布 = 这3个版本的平均概率分布
```

### 为什么是3而不是12？

- **3**: 理论值，每道题的其他选项数量
- **12**: 实际平均值，因为收集了多道题的错误选项

我们使用**N=3**因为：
1. 逻辑清晰：对应每道题的3个错误选项
2. 计算高效：前向传播次数更少
3. 已有代码支持：teacher.py会灵活使用实际可用的替换数

## ⚙️ 自定义配置

### 修改学习率
```bash
python forget.py --config-name=forget_ai.yaml lr=5e-6
```

### 修改batch size（显存不足时）
```bash
python forget.py --config-name=forget_ai.yaml \
    batch_size=4 \
    gradient_accumulation_steps=4
```

### 修改训练轮数
```bash
python forget.py --config-name=forget_ai.yaml num_epochs=5
```

### 使用不同的N值测试
```bash
# 只使用前3个替换
python teacher.py --config-name=forget_ai.yaml teacher.N=3

# 使用更多替换（如果可用）
python teacher.py --config-name=forget_ai.yaml teacher.N=5
```

## 🐛 常见问题

### Q1: 显存不足 (CUDA out of memory)
```bash
# 减小batch size，增加梯度累积
python forget.py --config-name=forget_ai.yaml \
    batch_size=2 \
    gradient_accumulation_steps=8
```

### Q2: teacher.py运行很慢
**原因**: 需要对每个样本生成N个替换版本并前向传播
**解决**: 正常现象，耐心等待或减小N值

### Q3: 数据集加载错误
```bash
# 检查数据集是否存在
ls -la data/data_ai/

# 重新生成数据集
python data_construct/5_create_dataset_with_options.py
```

### Q4: 模型下载失败
```bash
# 方法1: 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法2: 使用本地模型
# 修改 config/forget_ai.yaml:
#   model_path: "/path/to/local/mistral-7b"
#   from_hf_hub: false
```

## 📈 评估指标

训练完成后查看 `aggregate_stat.csv`：

| 指标 | 说明 | 期望值 |
|------|------|--------|
| forget_loss | 遗忘集损失 | ↑ 越高越好 |
| retain_accuracy | 保留集准确率 | ↑ 越高越好 |
| model_utility | 模型整体效用 | ↑ 越高越好 |
| forget_quality | 遗忘质量 | ↑ 越高越好 |

## 📝 18个遗忘的知识点

```
1. Functions               10. Bitwise
2. String                  11. Namespaces and Scope
3. Variable Names          12. output
4. Input & Output          13. Variables
5. Shallow copy vs Deep    14. Precedence and Associativity
6. Argument Parsing        15. Data Type
7. Function                16. Files
8. Lists                   17. Basic Operators
9. File read and write     18. Exception
```

## 🎯 下一步

1. **对比实验**: 尝试不同的forget_loss方法
   - intervention（当前）
   - grad_diff
   - npo

2. **参数调优**: 尝试不同的学习率和训练轮数

3. **扩展数据**: 添加更多知识点或问题

## 💡 提示

- 建议先在小batch上测试（2-3个样本）确保流程正确
- 可以使用 `teacher.verbose=true` 查看详细日志
- 训练过程中可以监控显存使用：`nvidia-smi -l 1`
- 建议保存中间checkpoint以防中断

---

如有问题，请查看详细文档：`README_AI_DATASET.md`



































