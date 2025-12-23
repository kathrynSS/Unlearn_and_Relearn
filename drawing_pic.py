import json
import matplotlib.pyplot as plt
import os

# 日志文件路径（按你当前目录调整）
log_path = os.path.join(
    "/Users/songjiajia/Desktop/causal_unlearn-main",
    "log_results/1/results/mistralai/Mistral-7B-Instruct-v0.3/intervention/100_0.0003_forget/training_log.txt"
)

steps, epochs, losses = [], [], []

with open(log_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        # 训练过程中的记录都有 loss 字段
        if "loss" in rec:
            steps.append(rec["global_step"])
            epochs.append(rec["epoch"])
            losses.append(rec["loss"])

# 画 step-loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(steps, losses, marker="o", linewidth=1)
plt.xlabel("global_step")
plt.ylabel("loss")
plt.title("Training Loss vs Global Step")
plt.grid(True)
plt.tight_layout()
plt.savefig("./evolution/loss_curve_step.png", dpi=200)

# 如果想按 epoch 画，可以另外再画一张
plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker="o", linewidth=1)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("./evolution/loss_curve_epoch.png", dpi=200)

print("Saved to loss_curve_step.png and loss_curve_epoch.png")