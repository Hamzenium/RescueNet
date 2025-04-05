import json
import matplotlib.pyplot as plt

trainer_state_path = "./results/checkpoint-250/trainer_state.json"

with open(trainer_state_path, "r") as f:
    state = json.load(f)

train_steps = []
train_loss = []
eval_steps = []
eval_loss = []

for log in state["log_history"]:
    if "loss" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
    elif "eval_loss" in log:
        eval_steps.append(log["step"])
        eval_loss.append(log["eval_loss"])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Eval Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
