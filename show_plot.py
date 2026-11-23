import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("reward_log.jsonl", lines=True)
plt.plot(df["step"], df["reward"], marker="o", markersize=3)
plt.xlabel("Optimizer step")
plt.ylabel("Mean reward")
plt.title("GRPO reward over training")
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150)
plt.show()
