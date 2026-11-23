
# uv run python scripts/MLP/MLP_summary_plot.py

import pandas as pd
import matplotlib.pyplot as plt

folder = "hydra_results/training/MLP_train_sweep_11-23_12-26-09"

train_summary = pd.read_csv(f"{folder}/train_summary.csv")
train_summary.plot(x="epoch")
train_summary.plot(x="epoch", figsize=(12, 6))  # size of the whole image
plt.subplots_adjust(right=0.70) 
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.savefig(f"{folder}/train_summary.png", dpi=150)
plt.close()

val_summary = pd.read_csv(f"{folder}/val_summary.csv")
val_summary.plot(x = "epoch")
val_summary.plot(x="epoch", figsize=(12, 6))  # size of the whole image
plt.subplots_adjust(right=0.70) 
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.savefig(f"{folder}/val_summary.png", dpi=150) # dpi is for resolution
plt.close()
