
import hydra 
from hydra.core.hydra_config import HydraConfig

from protgen.models.mlp import MLP
import torch
from torch import nn
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset, total_tokens
import pandas as pd 
import os
from pathlib import Path

@hydra.main(
    version_base=None, 
    config_path="../../config",
)

def main(cfg):

    checkpoint = torch.load(cfg.weights, map_location="cpu", weights_only=True)
    num_hidden = checkpoint["num_hidden"]
    weights = checkpoint["weights"]

    mlp = MLP(num_hidden=num_hidden)
    mlp.load_state_dict(weights)
    mlp.to(cfg.hardware)

    mlp.eval()
    torch.manual_seed(cfg.seed)
    sequences = []

    with torch.no_grad():
        for _ in range(cfg.num_samples):
            tokens = [1]  # BOS start

            for _ in range(cfg.max_length):
                x = torch.tensor([tokens],device=cfg.hardware)
                out = mlp(x)
                last_logits = out[0, -1]
                logits = last_logits / cfg.temperature
                logits[0] = -float("inf")  # no sampling of PAD
                logits[1] = -float("inf")  # no sampling of BOS

                probabilities = torch.softmax(logits, dim=0)
                next_token = torch.multinomial(probabilities, 1).item()

                if next_token == 2:  # stop if EOS
                    break

                tokens.append(next_token)

            sequence = ""
            for token in tokens[1:]:
                sequence += total_tokens[token]
            sequences.append(sequence)

    output_dir = HydraConfig.get().runtime.output_dir # get output dir from hydra for the run
    generated_results = pd.DataFrame({"sequence": sequences})
    generated_results.to_csv(f"{output_dir}/sequence.csv", index=False)

if __name__ == "__main__":
    main() 