import hydra
from hydra.core.hydra_config import HydraConfig

from protgen.models.transformer_freeze import load_frozen_transformer
from protgen.data import total_tokens
import torch
import pandas as pd

@hydra.main(
    version_base=None, 
    config_path="../../config",
)

def sample(config):
    model = load_frozen_transformer(config.weights, device=config.hardware)
    torch.manual_seed(config.seed)
    sequences = []

# w EOS stopping + PAD/BOS masking (added) because the code from 
# where was adapted from,had a limited size. 
    with torch.no_grad():
        for t in range(config.num_samples):
            tokens = [1] #BOS 

            for t in range(config.max_length):
                tensor=torch.tensor([tokens], device=config.hardware)
                output=model(tensor)
                logits=output[0,-1]/config.temperature
                logits[0]=-float("inf") #PAD - from special_tokens = ["<PAD>", "<BOS>", "<EOS>"] as  0, 1, 2 in make_data.py
                logits[1]=-float("inf") #BOS - from special_tokens = ["<PAD>", "<BOS>", "<EOS>"] as  0, 1, 2 in make_data.py
                next_token = torch.distributions.Categorical(logits=logits).sample().item()

                if next_token==2: #stop if it is the EOS
                    break

                tokens.append(next_token)

            sequence= ""
            for token in tokens[1:]:
                sequence = sequence + total_tokens[token]
            sequences.append(sequence)

    output_dir = HydraConfig.get().runtime.output_dir
    pd.DataFrame({"sequence": sequences}).to_csv(f"{output_dir}/sequence.csv", index=False)

if __name__ == "__main__":
    sample()