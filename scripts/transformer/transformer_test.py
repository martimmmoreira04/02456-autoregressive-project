import hydra
from hydra.core.hydra_config import HydraConfig

from protgen.models.transformer_freeze import load_frozen_transformer
import torch
from torch import nn
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
import pandas as pd


@hydra.main(
    version_base=None,
    config_path="../../config",
)

def main(cfg):
    # LOAD
    test_data = ProteinDataset(path_to_data=cfg.test_data)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    model = load_frozen_transformer(cfg.weights, device=cfg.hardware)  

    total_test_loss = 0

    with torch.no_grad():
        for x, x_shifted, y, length in test_loader:
            x = x.to(cfg.hardware)
            x_shifted = x_shifted.to(cfg.hardware)

            out = model(x)
            loss = loss_fn(out.transpose(1, 2), x_shifted)
            total_test_loss = total_test_loss + loss.item()

    test_loss = total_test_loss / len(test_loader)

    output_dir = HydraConfig.get().runtime.output_dir
    pd.DataFrame({"test_loss": [test_loss]}).to_csv(f"{output_dir}/test_loss.csv", index=False)


if __name__ == "__main__":
    main()