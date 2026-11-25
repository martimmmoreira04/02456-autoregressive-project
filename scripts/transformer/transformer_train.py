from networkx import config

import hydra
from hydra.core.hydra_config import HydraConfig

from protgen.models.transformer_rocha import Transformer
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from protgen.data import ProteinDataset
import pandas as pd
from pathlib import Path

@hydra.main(
    version_base=None, 
    config_path="../../config",
)

def main(config):
    # LOAD DATA
    train_data = ProteinDataset(path_to_data="data/train.csv")
    val_data = ProteinDataset(path_to_data="data/val.csv")

    #load train and validation 
    train_loader=DataLoader(train_data, batch_size=config.training_parameters.batch_size, shuffle=True)
    val_loader=DataLoader(val_data, batch_size=config.training_parameters.batch_size, shuffle=False)  


    # SETUP
    torch.manual_seed(config.training_parameters.seed)
    model = Transformer(vocabulary_size=23, embedding_size=config.model_layout.embedding_size,num_heads=config.model_layout.num_heads,num_layers=config.model_layout.num_layers)
    model.to(config.hardware)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # choose optimizer
    if config.training_parameters.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.training_parameters.lr, momentum=0.9) #momentum=0.9 to make the optimization faster and more stable - ⚠️ check for the MLP
    elif config.training_parameters.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.training_parameters.lr)
    else:
        optimizer=optim.AdamW(model.parameters(), lr=config.training_parameters.lr)


    # TRAIN + VAL (prep for the loop)
    num_epochs=config.training_parameters.epochs
    val_losses=[]
    train_losses = [] 
    output_dir=HydraConfig.get().runtime.output_dir # get output dir from hydra for the run
    Path(output_dir, "weights").mkdir()


    # check if summaries are initialized, init if not
    epoch_index = range(1, num_epochs + 1) # index for creating init file-.
    train_summ_path=Path(f"{output_dir}/../../train_summary.csv")
    if not train_summ_path.exists():
        pd.DataFrame(index=epoch_index).to_csv(train_summ_path, index_label="epoch")

    val_summ_path = Path(f"{output_dir}/../../val_summary.csv")
    if not val_summ_path.exists():
        pd.DataFrame(index=epoch_index).to_csv(val_summ_path, index_label="epoch")



    for epoch in range(num_epochs):
        
        ### Training
        model.train()
        total = 0
        for x, x_shifted, y, length in train_loader:
            optimizer.zero_grad()
            x = x.to(config.hardware)
            x_shifted=x_shifted.to(config.hardware)
            out = model(x)
            loss = loss_fn(out.transpose(1, 2), x_shifted)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss=total / len(train_loader)
        train_losses.append(train_loss)

        # save weights for each epoch 
        checkpoint = {"weights": model.state_dict(), "embedding_size": config.model_layout.embedding_size, "num_heads": config.model_layout.num_heads, "num_layers": config.model_layout.num_layers}
        torch.save(checkpoint, f"{output_dir}/weights/epoch_{epoch+1}.pt")

        # validation
        model.eval()
        total = 0
        with torch.no_grad():
            for x, x_shifted, y, length in val_loader:
                x = x.to(config.hardware)
                x_shifted = x_shifted.to(config.hardware)
                out=model(x)
                loss = loss_fn(out.transpose(1, 2), x_shifted)
                total += loss.item()
        val_loss = total / len(val_loader)
        val_losses.append(val_loss)

    # SAVE RESULTS
    # define name and path
    col_name = (f"emb{config.model_layout.embedding_size}" f"_h{config.model_layout.num_heads}" f"_L{config.model_layout.num_layers}" f"_{config.training_parameters.hyper_run}")
    train_losses_path = f"{output_dir}/train_loss.csv"
    val_losses_path = f"{output_dir}/val_loss.csv"
    # rename col 
    df_train_losses = pd.DataFrame(train_losses, index=epoch_index).rename_axis("epoch")
    df_train_losses.index.name="epoch"
    df_train_losses.columns=[col_name]
    df_val_losses=pd.DataFrame(val_losses, index=epoch_index).rename_axis("epoch")
    df_val_losses.index.name= "epoch"
    df_val_losses.columns=[col_name]
    #save individual loss files 
    df_train_losses.to_csv(train_losses_path)
    df_val_losses.to_csv(val_losses_path)

    # ADD TO SUMMARY 
    # load summary 
    train_summary = pd.read_csv(train_summ_path, index_col="epoch")
    val_summary = pd.read_csv(val_summ_path, index_col="epoch")
    # add col   
    train_summary[col_name]=df_train_losses[col_name]
    val_summary[col_name] = df_val_losses[col_name]
    # save with added col
    train_summary.to_csv(train_summ_path)
    val_summary.to_csv(val_summ_path)


if __name__ == "__main__":
    main() 