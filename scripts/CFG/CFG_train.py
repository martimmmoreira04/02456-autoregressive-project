from protgen.models.CLG_model import Transformer

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = Path("results/transformer")
out_dir.mkdir(parents=True, exist_ok=True)

# data 

train_data = ProteinDataset(path_to_data="data/train.csv")
val_data = ProteinDataset(path_to_data="data/val.csv")
#test_data = ProteinDataset(path_to_data="../data/test.csv")

print(f"Data class: {train_data}")

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

# model 
transformer = Transformer(num_heads=4, num_layers=2, vocabulary_size=23, embedding_size=32, n_labels=3)
transformer = transformer.to(device) 

loss_fn = nn.CrossEntropyLoss(ignore_index = -1)
optimizer = optim.AdamW(transformer.parameters(), lr= 0.001)

print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")

# training 
num_epochs = 30
val_losses = []
train_losses = []
label_dropped = 3 # defines how many labels there is 
LABEL_DROPOUT = 0.1 # drops 10% of labels during training 

for epoch in range(num_epochs):
    
    ### Training
    transformer.train()
    total = 0
    for x, x_shifted, y, length in train_loader:
        x = x.to(device)
        x_shifted = x_shifted.to(device)
        y = y.to(device)
        label = y.clone()
        label_to_drop_mask = torch.rand(y.shape, device=device) < LABEL_DROPOUT # choose at random which observation to drop its label
        label[label_to_drop_mask] = label_dropped 

        optimizer.zero_grad()
        out = transformer(x, label) # added label now we have added it in model definition
        loss = loss_fn(out.transpose(1, 2), x_shifted)
        loss.backward()
        optimizer.step()
        total += loss.item()
    train_loss = total / len(train_loader)
    train_losses.append(train_loss)

    # validation
    transformer.eval()
    total = 0
    with torch.no_grad():
        for x, x_shifted, y, length in val_loader:
            x, x_shifted = x.to(device), x_shifted.to(device)
            out = transformer(x)
            loss = loss_fn(out.transpose(1, 2), x_shifted)
            total += loss.item()
    val_loss = total / len(val_loader)
    val_losses.append(val_loss)
    saved = ""
    if val_loss <= min(val_losses):
        torch.save(transformer.state_dict(), out_dir / "transformer_beautiful_model.pt")
        saved = " model with the lawest validation loss"

    print(f"Epoch {epoch+1}/{num_epochs} | train {train_loss:.4f} | val {val_loss:.4f} {saved}")


# resuls 

print(f"Lowest validation loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")

epochs = range(1, num_epochs + 1)
losses = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}, index=epochs)
losses.to_csv(out_dir / "losses.csv", index_label="epoch")

plt.figure()
plt.plot(epochs, train_losses, 'r', label='Train loss')
plt.plot(epochs, val_losses, 'b', label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(out_dir / "loss.png")
print(f"transformer_beautiful_model.pt, losses.csv and loss.png in {out_dir}/")
