import torch
from protgen.models.transformer_rocha import Transformer

def load_frozen_transformer(path, device="cpu"):
# Rebuild the transformer model with the best architecture according to hydra's validation
# loss 

    checkpoint = torch.load(path, map_location=device)
    model = Transformer(vocabulary_size=23, embedding_size=checkpoint["embedding_size"], num_heads=checkpoint["num_heads"], num_layers=checkpoint["num_layers"], dropout=checkpoint["dropout"])
    model.load_state_dict(checkpoint["weights"])
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

