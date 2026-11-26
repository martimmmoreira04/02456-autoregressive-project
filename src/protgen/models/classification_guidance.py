import torch
from torch import nn
import torch.nn.functional as F
import math

from protgen.models.transformer_rocha import SinusoidalPositionalEncoding

class FamilyClass(nn.Module):
    def __init__(
        self,
        num_heads,
        num_layers,
        vocabulary_size,
        embedding_size,
        num_families = 3,
        padding_idx= 0):

        super().__init__()
        self.padding_idx = padding_idx
        self.vocabulary_size = vocabulary_size
        self.sinusoid_positional_encoding = SinusoidalPositionalEncoding(token_dim=embedding_size)
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_idx)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=embedding_size * 4, batch_first=True,
            norm_first = True, #before LN - but check this 
            activation = "relu", #we need to test if this is the best activation function for our use case
            ) 
        self.encoder = nn.TransformerEncoder(encoder_layers, norm=nn.LayerNorm(embedding_size), num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_size, num_families)


    def forward(self, x): # x: (B, T) token ids
        T = x.size(1)
        padding_mask = (x == self.padding_idx)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device).isinf()

        x = self.embedding(x)   #pe[:T]
        x = self.sinusoid_positional_encoding(x)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)

        x = self.output_layer(x)
        return x 

if __name__ == "__main__":
    clf = FamilyClass(num_heads=4, num_layers=2, vocabulary_size=23, embedding_size=64)
    out = clf(torch.randint(1, 23, (8, 129)))
    print(out.shape)
    print(out.softmax(-1)[0, 0])