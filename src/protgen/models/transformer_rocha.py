import torch
from torch import nn
import torch.nn.functional as F
import math


class SinusoidPositionalEncoding(nn.Module): # https://medium.com/@uzbrainai/transformers-and-sequence-order-sinusoidal-vs-learned-positional-embeddings-b3802df5ce24
    def __init__(
              self, 
              token_dim, 
              max_len=129):
        
        super().__init__()
        pe = torch.zeros(max_len, token_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, token_dim, 2).float() * -(math.log(10000.0) / token_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    
class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        vocabulary_size: int,
        embedding_dim: int,
        padding_idx: int = 0):

        super().__init__()
        self.padding_idx = padding_idx
        self.vocabulary_size = vocabulary_size
        self.sinusoid_positional_encoding = SinusoidPositionalEncoding(token_dim=embedding_dim)    
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=padding_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim * 4, batch_first=True,
            norm_first = True, #before LN - but check this 
            activation = "relu", #we need to test if this is the best activation function for our use case
            )   
        self.encoder = nn.TransformerEncoder(encoder_layer, norm=nn.LayerNorm(embedding_dim), num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocabulary_size)


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
    m = Transformer(num_heads=4, num_layers=2, vocabulary_size=23, embedding_dim=24)
    x = torch.randint(1, 23, (4, 129))
    x[:, 100:] = 0
    out = m(x)
    print(out.shape)