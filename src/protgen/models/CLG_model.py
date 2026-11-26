
# copy of 'transformer_rocha.py' for classifier free model (CFG)

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
        embedding_size: int,
        padding_idx: int = 0,
        n_labels: int = 3): #ADDED – so it can process the three labels of the data. 

        super().__init__()
        self.padding_idx = padding_idx
        self.vocabulary_size = vocabulary_size
        self.sinusoid_positional_encoding = SinusoidPositionalEncoding(token_dim=embedding_size)    
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_idx)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=embedding_size * 4, batch_first=True,
            norm_first = True, #before LN - but check this ? -----check wat? 
            activation = "relu", 
            )   
        self.encoder = nn.TransformerEncoder(encoder_layers, norm=nn.LayerNorm(embedding_size), num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_size, vocabulary_size)
        self.label_embedding = nn.Embedding(n_labels + 1,embedding_size) # ADDED! – so we reference to label embedding


    def forward(self, x, label, return_hidden=False): # ADDED variable for recieving labels 
        T = x.size(1)
        padding_mask = (x == self.padding_idx)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device).isinf()

        x = self.embedding(x)   
        label_forward_embed = self.label_embedding(label) # ADDED - converts labels into embeddings. 
        x = x + label_forward_embed.unsqueeze(1) # ADDED – adds family embedding to all amino acids positions
        x = self.sinusoid_positional_encoding(x)
        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)

        logits = self.output_layer(h)
        if return_hidden:
            return logits, h
        return logits 
