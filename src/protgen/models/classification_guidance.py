#https://www.youtube.com/watch?v=PTzHj5UmP9Y Inspiration for family FamilyHead architecture

import torch
from torch import nn


class FamilyHead(nn.Module):
    def __init__(self, frozen_transformer, embedding_size, hidden_units, num_families=3):
        super().__init__()

        # gotta take the optimal transformer we found
        frozen_transformer.eval()
        for param in frozen_transformer.parameters():
            param.requires_grad = False

        self.frozen_transformer = frozen_transformer
        self.head = nn.Sequential(
                nn.Linear(embedding_size, hidden_units),
                nn.GELU(),
                nn.Linear(hidden_units, num_families),
            )

    def forward(self, x):
        self.frozen_transformer.eval()

        with torch.no_grad():
            outputs = self.frozen_transformer(x, return_hidden=True)

        h = outputs[1]                   # Gives me the hidden states without retweaking

        family_class = self.head(h)     #save scores for 3
        return family_class