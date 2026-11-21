import torch
from torch import nn
from models import Attention


class Block(nn.Module):
    def __init__(self, width, context):
        super().__init__()
        self.attention = Attention(width, context)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                nn.Linear(4 * width, width), nn.Dropout(0.1))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, context=64, width=128, layers=4):
        super().__init__()
        self.block_size = context
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(width, context) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        return self.head(self.norm(self.blocks(self.tokens(tokens) + self.positions(positions))))
