# Q6 - Attention

from torch import stack, tril
from torch.nn import Module
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros
from torch.nn.functional import relu, softmax
from torch import movedim


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()

        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        B, T, C = input.size()

        q = self.q_layer(input)
        k = self.k_layer(input)
        v = self.v_layer(input)

        scores = matmul(q, k.transpose(1, 2)) / (self.layer_size ** 0.5)

        scores = scores.masked_fill(
            self.mask[:, :, :T, :T][0] == 0,
            float('-inf')
        )

        weights = softmax(scores, dim=-1)

        out = matmul(weights, v)

        return out

