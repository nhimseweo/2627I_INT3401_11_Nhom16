# Q2 - RegressionModel
from torch import stack, tril
from torch.nn import Module
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros
from torch.nn.functional import relu, softmax
from torch import movedim


class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        super().__init__()

        self.layer1 = Linear(1, 64)
        self.layer2 = Linear(64, 64)
        self.layer3 = Linear(64, 1)

    def forward(self, x):
        h = relu(self.layer1(x))
        h = relu(self.layer2(h))
        out = self.layer3(h)
        return out
