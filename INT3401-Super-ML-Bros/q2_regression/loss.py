# Q2 - regression_loss
from torch.nn.functional import mse_loss, cross_entropy


def regression_loss(y_pred, y):
    return mse_loss(y_pred, y)
