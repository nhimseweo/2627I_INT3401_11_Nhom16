# Q3 - digitclassifier_loss
from torch.nn.functional import mse_loss, cross_entropy


def digitclassifier_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred: a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    # y is one-hot (batch_size x 10); cross_entropy accepts class-probability targets.
    return cross_entropy(y_pred, y)