# AUTO-GENERATED bởi tools/build_submission.py - KHÔNG sửa tay file này.
from torch.nn.functional import mse_loss, cross_entropy


# ======================================================================
# q2_regression/loss.py
# ======================================================================
# Q2 - regression_loss


def regression_loss(y_pred, y):
    return mse_loss(y_pred, y)


# ======================================================================
# q3_digit/loss.py
# ======================================================================
# Q3 - digitclassifier_loss


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


# ======================================================================
# q4_langid/loss.py
# ======================================================================
# Q4 - languageid_loss


def languageid_loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a node with shape
    (batch_size x 5). Each row is a one-hot vector encoding the correct
    language.

    Inputs:
        model: Pytorch model to use
        y_pred: a node with shape (batch_size x 5)
        y: a node with shape (batch_size x 5)
    Returns: a loss node
    """
    "*** YOUR CODE HERE ***"
    return cross_entropy(y_pred, y)


# ======================================================================
# q5_conv/loss.py
# ======================================================================
# Q5 - digitconvolution_Loss


def digitconvolution_Loss(y_pred, y):
    """
    Computes the loss for a batch of examples.

    The correct labels `y` are represented as a tensor with shape
    (batch_size x 10). Each row is a one-hot vector encoding the correct
    digit class (0-9).

    Inputs:
        y_pred : a node with shape (batch_size x 10)
        y: a node with shape (batch_size x 10)
    Returns: a loss tensor
    """
    """ YOUR CODE HERE """
    return cross_entropy(y_pred, y)
