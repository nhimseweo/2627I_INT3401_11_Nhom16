# Q5 - Convolve + DigitConvolutionalModel
from torch import stack, tril
from torch.nn import Module
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros
from torch.nn.functional import relu, softmax
from torch import movedim


def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    in_h, in_w = input_tensor_dimensions
    w_h, w_w = weight_dimensions
    out_h = in_h - w_h + 1
    out_w = in_w - w_w + 1
    Output_Tensor = zeros((out_h, out_w))

    # Output[y, x] = sum_{i,j} weight[i, j] * input[y + i, x + j]
    # Duyệt qua các phần tử của kernel (3x3 = 9 lần) thay vì 26x26 = 676 lần:
    # kết quả giống hệt nhưng nhanh hơn rất nhiều, gradient vẫn chảy về `weight`.
    for i in range(w_h):
        for j in range(w_w):
            Output_Tensor = Output_Tensor + weight[i, j] * input[i:i + out_h, j:j + out_w]
    "*** End Code ***"
    return Output_Tensor


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.

    Note that this class looks different from a standard pytorch model since we don't need to train it
    as it will be run on preset weights.
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        self.fc1 = Linear(26 * 26, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)

    def forward(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimensional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)