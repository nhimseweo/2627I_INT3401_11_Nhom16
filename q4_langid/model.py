# Q4 - LanguageIDModel
from torch import stack, tril
from torch.nn import Module
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros
from torch.nn.functional import relu, softmax
from torch import movedim


class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"
        # Initialize your model parameters here
        self.hidden_size = 192
        self.input_layer = Linear(self.num_chars, self.hidden_size)
        self.hidden_layer = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size * 2, len(self.languages))
        self.reverse_input = Linear(self.num_chars, self.hidden_size)
        self.reverse_hidden = Linear(self.hidden_size, self.hidden_size)


    def forward(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch_size = xs[0].shape[0]
        hidden = zeros((batch_size, self.hidden_size), dtype=xs[0].dtype, device=xs[0].device)
        reverse = zeros((batch_size, self.hidden_size), dtype=xs[0].dtype, device=xs[0].device)
        for x in xs:
            new = relu(self.input_layer(x) + self.hidden_layer(hidden))
            mask = x.sum(dim=1, keepdim=True)
            hidden = mask * new + (1 - mask) * hidden
        for i in range(len(xs)-1, -1, -1):
            x = xs[i]
            new = relu(self.reverse_input(x) + self.reverse_hidden(reverse))
            mask = x.sum(dim=1, keepdim=True)
            reverse = mask * new + (1 - mask) * reverse
        w = self.output_layer.weight
        return hidden @ w[:, :self.hidden_size].T + reverse @ w[:, self.hidden_size:].T + self.output_layer.bias
