# Q1 - train_perceptron
from torch import no_grad
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        converged = False
        while not converged:
            converged = True
            for batch in dataloader:
                x = batch['x'][0]
                label = batch['label'][0]
                prediction = model.get_prediction(x)
                if prediction != label:
                    model.get_weights().data += (x * label).unsqueeze(0)
                    converged = False