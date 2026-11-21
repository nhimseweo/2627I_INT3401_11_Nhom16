# Q3 - train_digitclassifier
from torch import no_grad
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    batch_size = 100  # 60000 is evenly divisible by 100
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 30
    target_val_acc = 0.98

    for epoch in range(max_epochs):
        for batch in dataloader:
            x, y = batch['x'], batch['label']
            optimizer.zero_grad()
            loss = digitclassifier_loss(model(x), y)
            loss.backward()
            optimizer.step()

        with no_grad():
            val_acc = dataset.get_validation_accuracy()
        if val_acc >= target_val_acc:
            break