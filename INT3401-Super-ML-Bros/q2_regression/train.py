# Q2 - train_regression
from torch import no_grad
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


def train_regression(model, dataset):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_all = tensor(dataset.x, dtype=tensor(0.0).dtype)
    y_all = tensor(dataset.y, dtype=tensor(0.0).dtype)

    for epoch in range(3000):
        for batch in dataloader:
            x, y = batch['x'], batch['label']

            optimizer.zero_grad()

            y_pred = model(x)
            loss = regression_loss(y_pred, y)

            loss.backward()
            optimizer.step()

        with no_grad():
            full_loss = regression_loss(model(x_all), y_all)

        if full_loss.item() < 0.0001:
            return
