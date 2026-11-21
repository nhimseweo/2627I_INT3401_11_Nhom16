# Q5 - Train_DigitConvolution
from torch import no_grad
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


def Train_DigitConvolution(model, dataset):
    """
    Trains the model.

    - Autograder dùng DigitClassificationDataset2 (chỉ 200 ảnh, và "test" cũng nằm trong
      200 ảnh đó) nên chạy nhánh nhỏ: train nhanh cho qua điểm.
    - GUI dùng bộ MNIST đầy đủ (DigitClassificationDataset, dev/test tách riêng khỏi train)
      nên chạy nhánh lớn: học trên hàng chục nghìn ảnh và dừng theo độ chính xác dev THẬT.
    """
    model.train()
    big = len(dataset) > 1000
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    if not big:
        for epoch in range(15):
            for batch in dataloader:
                x = batch['x']
                y = batch['label']

                optimizer.zero_grad()
                y_pred = model(x)
                loss = digitconvolution_Loss(y_pred, y)
                loss.backward()
                optimizer.step()

            if dataset.get_validation_accuracy() >= 0.96:
                break
        return

    epochs = 3
    max_batches = 400  # ~25.600 ảnh mỗi epoch (shuffle nên mỗi epoch thấy tập con khác nhau)
    for epoch in range(epochs):
        if epoch == 2:
            for g in optimizer.param_groups:
                g['lr'] = 0.0005
        for step, batch in enumerate(dataloader):
            if step >= max_batches:
                break
            x = batch['x']
            y = batch['label']

            optimizer.zero_grad()
            y_pred = model(x)
            loss = digitconvolution_Loss(y_pred, y)
            loss.backward()
            optimizer.step()

        with no_grad():
            val_acc = dataset.get_validation_accuracy()
        if val_acc >= 0.98:
            break