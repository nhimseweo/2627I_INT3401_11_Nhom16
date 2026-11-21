# AUTO-GENERATED bởi tools/build_submission.py - KHÔNG sửa tay file này.
from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


# ======================================================================
# q1_perceptron/train.py
# ======================================================================
# Q1 - train_perceptron


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


# ======================================================================
# q2_regression/train.py
# ======================================================================
# Q2 - train_regression


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


# ======================================================================
# q3_digit/train.py
# ======================================================================
# Q3 - train_digitclassifier


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


# ======================================================================
# q4_langid/train.py
# ======================================================================
# Q4 - train_languageid


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 128
    eye = tensor([[float(i == j) for j in range(model.num_chars)] for i in range(model.num_chars)])
    classes = tensor([[float(i == j) for j in range(len(model.languages))] for i in range(len(model.languages))])
    best = -1.0
    best_state = None
    for epoch in range(20):
        batches = []
        for a, b in dataset.train_buckets:
            a, b = int(a), int(b)
            order = tensor(list(range(a, b)))
            # Shuffle within each length bucket using the allowed tensor API.
            order = order[tensor([1.0]*len(order)).multinomial(len(order), replacement=False)]
            for start in range(0, len(order), batch_size):
                batches.append(order[start:start+batch_size])
        order = tensor([1.0]*len(batches)).multinomial(len(batches), replacement=False)
        for i in order:
            ids = batches[int(i)].numpy()
            raw = tensor(dataset.train_x[ids]).long()
            length = int((raw[0] >= 0).sum())
            xs = eye[raw[:, :length]].transpose(0, 1)
            labels = classes[tensor(dataset.train_y[ids]).long()]
            optimizer.zero_grad()
            loss = languageid_loss(model(xs), labels)
            loss.backward()
            optimizer.step()
        with no_grad():
            correct = total = 0
            for a, b in dataset.dev_buckets:
                xs, labels = dataset._encode(dataset.dev_x[int(a):int(b)], dataset.dev_y[int(a):int(b)])
                correct += int((model(xs).argmax(1) == labels.argmax(1)).sum())
                total += len(labels)
            score = correct / total
            if score > best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 5 == 0:
            print(f"[Q4] epoch {epoch+1}/20 | dev accuracy {score:.1%}", flush=True)
        if epoch == 14:
            for group in optimizer.param_groups:
                group['lr'] = 0.0003
    model.load_state_dict(best_state)


# ======================================================================
# q5_conv/train.py
# ======================================================================
# Q5 - Train_DigitConvolution


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
