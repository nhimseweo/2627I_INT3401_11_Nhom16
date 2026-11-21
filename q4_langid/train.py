# Q4 - train_languageid
from torch import no_grad
from torch.utils.data import DataLoader
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


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
