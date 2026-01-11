import torch

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(X)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")
    avg_loss = running_loss / size
    accuracy = correct / size
    return avg_loss, accuracy

def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item() * X.size(0)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy

def validate(model, dataloader, loss_fn, device):
    val_loss = 0
    val_correct = 0
    size = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            val_loss += loss_fn(prediction, y).item() * X.size(0)
            val_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    val_loss /= size
    accuracy = val_correct / size
    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss, accuracy
