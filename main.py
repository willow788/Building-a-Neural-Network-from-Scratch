import torch
import torch.nn as nn
from model import NeuralNet
from dataloader import get_data_loaders
from train import train, evaluate, validate

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Data loaders
train_loader, val_loader, test_loader = get_data_loaders(batch_size=128)
print("Number of batches in training data loader:")
print(len(train_loader))
print("Number of batches in validation data loader:")
print(len(val_loader))
print("Number of batches in test data loader:")
print(len(test_loader))

# Model, Loss & Optimizer
model = NeuralNet().to(device)
print(model)
loss_fn = nn.CrossEntropyLoss()
print(loss_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(optimizer)

# Training loop
epochs = 30  # For example
for epoch in range(epochs):
    print(f"running epoch {epoch+1}\n-------------------")
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device)
    print(f"Train: Loss={train_loss:.4f}, Accuracy={train_acc*100:.2f}%")
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)
    print(f"Validation: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")
    test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
    print(f"Test: Loss={test_loss:.4f}, Accuracy={test_acc*100:.2f}%\n")
