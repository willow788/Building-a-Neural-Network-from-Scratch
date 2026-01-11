import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            root="data", train=True, download=True, transform=transform
        ),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            root="data", train=False, download=True, transform=transform
        ),
        batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            root="data", train=False, download=True, transform=transform
        ),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
