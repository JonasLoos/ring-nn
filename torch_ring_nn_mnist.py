import urllib.request
import os
from math import pi
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset



def load_mnist(batch_size: int) -> tuple[DataLoader, DataLoader]:
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = 'mnist.npz'

    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)

    data = np.load(mnist_path)
    
    # Load and preprocess training data
    x_train = torch.from_numpy(data['x_train']).float() / 255.0  # Normalize to [0, 1]
    x_train = x_train.reshape(-1, 784)  # Flatten to 784 features
    y_train = torch.from_numpy(data['y_train']).long()
    
    # Load and preprocess test data
    x_test = torch.from_numpy(data['x_test']).float() / 255.0  # Normalize to [0, 1]
    x_test = x_test.reshape(-1, 784)  # Flatten to 784 features
    y_test = torch.from_numpy(data['y_test']).long()
    
    # Create datasets
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader)



def ringify(x: torch.Tensor) -> torch.Tensor:
    return (x + pi) % (2*pi) - pi



def complex_mean(x: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    dir_x = ringify(torch.cos(x)).sum(dim=dim)
    dir_y = torch.sin(x).sum(dim=dim)
    return torch.atan2(dir_y, dir_x)



class RingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_weight = nn.Parameter(torch.randn(4, 1, 2, 2))
        self.conv2_weight = nn.Parameter(torch.randn(8, 4, 4, 4))
        self.ff_weight = nn.Parameter(torch.randn(288, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ringify(x).reshape(-1, 1, 28, 28)
        x = complex_mean(ringify(x.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5).unsqueeze(-1) - self.conv1_weight.permute(1, 2, 3, 0).unsqueeze(0).unsqueeze(0).unsqueeze(0)), dim=(-4, -3, -2))
        x = complex_mean(ringify(x.permute(0, 3, 1, 2).unfold(2, 4, 2).unfold(3, 4, 2).permute(0, 2, 3, 1, 4, 5).unsqueeze(-1) - self.conv2_weight.permute(1, 2, 3, 0).unsqueeze(0).unsqueeze(0).unsqueeze(0)), dim=(-4, -3, -2))
        x = x.flatten(1)
        x = complex_mean(ringify(x.unsqueeze(-1) - self.ff_weight), dim=(-2,))
        return torch.sin(x)



def train():
    model = RingNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dl, test_dl = load_mnist(batch_size=200)

    for epoch in range(10):
        for batch in train_dl:
            x, y = batch
            pred = model(x)
            loss = torch.nn.functional.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch in test_dl:
                x, y = batch
                pred = model(x)
                test_loss += torch.nn.functional.cross_entropy(pred, y, reduction='sum').item()
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += len(y)
            test_loss /= total
            test_accuracy = correct / total
            print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")



if __name__ == "__main__":
    train()
