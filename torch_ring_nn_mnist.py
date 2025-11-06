import urllib.request
import os
from math import pi
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm


def load_mnist(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load the MNIST dataset."""
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = 'mnist.npz'

    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)

    data = np.load(mnist_path)

    # Load and preprocess training data
    x_train = torch.from_numpy(data['x_train']).reshape(-1, 1, 28, 28).float() / 255.0  # Normalize to [0, 1]
    y_train = torch.from_numpy(data['y_train']).long()

    # Load and preprocess test data
    x_test = torch.from_numpy(data['x_test']).reshape(-1, 1, 28, 28).float() / 255.0  # Normalize to [0, 1]
    y_test = torch.from_numpy(data['y_test']).long()

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader)


def ringify(x: torch.Tensor) -> torch.Tensor:
    """Handle the circular nature of the number ring."""
    return (x + pi) % (2*pi) - pi



def complex_mean(x: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    """Compute the mean angle, by summing the complex unit numbers and taking the resulting complex number's angle."""
    dir_x = ringify(torch.cos(x)).sum(dim=dim)
    dir_y = torch.sin(x).sum(dim=dim)
    return torch.atan2(dir_y, dir_x)


def ring_conv2d(x: torch.Tensor, weight: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Performs ring convolution by extracting patches and computing complex mean.

    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        kernel_size: Size of the convolution kernel (assumed square)
        stride: Stride of the convolution

    Returns:
        Output tensor of shape (B, H', W', out_channels)
    """
    # Extract patches: (B, C, H', W', kernel_h, kernel_w)
    x = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    # Rearrange to (B, H', W', C, kernel_h, kernel_w, 1)
    x = x.permute(0, 2, 3, 1, 4, 5).unsqueeze(-1)

    # Rearrange weight from (out_channels, in_channels, kernel_h, kernel_w)
    # to (1, 1, 1, in_channels, kernel_h, kernel_w, out_channels)
    weight = weight.permute(1, 2, 3, 0)[(None,) * 3]

    # Subtract and compute complex mean over input channels and kernel dimensions
    diff = ringify(x - weight)
    return complex_mean(diff, dim=(-4, -3, -2))


class RingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_weight = nn.Parameter(torch.randn(4, 1, 2, 2))
        self.conv2_weight = nn.Parameter(torch.randn(8, 4, 4, 4))
        self.ff_weight = nn.Parameter(torch.randn(288, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ringify(x).reshape(-1, 1, 28, 28)
        x = ring_conv2d(x, self.conv1_weight, kernel_size=2, stride=2)
        x = x.permute(0, 3, 1, 2)
        x = ring_conv2d(x, self.conv2_weight, kernel_size=4, stride=2)
        x = x.flatten(1)
        x = complex_mean(ringify(x.unsqueeze(-1) - self.ff_weight), dim=(-2,))
        return torch.sin(x)


def train():
    model = RingNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    train_dl, test_dl = load_mnist(batch_size=200)

    for epoch in (te:=trange(10, desc="Epoch")):
        for batch in (tb:=tqdm(train_dl, desc="Train", leave=False)):
            x, y = batch
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tb.set_postfix(loss=loss.item())

        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(test_dl, desc="Test", leave=False):
                x, y = batch
                pred = model(x)
                test_loss += F.cross_entropy(pred, y, reduction='sum').item()
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += len(y)
            test_loss /= total
            test_accuracy = correct / total
            te.set_postfix(test_loss=test_loss, test_accuracy=test_accuracy)



if __name__ == "__main__":
    train()
