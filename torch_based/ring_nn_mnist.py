import urllib.request
import os
from math import pi

import numpy as np
import torch
from torch.nn import functional as F, Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm

from lib_ring_nn2 import RingFF, RingConv2dCUDA as RingConv2d
# from lib_ring_nn import RingFF, RingConv2dCUDA as RingConv2d


def load_mnist(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load the MNIST dataset."""
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = '../mnist.npz'

    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)
    data = np.load(mnist_path)

    # Load and preprocess training data
    x_train = torch.from_numpy(data['x_train']).reshape(-1, 1, 28, 28).float() / 255.0 * pi  # Normalize to [0, pi]
    y_train = torch.from_numpy(data['y_train']).long()
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    # Load and preprocess test data
    x_test = torch.from_numpy(data['x_test']).reshape(-1, 1, 28, 28).float() / 255.0 * pi  # Normalize to [0, pi]
    y_test = torch.from_numpy(data['y_test']).long()
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader)



class RingNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RingConv2d(1, 10, 3, 2, 1)
        self.conv2 = RingConv2d(10, 20, 3, 2, 0)
        self.conv3 = RingConv2d(20, 20, 3, 2, 1)
        self.ff = RingFF(20*3*3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.ff(x)
        return torch.sin(x)


def train():
    # if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    device = torch.device("cpu")
    model = RingNN().to(device)
    # model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_dl, test_dl = load_mnist(batch_size=256)

    for epoch in (t:=trange(100, desc="Epoch")):
        for batch in tqdm(train_dl, desc="Train", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(test_dl, desc="Test", leave=False):
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_loss += F.cross_entropy(pred, y, reduction='sum').item()
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += len(y)
            test_loss /= total
            test_accuracy = correct / total
            t.set_postfix(test_loss=test_loss, test_accuracy=test_accuracy)



if __name__ == "__main__":
    train()
