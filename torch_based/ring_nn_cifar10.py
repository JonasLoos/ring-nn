import urllib.request
import os
from math import pi
import numpy as np
import torch
from torch.nn import functional as F, Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import wandb

from lib_ring_nn import RingFF, RingConv2dFused as RingConv2d, pool2d


def load_cifar10(batch_size: int) -> tuple[DataLoader, DataLoader]:
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar10_path = '../cifar-10-python.tar.gz'
    cifar10_dir = '../cifar-10-batches-py'

    if not os.path.exists(cifar10_dir):
        if not os.path.exists(cifar10_path):
            print("Downloading CIFAR-10 dataset...")
            urllib.request.urlretrieve(cifar10_url, cifar10_path)

        print("Extracting CIFAR-10 dataset...")
        import tarfile
        with tarfile.open(cifar10_path) as tar:
            tar.extractall()

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # Load training data
    x_train = []
    y_train = []
    for i in range(1, 6):
        batch = unpickle(f'{cifar10_dir}/data_batch_{i}')
        x_train.append(batch[b'data'].reshape(-1, 3, 32, 32) / 255.0 * pi)
        y_train.append(batch[b'labels'])

    # Concatenate all batches and convert to tensors
    x_train = torch.tensor(np.concatenate(x_train, axis=0), dtype=torch.float32)
    y_train = torch.tensor(np.concatenate(y_train, axis=0), dtype=torch.long)

    # Load test data
    test_batch = unpickle(f'{cifar10_dir}/test_batch')
    x_test = torch.tensor(test_batch[b'data'].reshape(-1, 3, 32, 32) / 255.0 * pi, dtype=torch.float32)
    y_test = torch.tensor(test_batch[b'labels'], dtype=torch.long)

    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
    )


class RingNN(Module):
    def __init__(self):
        super().__init__()
        self.stem = RingConv2d(3, 64, 3, 1, 1)  # 32 -> 32
        self.s1c1 = RingConv2d(64, 64, 3, 1, 1)  # 32 -> 32
        self.s1c2 = RingConv2d(64, 64, 3, 1, 1)  # 32 -> 32
        self.s1c3 = RingConv2d(64, 64, 3, 1, 1)  # 32 -> 32
        self.s1c4 = RingConv2d(64, 64, 3, 1, 1)  # 32 -> 32

        self.s2c1 = RingConv2d(64, 128, 3, 2, 1)  # 32 -> 16, downsample
        self.s2c2 = RingConv2d(128, 128, 3, 1, 1)  # 16 -> 16
        self.s2s1 = RingConv2d(64, 128, 1, 1, 0)  # 16 -> 16, skip
        self.s2c3 = RingConv2d(128, 128, 3, 1, 1)  # 16 -> 16
        self.s2c4 = RingConv2d(128, 128, 3, 1, 1)  # 16 -> 16

        self.s3c1 = RingConv2d(128, 256, 3, 2, 1)  # 16 -> 8, downsample
        self.s3c2 = RingConv2d(256, 256, 3, 1, 1)  # 8 -> 8
        self.s3s2 = RingConv2d(128, 256, 1, 1, 0)  # 8 -> 8, skip
        self.s3c3 = RingConv2d(256, 256, 3, 1, 1)  # 8 -> 8
        self.s3c4 = RingConv2d(256, 256, 3, 1, 1)  # 8 -> 8

        self.ff = RingFF(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        x = self.stem(x)
        # stage 1
        tmp = x
        x = self.s1c1(x)
        x = self.s1c2(x)
        x += tmp
        tmp = x
        x = self.s1c3(x)
        x = self.s1c4(x)
        x += tmp
        # stage 2
        tmp = x
        x = self.s2c1(x)
        x = self.s2c2(x)
        x += self.s2s1(pool2d(tmp, 2))
        tmp = x
        x = self.s2c3(x)
        x = self.s2c4(x)
        x += tmp
        # stage 3
        tmp = x
        x = self.s3c1(x)
        x = self.s3c2(x)
        x += self.s3s2(pool2d(tmp, 2))
        tmp = x
        x = self.s3c3(x)
        x = self.s3c4(x)
        x += tmp
        tmp = x
        # final layer
        x = pool2d(x, 8).flatten(1)
        x = self.ff(x)

        return torch.sin(x)


def train():
    epochs = 10
    batch_size = 128
    lr = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RingNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dl, test_dl = load_cifar10(batch_size=batch_size)

    wandb.init(project="ring-nn-cifar10-torch")
    wandb.config.update({
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    })
    wandb.watch(model, log="all")

    num_train_samples = 0
    for epoch in (t:=trange(epochs, desc="Epoch")):
        for batch in tqdm(train_dl, desc="Train", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = (pred.argmax(dim=1) == y).float().mean()

            num_train_samples += x.shape[0]
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy.item(),
                "num_train_samples": num_train_samples,
                "epoch": epoch,
            })

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
            wandb.log({
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "num_train_samples": num_train_samples,
                "epoch": epoch,
            })

    wandb.finish()


if __name__ == "__main__":
    train()
