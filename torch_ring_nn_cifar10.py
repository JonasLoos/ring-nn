import urllib.request
import os
from math import pi
import numpy as np
import torch
from torch.nn import functional as F, Module, Parameter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm
import wandb


def load_cifar10(batch_size: int) -> tuple[DataLoader, DataLoader]:
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar10_path = 'cifar-10-python.tar.gz'
    cifar10_dir = 'cifar-10-batches-py'

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

def ringify(x: torch.Tensor) -> torch.Tensor:
    """Handle the circular nature of the number ring."""
    return (x + pi) % (2*pi) - pi


def complex_mean(x: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    """Compute the mean angle, by summing the complex unit numbers and taking the resulting complex number's angle."""
    dir_x = ringify(torch.cos(x)).sum(dim=dim)
    dir_y = torch.sin(x).sum(dim=dim)
    return torch.atan2(dir_y, dir_x)


class RingFF(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = Parameter(torch.randn(input_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return complex_mean(ringify(x.unsqueeze(-1) - self.weight), dim=(-2,))


class RingConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        self.weight = Parameter(torch.randn(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size))
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            # For (B, C, H, W) tensor, pad the last two dimensions (H, W)
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # Extract patches: (B, C, 1, H', W', kernel_h, kernel_w)
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unsqueeze(2)

        # Subtract and compute complex mean over input channels and kernel dimensions
        diff = ringify(x - self.weight)
        return complex_mean(diff, dim=(1, 5, 6))


def pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Pool input x (B, C, H, W) by applying complex_mean over non-overlapping patches
    of size kernel_size x kernel_size.
    """
    B, C, H, W = x.shape
    out_h = H // kernel_size
    out_w = W // kernel_size
    # Reshape into patches for pooling
    x = x.view(B, C, out_h, kernel_size, out_w, kernel_size)
    # Pool: apply complex_mean over the patch dims (-3, -1) = (kernel_size, kernel_size)
    return complex_mean(x, dim=(-3, -1))


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
    batch_size = 10
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

            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy.item(),
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
            })

    wandb.finish()


if __name__ == "__main__":
    train()
