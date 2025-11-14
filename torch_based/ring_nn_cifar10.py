from pathlib import Path
from datetime import datetime
import urllib.request
import os
from math import pi
import tarfile
import pickle
from time import time

import numpy as np
import torch
from torch.nn import functional as F, Module, ModuleList
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from tqdm import trange, tqdm
import wandb

from lib_ring_nn import RingFF, RingConv2dCUDA as RingConv2d, pool2d


def load_cifar10(batch_size: int) -> tuple[DataLoader, DataLoader]:
    cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    cifar10_path = '../cifar-10-python.tar.gz'
    cifar10_dir = '../cifar-10-batches-py'

    if not os.path.exists(cifar10_dir):
        if not os.path.exists(cifar10_path):
            print("Downloading CIFAR-10 dataset...")
            urllib.request.urlretrieve(cifar10_url, cifar10_path)

        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(cifar10_path) as tar:
            tar.extractall()

    def unpickle(file):
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


#########################
# Model definitions below
#########################

# Doesn't seem to really work yet
class RingNNOld(Module):
    """Resnet inspired architecture."""
    def __init__(self):
        super().__init__()
        self.stem = RingConv2d(3, 32, 3, 1, 1)  # 32 -> 32
        self.s1c1 = RingConv2d(32, 32, 3, 1, 1)  # 32 -> 32
        self.s1c2 = RingConv2d(32, 32, 3, 1, 1)  # 32 -> 32
        # self.s1c3 = RingConv2d(32, 32, 3, 1, 1)  # 32 -> 32
        # self.s1c4 = RingConv2d(32, 32, 3, 1, 1)  # 32 -> 32

        self.s2c1 = RingConv2d(32, 64, 3, 2, 1)  # 32 -> 16, downsample
        self.s2c2 = RingConv2d(64, 64, 3, 1, 1)  # 16 -> 16
        self.s2s1 = RingConv2d(32, 64, 1, 1, 0)  # 16 -> 16, skip
        # self.s2c3 = RingConv2d(64, 64, 3, 1, 1)  # 16 -> 16
        # self.s2c4 = RingConv2d(64, 64, 3, 1, 1)  # 16 -> 16

        self.s3c1 = RingConv2d(64, 128, 3, 2, 1)  # 16 -> 8, downsample
        self.s3c2 = RingConv2d(128, 128, 3, 1, 1)  # 8 -> 8
        self.s3s2 = RingConv2d(64, 128, 1, 1, 0)  # 8 -> 8, skip
        # self.s3c3 = RingConv2d(128, 128, 3, 1, 1)  # 8 -> 8
        # self.s3c4 = RingConv2d(128, 128, 3, 1, 1)  # 8 -> 8

        self.ff = RingFF(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        x = self.stem(x)  # input conv
        # stage 1
        tmp = x
        x = self.s1c1(x)  # conv
        x = self.s1c2(x)  # conv
        x = ringify(x + tmp)  # skip connection
        # tmp = x
        # x = self.s1c3(x)
        # x = self.s1c4(x)
        # x = ringify(x + tmp)
        # stage 2
        tmp = x
        x = self.s2c1(x)  # conv
        x = self.s2c2(x)  # conv
        x = ringify(x + self.s2s1(pool2d(tmp, 2)))  # skip connection with 2x2 average pooling
        # tmp = x
        # x = self.s2c3(x)
        # x = self.s2c4(x)
        # x = ringify(x + tmp)
        # stage 3
        tmp = x
        x = self.s3c1(x)  # conv
        x = self.s3c2(x)  # conv
        x = ringify(x + self.s3s2(pool2d(tmp, 2)))  # skip connection with 2x2 average pooling
        # tmp = x
        # x = self.s3c3(x)
        # x = self.s3c4(x)
        # x = ringify(x + tmp)
        # final layer
        x = pool2d(x, 8).flatten(1)  # global average pooling
        x = self.ff(x)

        return torch.sin(x)


class RingNNStage(Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels * 2
        self.c1 = RingConv2d(in_channels, out_channels, 3, 2, 1)
        # self.c2 = RingConv2d(out_channels, out_channels, 3, 1, 1)
        self.s1 = RingConv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.c1(x)
        # x = self.c2(x)
        x = ringify(x + pool2d(self.s1(skip), 2))
        return x


# Doesn't seem to really work yet
class RingNN(Module):
    def __init__(self):
        super().__init__()
        self.stages = ModuleList([
            RingConv2d(3, 16, 3, 1, 1),
            RingNNStage(16),
            RingNNStage(32),
            RingNNStage(64),
            RingConv2d(128, 256, 2, 2, 0),
        ])
        self.ff = RingFF(256*2*2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = x.flatten(1)
        x = self.ff(x)
        return torch.sin(x)


# works at least a bit
class RingNNSimple(Module):
    def __init__(self):
        super().__init__()
        self.convs = ModuleList([
            RingConv2d(3, 32, 3, 1, 1),  # stem
            RingConv2d(32, 64, 3, 2, 1),  # down conv -> 16x16
            RingConv2d(64, 64, 3, 1, 1),  # conv
            RingConv2d(64, 128, 3, 2, 1),  # down conv -> 8x8
            RingConv2d(32, 128, 1, 1, 0),  # skip conv
            RingConv2d(128, 128, 3, 1, 1),  # conv
            RingConv2d(128, 256, 3, 2, 1),  # down conv -> 4x4
            RingConv2d(256, 256, 3, 1, 1),  # conv
            RingConv2d(256, 256, 2, 2, 0),  # down conv -> 2x2
            RingConv2d(128, 256, 1, 1, 0),  # skip conv
        ])
        self.ff1 = RingFF(256*2*2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs[0](x)
        tmp = x
        x = self.convs[1](x)
        x = self.convs[2](x)
        x = self.convs[3](x)
        x = x + pool2d(self.convs[4](tmp), 4)
        tmp = x
        x = self.convs[5](x)
        x = self.convs[6](x)
        x = self.convs[7](x)
        x = self.convs[8](x)
        x = x + pool2d(self.convs[9](tmp), 4)
        x = x.flatten(1)
        x = self.ff1(x)
        return torch.sin(x)


def train():
    epochs = 500
    batch_size = 512
    lr = 0.03
    lr_decay = 0.995
    compile = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RingNNSimple().to(device)
    if compile:
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)

    # torch.cuda.reset_peak_memory_stats()

    train_dl, test_dl = load_cifar10(batch_size=batch_size)

    wandb.init(project="ring-nn-cifar10-torch")
    wandb.config.update({
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "lr": lr,
        "lr_decay": lr_decay,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(device),
        "compiled": compile,
    })
    wandb.watch(model, log="all")

    num_train_samples = 0
    for epoch in (t:=trange(epochs, desc="Epoch")):
        for batch in tqdm(train_dl, desc="Train", leave=False):
            start_time = time()
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time()
            accuracy = (pred.argmax(dim=1) == y).float().mean()

            num_train_samples += x.shape[0]
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": accuracy.item(),
                "num_train_samples": num_train_samples,
                "epoch": epoch,
                # "vram_usage": torch.cuda.max_memory_allocated() / 1024**3,
                "train_batch_time": end_time - start_time,
            })
            # torch.cuda.reset_peak_memory_stats()

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
                "learning_rate": scheduler.get_last_lr()[0],
            })

        scheduler.step()  # Decay learning rate

    # Save model to wandb
    model_path = Path(f"models/ring_nn_cifar10_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{wandb.run.id}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()


if __name__ == "__main__":
    train()
