import numpy as np
import math
import urllib.request
import os
import gzip
import csv
from tensor import RingTensor, RealTensor



class Dataloader:
    def __init__(self, x, y, batch_size=1, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.perm = np.random.permutation(len(x)) if shuffle else np.arange(len(x))
        self.i = 0

    def __iter__(self):
        if self.shuffle:
            self.perm = np.random.permutation(len(self.x))
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.x):
            raise StopIteration
        indices = self.perm[self.i:self.i+self.batch_size]
        xs = [self.x[j] for j in indices]
        ys = [self.y[j] for j in indices]
        self.i += self.batch_size
        return (
            xs[0].__class__.stack(*xs),
            ys[0].__class__.stack(*ys)
        )

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


def load_mnist(batch_size=1):
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = 'mnist.npz'

    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)

    def convert_x(data):
        return [RingTensor(x / 255).reshape((28, 28, 1)) for x in data]

    def convert_y(data):
        result = []
        for y in data:
            tmp = np.zeros((10,))
            tmp[y] = 1
            result.append(RealTensor(tmp))
        return result

    data = np.load(mnist_path)
    x_train, y_train = convert_x(data['x_train']), convert_y(data['y_train'])
    x_test, y_test = convert_x(data['x_test']), convert_y(data['y_test'])

    return (
        Dataloader(x_train, y_train, batch_size=batch_size),
        Dataloader(x_test, y_test, batch_size=batch_size),
    )



def load_cifar10(batch_size=1):
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
        x_train.extend(batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        y_train.extend(batch[b'labels'])

    # Load test data
    test_batch = unpickle(f'{cifar10_dir}/test_batch')
    x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = test_batch[b'labels']

    def convert_x(data):
        return [RingTensor(x / 255) for x in data]

    def convert_y(data):
        result = []
        for y in data:
            tmp = np.zeros((10,))
            tmp[y] = 1
            result.append(RealTensor(tmp))
        return result

    x_train, y_train = convert_x(np.array(x_train)), convert_y(y_train)
    x_test, y_test = convert_x(np.array(x_test)), convert_y(y_test)

    return (
        Dataloader(x_train, y_train, batch_size=batch_size),
        Dataloader(x_test, y_test, batch_size=batch_size),
    )


def load_higgs(batch_size=1, train_size=100000, test_size=10000):
    """
    Load the Higgs Boson dataset from UCI repository.

    The dataset has 28 features and binary classification (signal vs background).
    We'll use a subset for faster training/testing.
    """
    higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    higgs_path = 'HIGGS.csv.gz'

    if not os.path.exists(higgs_path):
        print("Downloading Higgs Boson dataset...")
        urllib.request.urlretrieve(higgs_url, higgs_path)

    print("Loading Higgs Boson dataset...")

    # Read the compressed CSV file
    rows = []
    total_read = 0

    with gzip.open(higgs_path, 'rt') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            rows.append([float(x) for x in row])
            total_read += 1
            if total_read >= train_size + test_size:
                break

    # Convert to numpy array
    data = np.array(rows)

    # Split features and labels
    # First column is the class label (1 for signal, 0 for background)
    y_data = data[:, 0]  # Labels
    x_data = data[:, 1:]  # Features (28 features)

    # Normalize features
    x_mean = np.mean(x_data, axis=0)
    x_std = np.std(x_data, axis=0)
    x_data = (x_data - x_mean) / (x_std + 1e-8)  # Add small epsilon to avoid division by zero

    # Split into train and test
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    x_test = x_data[train_size:train_size + test_size]
    y_test = y_data[train_size:train_size + test_size]

    def convert_x(data):
        return [RingTensor(x) for x in data]

    def convert_y(data):
        result = []
        for y in data:
            # Convert to one-hot encoding for binary classification
            tmp = np.zeros((2,))
            tmp[int(y)] = 1
            result.append(RealTensor(tmp))
        return result

    x_train, y_train = convert_x(x_train), convert_y(y_train)
    x_test, y_test = convert_x(x_test), convert_y(y_test)

    print(f"Loaded Higgs dataset: {len(x_train)} training samples, {len(x_test)} test samples")
    print(f"Features: {x_train[0].shape[0]}, Classes: 2 (signal/background)")

    return (
        Dataloader(x_train, y_train, batch_size=batch_size),
        Dataloader(x_test, y_test, batch_size=batch_size),
    )
