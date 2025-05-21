import numpy as np
import math
import urllib.request
import os
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
        return [RingTensor(x / 255).reshape((784, 1)) for x in data]

    def convert_y(data):
        result = []
        for y in data:
            tmp = np.zeros((10, 1))
            tmp[y, 0] = 1
            result.append(RealTensor(tmp))
        return result

    data = np.load(mnist_path)
    x_train, y_train = convert_x(data['x_train']), convert_y(data['y_train'])
    x_test, y_test = convert_x(data['x_test']), convert_y(data['y_test'])

    return (
        Dataloader(x_train, y_train, batch_size=batch_size),
        Dataloader(x_test, y_test, batch_size=batch_size),
    )
