import numpy as np
from typing import Self
import urllib.request
import os
import pickle


def default_tensor_backward_op():
    """Default backward operation for a Tensor, does nothing."""
    pass


class Tensor:
    dtype = np.int8
    min_value = -128
    max_value = 127

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=self.dtype)
        self._rg = requires_grad
        self._grad = None
        self.reset_grad()
        self._backward = None
        self._prev = set()

    def reset_grad(self):
        self._grad = np.zeros_like(self.data, dtype=np.float32) if self._rg else None

    def __add__(self, other: Self) -> Self:
        out = Tensor(self.data + other.data, requires_grad=self._rg or other._rg)
        out._prev = {self, other}

        def _backward():
            if self._rg:
                self._grad = self._grad + Tensor._unbroadcast_gradient(out._grad, self.data.shape)
            if other._rg:
                other._grad = other._grad + Tensor._unbroadcast_gradient(out._grad, other.data.shape)
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                g = out._grad
                # restore the lost dimensions so broadcasting works
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                self._grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                g = out._grad / (self.data.size / out.data.size)  # scale by 1/N
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                self._grad += np.broadcast_to(g, self.data.shape)

        out._backward = _backward
        return out


    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __neg__(self) -> Self:
        out = Tensor(-self.data.clip(-self.max_value, self.max_value), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad = self._grad - out._grad
        out._backward = _backward
        return out


    def sin(self) -> Self:
        # double sigmoid-like activation
        x = self.data.astype(np.float32) / (self.max_value - self.min_value)
        activation = ((np.sin(x*np.pi - np.pi/2) + 1) * np.sign(x) * self.max_value).clip(self.min_value, self.max_value)
        out = Tensor(activation, requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                x = self.data.astype(np.float32)
                local_grad = (np.cos(x*np.pi - np.pi/2) * np.pi * np.sign(x) * 0.5)
                self._grad = self._grad + out._grad * local_grad
        out._backward = _backward
        return out

    def abs(self) -> Self:
        out = Tensor(np.abs(self.data.clip(-self.max_value, self.max_value)), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad = self._grad + out._grad * np.sign(self.data)
        out._backward = _backward
        return out

    def softmin(self, axis=0) -> Self:
        # ---------- forward ----------
        abs_x = np.abs(self.data.astype(np.float32))              # |x|
        S     = abs_x.sum(axis=axis, keepdims=True)               # Σ|x|
        n     = abs_x / S                                         # normalised |x|
        exp_n = np.exp(-n)
        Z     = exp_n.sum(axis=axis, keepdims=True)
        y     = exp_n / Z                                         # soft-min
        out   = Tensor((y * self.max_value).clip(self.min_value, self.max_value).astype(self.dtype), requires_grad=self._rg)

        # store everything needed for grad
        cache = (abs_x, y, S, axis)
        out._prev = {self}

        # ---------- backward ----------
        def _backward():
            if self._rg:
                abs_x, y, S, ax = cache
                g_y  = out._grad.astype(np.float32) * self.max_value   # dL/dy

                dot  = np.sum(g_y * y, axis=ax, keepdims=True)
                g_n  = y * (dot - g_y)                                 # dL/dn

                dot_a = np.sum(g_n * abs_x, axis=ax, keepdims=True)
                g_abs = (S * g_n - dot_a) / (S * S)                    # dL/d|x|

                self._grad += g_abs * np.sign(self.data)               # back through |·|

        out._backward = _backward
        return out


    def backward(self):
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)
        self._grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            if node._backward:
                node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self._grad})"

    @classmethod
    def rand(cls, shape, requires_grad=False):
        return cls(np.random.randint(cls.min_value, cls.max_value, size=shape, dtype=cls.dtype), requires_grad=requires_grad)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # Reshape the gradient back to the original shape
                self._grad = self._grad + out._grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    @staticmethod
    def _unbroadcast_gradient(grad_output, original_input_shape):
        """
        Sums grad_output along axes to revert broadcasting, so it matches original_input_shape.
        """
        processed_grad = np.array(grad_output, copy=False) # Operate on a view
        shape_out = list(processed_grad.shape)
        shape_in = list(original_input_shape)

        # 1. Sum over leading axes in grad_output not present in original_input_shape
        delta_dims = len(shape_out) - len(shape_in)
        if delta_dims > 0:
            processed_grad = np.sum(processed_grad, axis=tuple(range(delta_dims)))
            shape_out = list(processed_grad.shape) # Update shape_out

        # 2. Sum over axes that were 1 in original_input_shape but >1 in processed_grad's shape
        axes_to_sum_expansion = []
        len_shape_out = len(shape_out)
        len_shape_in = len(shape_in)
        for i in range(len_shape_out): # Iterate based on current processed_grad dimensions
            out_axis_idx = len_shape_out - 1 - i # Current axis in processed_grad (from right)
            in_axis_idx = len_shape_in - 1 - i   # Corresponding axis in original_input_shape (from right)

            if in_axis_idx >= 0: # Check if original_input_shape has this dimension
                if shape_in[in_axis_idx] == 1 and shape_out[out_axis_idx] > 1:
                    axes_to_sum_expansion.append(out_axis_idx)

        if axes_to_sum_expansion:
            processed_grad = np.sum(processed_grad, axis=tuple(axes_to_sum_expansion), keepdims=True)

        return np.reshape(processed_grad, original_input_shape)


class RingNN:
    def __init__(self, sizes):
        self.weights = [Tensor.rand((sizes[i], sizes[i+1]), requires_grad=True) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for w in self.weights:
            x = (x - w).sin().mean(axis=0).reshape((-1, 1))
        return x.softmin()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        nn = RingNN([w.shape[0] for w in weights] + [weights[-1].shape[1]])
        nn.weights = weights
        return nn


def load_mnist():
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = 'mnist.npz'

    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)

    data = np.load(mnist_path)

    def convert_x(data):
        return [Tensor(x).reshape((784, 1)) for x in data]

    def convert_y(data):
        result = []
        for y in data:
            tmp = np.zeros((10, 1))
            tmp[y, 0] = Tensor.min_value
            result.append(Tensor(tmp))
        return result

    x_train, y_train = convert_x(data['x_train']), convert_y(data['y_train'])
    x_test, y_test = convert_x(data['x_test']), convert_y(data['y_test'])

    return x_train, y_train, x_test, y_test


def train(nn, lr=100000000, epochs=10):
    x_train, y_train, x_test, y_test = load_mnist()

    for epoch in range(epochs):
        print("-" * 100)
        print(f"Epoch {epoch}")
        print("-" * 100)
        loss = Tensor([0])
        accuracy = 0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            loss = (nn(x) - y).sum().abs()
            accuracy += nn(x).abs().data.argmax() == y.abs().data.argmax()
            if i % 500 == 0:
                avg_grandient_change = 0
                loss.backward()
                for w in nn.weights:
                    w.data = w.data + w._grad * lr
                    print(w._grad)
                    # print(np.abs((w._grad * lr).astype(np.int8)).sum() / np.prod(w.shape))
                    avg_grandient_change += np.abs((w._grad * lr).astype(np.int8)).sum() / np.prod(w.shape)
                    w.reset_grad()
                avg_grandient_change /= len(nn.weights)
                print(f"[{i:05}/{len(x_train)}]: loss: {loss.data.item():5} | accuracy: {accuracy / 500:.2%} | Avg. gradient change: {avg_grandient_change}")
                loss = Tensor([0])
                accuracy = 0

        # Test on validation set
        test_loss = Tensor([0])
        test_accuracy = 0
        for x, y in zip(x_test, y_test):
            test_loss = test_loss + (nn(x) - y).sum().abs()
            test_accuracy += nn(x).data.argmax() == y.data.argmax()
        print(f"\nTest loss: {test_loss.data.item()} | accuracy: {test_accuracy / len(x_test):.2%}")


def ring_nn():
    nn = RingNN([784, 10])
    try:
        train(nn, lr=100000000, epochs=10)
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving model...")
        nn.save('ring_nn.pkl')


if __name__ == '__main__':
    ring_nn()
