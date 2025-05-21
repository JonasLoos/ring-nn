import numpy as np
import math
from typing import Self, TypeVar
import urllib.request
import os
import pickle
from functools import wraps


def convert_other(f):
    '''decorator to convert other to a Tensor if it is not already'''
    @wraps(f)
    def wrapper(self, other):
        if isinstance(other, self.__class__):
            return f(self, other)
        else:
            return f(self, self.__class__(other))
    return wrapper


Selflike = TypeVar('Selflike', bound = Self | int | float | np.ndarray)


class Tensor:
    dtype = None
    min_value = None
    max_value = None

    def __init__(self, *, raw_data: np.ndarray | None = None, requires_grad=False):
        if self.dtype is None: raise ValueError("Use a subclass of Tensor")
        if raw_data is None: raise ValueError("raw_data must be provided")
        self.data = raw_data
        self._rg = requires_grad
        self._backward = None
        self._prev = set()
        self.reset_grad()

    def as_float(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def reset_grad(self):
        self._grad = np.zeros_like(self.data, dtype=np.float32) if self._rg else None

    @property
    def sign(self) -> np.ndarray:
        return np.sign(self.data)

    @convert_other
    def __add__(self, other: Selflike) -> Self:
        out = self.__class__(raw_data=self.data + other.data, requires_grad=self._rg or other._rg)
        out._prev = {self, other}

        def _backward():
            if self._rg:
                self._grad += Tensor._unbroadcast_gradient(out._grad, self.shape)
            if other._rg:
                other._grad += Tensor._unbroadcast_gradient(out._grad, other.shape)
        out._backward = _backward
        return out

    def __radd__(self, other: Selflike) -> Self:
        return self + other

    def sum(self, axis=None, keepdims=False):
        out = self.__class__(raw_data=self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                g = out._grad
                # restore the lost dimensions so broadcasting works
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                self._grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = self.__class__(raw_data=self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                g = out._grad / (self.size / out.size)  # scale by 1/N
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis=axis)
                self._grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def __sub__(self, other: Selflike) -> Self:
        return self + (-other)

    def __rsub__(self, other: Selflike) -> Self:
        return other + (-self)

    def __neg__(self) -> Self:
        out = self.__class__(raw_data=-self.data, requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad += -out._grad
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
    def full(cls, shape, value, requires_grad=False):
        return cls(np.full(shape, value, dtype=cls.dtype), requires_grad=requires_grad)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def reshape(self, shape):
        out = self.__class__(raw_data=self.data.reshape(shape), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # Reshape the gradient back to the original shape
                self._grad += out._grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def unsqueeze(self, axis):
        out = self.__class__(raw_data=np.expand_dims(self.data, axis=axis), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad += np.squeeze(out._grad, axis=axis)
        out._backward = _backward
        return out

    def squeeze(self, axis):
        out = self.__class__(raw_data=np.squeeze(self.data, axis=axis), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad += np.expand_dims(out._grad, axis=axis)
        out._backward = _backward
        return out

    @classmethod
    def stack(cls, *tensors, axis=0):
        out = cls(raw_data=np.stack([t.data for t in tensors], axis=axis), requires_grad=any(t._rg for t in tensors))
        out._prev = set(tensors)

        def _backward():
            for i, t in enumerate(tensors):
                if t._rg:
                    # We need to slice the gradient corresponding to this tensor
                    slices = [slice(None)] * out._grad.ndim
                    slices[axis] = i
                    t._grad += Tensor._unbroadcast_gradient(out._grad[tuple(slices)], t.shape)
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


class RingTensor(Tensor):
    # dtype = np.int8
    dtype = np.int16
    min_value = np.iinfo(dtype).min
    max_value = np.iinfo(dtype).max
    # [min, max] corresponds to [-1, 1], with -1 and 1 being next to each other the number ring
    # the implementation assumes that -min_value is roughly equal to max_value

    def __init__(self, data=None, *, raw_data=None, requires_grad=False):
        if data is not None and raw_data is not None:
            raise ValueError("Only one of data or raw_data can be provided")
        if data is not None:
            # convert data from [-1, 1] to [min, max]
            raw_data = (data * -self.min_value).clip(self.min_value, self.max_value).astype(self.dtype)
        super().__init__(raw_data=raw_data, requires_grad=requires_grad)

    def as_float(self) -> np.ndarray:
        return self.data.astype(np.float32) / -self.min_value

    def poly(self, order: float) -> Self:
        # polynomial activation: |x|^order * sign(x)
        out = RingTensor(np.abs(self.as_float())**order * self.sign, requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad += out._grad * order * (np.abs(self.as_float()) + 1e-20)**(order-1)
        out._backward = _backward
        return out

    def sin(self) -> Self:
        # double sigmoid-like activation: (sin(x*pi - pi/2) + 1) * sign(x)
        activation = (np.sin(self.as_float()*np.pi - np.pi/2) + 1) * self.sign
        out = RingTensor(activation, requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # derivative of sin activation: pi * cos(x*pi - pi/2) * sign(x)
                self._grad += out._grad * np.pi * np.cos(self.as_float()*np.pi - np.pi/2) * self.sign
        out._backward = _backward
        return out

    @classmethod
    def rand(cls, shape, requires_grad=False):
        out = cls(raw_data=np.random.randint(cls.min_value, cls.max_value, size=shape, dtype=cls.dtype), requires_grad=requires_grad)
        return out

    def real(self) -> Self:
        out = RealTensor(self.as_float(), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad += out._grad
        out._backward = _backward
        return out


class RealTensor(Tensor):
    dtype = np.float32
    min_value = -np.inf
    max_value = np.inf

    def __init__(self, data=None, *, raw_data: np.ndarray | None = None, requires_grad=False):
        if data is not None and raw_data is not None:
            raise ValueError("Only one of data or raw_data can be provided")
        if data is not None:
            raw_data = np.array(data, dtype=self.dtype)
        super().__init__(raw_data=raw_data, requires_grad=requires_grad)

    def as_float(self) -> np.ndarray:
        return self.data

    @convert_other
    def __mul__(self, other: Selflike) -> Self:
        out = RealTensor(self.data * other.data, requires_grad=self._rg or other._rg)
        out._prev = {self, other}

        def _backward():
            if self._rg:
                self._grad += out._grad * other.data
            if other._rg:
                other._grad += out._grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other: Selflike) -> Self:
        return self * other
    
    def __truediv__(self, other: Selflike) -> Self:
        return self * other ** -1
    
    def __rtruediv__(self, other: Selflike) -> Self:
        return other * self ** -1

    @convert_other
    def __pow__(self, other: Selflike) -> Self:
        out = RealTensor(self.data ** other.data, requires_grad=self._rg or other._rg)
        out._prev = {self, other}

        def _backward():
            if self._rg:
                self._grad += out._grad * other.data * self.data ** (other.data - 1)
            if other._rg:
                other._grad += out._grad * np.log(self.data) * self.data ** other.data
        out._backward = _backward
        return out

    def __rpow__(self, other: Selflike) -> Self:
        return other ** self

    def abs(self) -> Self:
        out = RealTensor(np.abs(self.data), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                self._grad += out._grad * self.sign
        out._backward = _backward
        return out

    def cross_entropy(self, other: Selflike) -> Self:
        logits = self.as_float()
        tgt = other.as_float()
        n = logits.shape[0]

        # numerically-stable soft-max
        shift = logits - logits.max(axis=1, keepdims=True)
        exps  = np.exp(shift)
        probs = exps / exps.sum(axis=1, keepdims=True)

        # cross-entropy loss
        loss_val = -np.sum(tgt * np.log(probs + 1e-20)) / n
        out = RealTensor(loss_val, requires_grad=self._rg or other._rg)
        out._prev = {self, other}

        def _backward():
            if self._rg:
                self._grad += out._grad * (probs - tgt) / n
            if other._rg:
                other._grad += out._grad * (-np.log(probs + 1e-20) / n)
        out._backward = _backward
        return out


class RingNN:
    def __init__(self, sizes):
        self.weights = [RingTensor.rand((sizes[i], sizes[i+1]), requires_grad=True) for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for w in self.weights:
            x = (x - w).poly(1/2).mean(axis=-2).unsqueeze(-1)
        return 1 - x.real().abs()

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


class SGD:
    def __init__(self, nn, lr, lr_decay):
        self.nn = nn
        self.lr = lr
        self.lr_decay = lr_decay

    def __call__(self):
        abs_update_float = 0
        abs_update_final = 0
        updates_float = []
        updates_final = []
        for w in self.nn.weights:
            update = w._grad * self.lr
            update_final = (update.clip(-1, 1) * -RingTensor.min_value).astype(RingTensor.dtype)
            w.data -= update_final
            w.reset_grad()
            abs_update_float += np.abs(update).mean()
            abs_update_final += np.abs(update_final).mean() / -RingTensor.min_value
            updates_float.append(update)
            updates_final.append(update_final)
        self.lr *= self.lr_decay
        return {
            'abs_update_float': abs_update_float,
            'abs_update_final': abs_update_final,
            'updates_float': updates_float,
            'updates_final': updates_final
        }


class Adam:
    def __init__(self, nn, lr, lr_decay):
        self.nn = nn
        self.lr = lr
        self.lr_decay = lr_decay
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m = [np.zeros_like(w.data, dtype=np.float32) for w in self.nn.weights]
        self.v = [np.zeros_like(w.data, dtype=np.float32) for w in self.nn.weights]

    def __call__(self):
        self.t += 1
        abs_update_float = 0
        abs_update_final = 0
        for i, w in enumerate(self.nn.weights):
            if w._grad is None:
                continue

            grad = w._grad.astype(np.float32) # Ensure gradient is float32 for calculations

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Calculate the update
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Apply update to RingTensor (quantized)
            update_final = (update.clip(-1, 1) * -RingTensor.min_value).astype(RingTensor.dtype)
            w.data -= update_final
            
            w.reset_grad()

            abs_update_float += np.abs(update).mean()
            abs_update_final += np.abs(update_final).mean() / -RingTensor.min_value
        
        self.lr *= self.lr_decay
        return abs_update_float, abs_update_final


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

def print_frac(a, b):
    return f'{a:{len(str(b))}}/{b}'

def train(nn, epochs, lr, lr_decay, train_logs):
    train_dl, test_dl = load_mnist(batch_size=1000)

    # loss_fn = lambda a, b: ((a - b).abs() * (1 + 8*b)).mean()  # balanced loss
    # loss_fn = lambda a, b: ((a - b) ** 2).mean()  # MSE loss
    loss_fn = lambda a, b: a.cross_entropy(b)  # cross-entropy loss

    optimizer = SGD(nn, lr, lr_decay)
    # optimizer = Adam(nn, lr, lr_decay)

    for epoch in range(epochs):
        print("-" * 100)
        print(f"Epoch {print_frac(epoch+1, epochs)}")
        for i, (x, y) in enumerate(train_dl):
            pred = nn(x)
            loss = loss_fn(pred, y)
            accuracy = (pred.data.argmax(axis=-2) == y.abs().data.argmax(axis=-2)).mean()
            loss.backward()
            opt_logs = optimizer()
            print(f"\r[{print_frac(i+1, len(train_dl))}] Train loss: {loss.data.item():7.4f} | accuracy: {accuracy:6.2%} | avg. grad. change: {opt_logs['abs_update_final']:.2e} (f: {opt_logs['abs_update_float']:.2e}) | lr: {optimizer.lr:.2e}", end="")
            train_logs.append({
                'weights': [w.data.copy() for w in nn.weights],
                'loss': loss.data.item(),
                'accuracy': accuracy,
                'abs_update_float': opt_logs['abs_update_float'],
                'abs_update_final': opt_logs['abs_update_final'],
                'updates_float': opt_logs['updates_float'],
                'updates_final': opt_logs['updates_final'],
                'lr': optimizer.lr,
                'epoch': epoch,
                'i': i,
            })

        # Test on validation set
        test_loss = 0
        test_accuracy = 0
        for x, y in test_dl:
            test_loss = test_loss + loss_fn(nn(x), y).data.item()
            test_accuracy += (nn(x).data.argmax(axis=-2) == y.abs().data.argmax(axis=-2)).mean()
        print(f"\n{len(print_frac(i+1, len(train_dl)))*' '}   Test  loss: {test_loss / len(test_dl):7.4f} | accuracy: {test_accuracy / len(test_dl):6.2%}")

def ring_nn():
    nn = RingNN([784, 10])
    try:
        train_logs = []
        train(nn, epochs=10, lr=1e+3, lr_decay=0.995, train_logs=train_logs)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nSaving model...")
        nn.save('ring_nn.pkl')
        with open('train_logs.pkl', 'wb') as f:
            pickle.dump(train_logs, f)


if __name__ == '__main__':
    ring_nn()
