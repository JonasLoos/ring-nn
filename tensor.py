import numpy as np
from typing import Self, TypeVar
from functools import wraps
import contextlib


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
        # Disable gradient computation if in no_grad context
        self._rg = requires_grad and not _no_grad
        self._backward = None
        self._prev = set()
        self.reset_grad()

    def as_float(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def reset_grad(self):
        self._grad = np.zeros_like(self.data, dtype=np.float32) if self._rg else None

    def set_to(self, other):
        self.data = other.data.copy()
        self._grad = other._grad.copy() if self._grad is not None else None
        self._rg = other._rg
        self.backward = None
        self._prev = set()
        return self

    @property
    def sign(self) -> np.ndarray:
        return np.sign(self.data)

    @convert_other
    def __add__(self, other: Selflike) -> Self:
        out = self.__class__(raw_data=self.data + other.data, requires_grad=self._rg or other._rg)

        if not _no_grad:
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
        out = self.__class__(raw_data=self.data.sum(axis=axis, keepdims=keepdims).astype(self.dtype), requires_grad=self._rg)

        if not _no_grad:
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
        out = self.__class__(raw_data=self.data.mean(axis=axis, keepdims=keepdims).astype(self.dtype), requires_grad=self._rg)

        if not _no_grad:
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

        if not _no_grad:
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

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    self._grad += out._grad.reshape(self.data.shape)
            out._backward = _backward
        return out

    def unsqueeze(self, axis):
        out = self.__class__(raw_data=np.expand_dims(self.data, axis=axis), requires_grad=self._rg)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    self._grad += np.squeeze(out._grad, axis=axis)
            out._backward = _backward
        return out

    def squeeze(self, axis):
        out = self.__class__(raw_data=np.squeeze(self.data, axis=axis), requires_grad=self._rg)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    self._grad += np.expand_dims(out._grad, axis=axis)
            out._backward = _backward
        return out

    def sliding_window_2d(self, window_size: int, padding: int = 0, stride: int = 1) -> Self:
        # input shape: (batch, height, width, channels) -> output shape: (batch, new_height, new_width, channels, window_size, window_size)
        data = np.pad(self.data, ((0,0), (padding, padding), (padding, padding), (0,0)), mode='constant')
        data = np.lib.stride_tricks.sliding_window_view(data, (window_size, window_size), axis=(-3, -2))
        data = data[:, ::stride, ::stride, :, :, :].copy()
        out = self.__class__(raw_data=data, requires_grad=self._rg)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    # shape: (batch, out_h, out_w, channels, window_size, window_size)
                    B, H, W, C, _, _ = out._grad.shape

                    # Create gradient tensor for padded input
                    padded_h = self.shape[1] + 2 * padding
                    padded_w = self.shape[2] + 2 * padding
                    grad_pad = np.zeros((B, padded_h, padded_w, C), dtype=np.float32)

                    # Accumulate gradients
                    for h_start in range(window_size):
                        for w_start in range(window_size):
                            h_end = min(padded_h, h_start + H * stride)
                            w_end = min(padded_w, w_start + W * stride)
                            h = (h_end - h_start + stride - 1) // stride
                            w = (w_end - w_start + stride - 1) // stride
                            if h > 0 and w > 0:
                                grad_pad[:, h_start:h_end:stride, w_start:w_end:stride, :] += out._grad[:, :h, :w, :, h_start, w_start]

                    self._grad += grad_pad[:, padding:-padding, padding:-padding, :] if padding > 0 else grad_pad
            out._backward = _backward
        return out

    @classmethod
    def stack(cls, *tensors, axis=0):
        out = cls(raw_data=np.stack([t.data for t in tensors], axis=axis), requires_grad=any(t._rg for t in tensors))

        if not _no_grad:
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


# Global flag to disable gradient computation
_no_grad = False

@contextlib.contextmanager
def no_grad():
    """Context manager to disable gradient computation"""
    global _no_grad
    old_no_grad = _no_grad
    _no_grad = True
    try:
        yield
    finally:
        _no_grad = old_no_grad


class RingTensor(Tensor):
    # dtype = np.int8
    dtype = np.int16
    # dtype = np.int32
    min_value = np.iinfo(dtype).min
    max_value = np.iinfo(dtype).max
    # [min, max] corresponds to [-1, 1], with -1 and 1 being next to each other the number ring
    # the implementation assumes that -min_value is roughly equal to max_value

    def __init__(self, data=None, *, raw_data=None, requires_grad=False):
        if data is not None and raw_data is not None:
            raise ValueError("Only one of data or raw_data can be provided")
        if data is not None:
            # convert data from [-1, 1] to [min, max]
            raw_data = (np.array(data) * -self.min_value).clip(self.min_value, self.max_value).astype(self.dtype)
        super().__init__(raw_data=raw_data, requires_grad=requires_grad)

    def as_float(self) -> np.ndarray:
        return self.data.astype(np.float32) / -self.min_value

    def poly(self, order: float) -> Self:
        # polynomial activation: |x|^order * sign(x)
        out = RingTensor(np.abs(self.as_float() + 1e-20)**order * self.sign, requires_grad=self._rg)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    self._grad += out._grad * order * (np.abs(self.as_float()) + 1e-20)**(order-1)
            out._backward = _backward
        return out

    def poly_sigmoid(self, order: float, slope: float) -> Self:
        out = RingTensor((1 + slope) * self.as_float() - slope * self.sign * (np.abs(self.as_float() + 1e-20)**order), requires_grad=self._rg)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    # TODO: is a *self.sign missing here?
                    self._grad += out._grad * (1 + slope) - slope * out._grad * order * (np.abs(self.as_float() + 1e-20)**(order-1))
            out._backward = _backward
        return out

    def sin(self) -> Self:
        # double sigmoid-like activation: (sin(x*pi - pi/2) + 1) * sign(x) * 0.5
        activation = (np.sin(self.as_float()*np.pi - np.pi/2) + 1) * self.sign * 0.5
        out = RingTensor(activation, requires_grad=self._rg)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    # derivative of sin activation: pi * cos(x*pi - pi/2) * sign(x)
                    self._grad += out._grad * np.pi * np.cos(self.as_float()*np.pi - np.pi/2) * self.sign * 0.5
            out._backward = _backward
        return out

    @classmethod
    def rand(cls, shape, requires_grad=False):
        out = cls(raw_data=np.random.randint(cls.min_value, cls.max_value, size=shape, dtype=cls.dtype), requires_grad=requires_grad)
        return out

    def real(self) -> Self:
        out = RealTensor(self.as_float(), requires_grad=self._rg)

        if not _no_grad:
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

        if not _no_grad:
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

        if not _no_grad:
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

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if self._rg:
                    self._grad += out._grad * self.sign
            out._backward = _backward
        return out

    def cross_entropy(self, other: Selflike) -> Self:
        logits = self.as_float().reshape(self.shape[0], -1)
        tgt = other.as_float().reshape(other.shape[0], -1)
        n = logits.shape[0]

        # numerically-stable soft-max
        shift = logits - logits.max(axis=1, keepdims=True)
        exps  = np.exp(shift)
        probs = exps / exps.sum(axis=1, keepdims=True)

        # cross-entropy loss
        loss_val = -np.sum(tgt * np.log(probs + 1e-20)) / n
        out = RealTensor(loss_val, requires_grad=self._rg or other._rg)

        if not _no_grad:
            out._prev = {self, other}

            def _backward():
                if self._rg:
                    grad = out._grad * (probs - tgt) / n
                    if self.shape != grad.shape:
                        grad = grad.reshape(self.shape)
                    self._grad += grad

                if other._rg:
                    grad = out._grad * (-np.log(probs + 1e-20) / n)
                    if other.shape != grad.shape:
                        grad = grad.reshape(other.shape)
                    other._grad += grad

            out._backward = _backward
        return out
