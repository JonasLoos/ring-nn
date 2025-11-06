from math import pi
from abc import ABC, abstractmethod
import contextlib
from functools import wraps
from typing import Self, TypeGuard, TypeVar, Callable, TypeAlias

import torch
import numpy as np


# Set default device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Tensorlike: TypeAlias = "Tensor | int | float | torch.Tensor | np.ndarray"
TensorSubclass = TypeVar('TensorSubclass', bound="Tensor")

def convert_other(f: Callable[[TensorSubclass, TensorSubclass], TensorSubclass] ) -> Callable[[TensorSubclass, Tensorlike], TensorSubclass]:
    '''decorator to convert other to a Tensor like Self if it is not already'''
    @wraps(f)
    def wrapper(self: TensorSubclass, other: Tensorlike) -> TensorSubclass:
        if isinstance(other, self.__class__):
            return f(self, other)
        else:
            assert isinstance(self, RingTensor) or isinstance(self, RealTensor)
            return f(self, self.__class__(other))
    return wrapper


class Tensor(ABC):
    dtype: torch.dtype
    min_value: int | float
    max_value: int | float
    _grad: torch.Tensor | None

    def __init__(self, *, raw_data: torch.Tensor, requires_grad: bool = False):
        self.data = raw_data.to(device)
        # Disable gradient computation if in no_grad context
        self._backward = None
        self._prev = set()
        self._grad = torch.zeros_like(self.data, dtype=torch.float32) if requires_grad and not _no_grad else None

    def _new(self, raw_data: torch.Tensor, requires_grad: bool | None = None) -> Self:
        """Create a new tensor with the same class and the same requires_grad flag"""
        return self.__class__(raw_data=raw_data, requires_grad=requires_grad if requires_grad is not None else _rg(self))

    @abstractmethod
    def as_float(self) -> torch.Tensor: ...

    def reset_grad(self):
        if self._grad is not None:
            self._grad = torch.zeros_like(self.data, dtype=torch.float32)

    def set_to(self, other: "Tensor"):
        self.data = other.data.clone()
        self._grad = other._grad.clone() if _rg(other) else None
        self._backward : Callable[[], None] | None = None
        self._prev = set()
        return self

    @property
    def sign(self) -> torch.Tensor:
        return torch.sign(self.data)

    def __getitem__(self, index: int | slice | tuple[int | slice, ...]) -> Self:
        out = self._new(self.data[index])
        if not _no_grad:
            out._prev = {self}
            def _backward():
                if _rg(self) and _rg(out):
                    self._grad[index] += out._grad
            out._backward = _backward
        return out

    @convert_other
    def __add__(self, other: Self) -> Self:
        out = self._new(self.data + other.data, requires_grad=_rg(self) or _rg(other))

        if not _no_grad:
            out._prev = {self, other}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += Tensor._unbroadcast_gradient(out._grad, self.shape)
                if _rg(other) and _rg(out):
                    other._grad += Tensor._unbroadcast_gradient(out._grad, other.shape)
            out._backward = _backward
        return out

    def __radd__(self, other: Tensorlike) -> Self:
        return self + other

    def sum(self, axis=None, keepdims=False):
        # Handle None axis case - torch uses no dim argument for sum all
        if axis is None:
            result = self.data.sum()
        else:
            result = self.data.sum(dim=axis, keepdim=keepdims)
        out = self._new(result.to(self.dtype))

        if not _no_grad:
            out._prev = {self}

            # Store the original shape for backward
            original_shape = self.shape
            original_ndim = len(original_shape)

            def _backward():
                if _rg(self) and _rg(out):
                    g = out._grad
                    # restore the lost dimensions so broadcasting works
                    if not keepdims and axis is not None:
                        # Handle both single axis and tuple of axes
                        axes = axis if isinstance(axis, tuple) else (axis,)
                        # Normalize negative indices to positive
                        axes = tuple(ax % original_ndim for ax in axes)
                        # Sort axes to insert in correct order
                        for ax in sorted(axes):
                            g = g.unsqueeze(ax)
                    self._grad += g.expand(self.shape)

            out._backward = _backward
        return out

    @convert_other
    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    @convert_other
    def __rsub__(self, other: Self) -> Self:
        return other + (-self)

    def __neg__(self) -> Self:
        out = self._new(-self.data)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += -out._grad
            out._backward = _backward
        return out

    def backward(self):
        if _no_grad: return
        if self._grad is None: raise ValueError("Cannot call backward on a tensor with no gradient")
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)
        self._grad = torch.ones_like(self.data, dtype=torch.float32)
        for node in reversed(topo):
            if node._backward:
                node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self._grad})"

    @classmethod
    def full(cls, shape, value, requires_grad=False):
        return cls(raw_data=torch.full(shape, value, dtype=cls.dtype, device=device), requires_grad=requires_grad)

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.numel()

    def reshape(self, shape):
        out = self._new(self.data.reshape(shape))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad.reshape(self.data.shape)
            out._backward = _backward
        return out

    def flatten(self, *axes: int):
        new_shape = []
        remaining = 1
        for i, size in enumerate(self.shape):
            if i not in axes:
                new_shape.append(size * remaining)
                remaining = 1
            else:
                remaining *= size
        if remaining != 1:
            raise ValueError(f"Cannot flatten tensor with shape {self.shape} along last dimension")
        return self.reshape(new_shape)

    def unsqueeze(self, axis):
        out = self._new(self.data.unsqueeze(axis))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad.squeeze(axis)
            out._backward = _backward
        return out

    def squeeze(self, axis):
        out = self._new(self.data.squeeze(axis))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad.unsqueeze(axis)
            out._backward = _backward
        return out

    def sliding_window_2d(self, window_size: int, padding: int = 0, stride: int = 1) -> Self:
        # input shape: (batch, height, width, channels) -> output shape: (batch, new_height, new_width, channels, window_size, window_size)
        data = torch.nn.functional.pad(self.data, (0, 0, padding, padding, padding, padding, 0, 0), mode='constant', value=0)
        # Use unfold to create sliding windows
        # unfold works on the last dimensions, so we need to permute
        # NHWC -> NCHW for unfold, then back
        data = data.permute(0, 3, 1, 2)  # NHWC -> NCHW
        data = data.unfold(2, window_size, stride).unfold(3, window_size, stride)  # N, C, H', W', ws, ws
        data = data.permute(0, 2, 3, 1, 4, 5).contiguous()  # N, H', W', C, ws, ws
        out = self._new(data)

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    # shape: (batch, out_h, out_w, channels, window_size, window_size)
                    B, H, W, C, _, _ = out._grad.shape

                    # Create gradient tensor for padded input
                    padded_h = self.shape[1] + 2 * padding
                    padded_w = self.shape[2] + 2 * padding
                    grad_pad = torch.zeros((B, padded_h, padded_w, C), dtype=torch.float32, device=device)

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
        out = cls(raw_data=torch.stack([t.data for t in tensors], dim=axis), requires_grad=any(_rg(t) for t in tensors))

        if not _no_grad:
            out._prev = set(tensors)

            def _backward():
                for i, t in enumerate(tensors):
                    if _rg(t) and _rg(out):
                        # We need to slice the gradient corresponding to this tensor
                        slices: list[slice|int] = [slice(None)] * out._grad.ndim
                        slices[axis] = i
                        t._grad += Tensor._unbroadcast_gradient(out._grad[tuple(slices)], t.shape)
            out._backward = _backward
        return out

    @staticmethod
    def _unbroadcast_gradient(grad_output, original_input_shape):
        """
        Sums grad_output along axes to revert broadcasting, so it matches original_input_shape.
        """
        processed_grad = grad_output
        shape_out = list(processed_grad.shape)
        shape_in = list(original_input_shape)

        # 1. Sum over leading axes in grad_output not present in original_input_shape
        delta_dims = len(shape_out) - len(shape_in)
        if delta_dims > 0:
            processed_grad = torch.sum(processed_grad, dim=tuple(range(delta_dims)))
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
            processed_grad = torch.sum(processed_grad, dim=tuple(axes_to_sum_expansion), keepdim=True)

        return processed_grad.reshape(original_input_shape)


class RingTensor(Tensor):
    # dtype = torch.int8
    dtype = torch.int16
    # dtype = torch.int32
    min_value: int = torch.iinfo(dtype).min
    max_value: int = torch.iinfo(dtype).max
    # [min, max] corresponds to [-1, 1], with -1 and 1 being next to each other the number ring
    # the implementation assumes that -min_value is roughly equal to max_value

    def __init__(self, data=None, *, raw_data=None, requires_grad: bool = False):
        if (data is not None) == (raw_data is not None):
            raise ValueError("Exactly one of data or raw_data must be provided")
        if data is not None:
            if isinstance(data, RingTensor):
                raw_data = data.data.clone()
            else:
                data_tensor = data.data.clone() if isinstance(data, RealTensor) else torch.as_tensor(data, dtype=torch.float32, device=device)
                # convert data from [-1, 1] to [min, max]
                raw_data = (data_tensor * -self.min_value).clamp(self.min_value, self.max_value).to(self.dtype)
        assert raw_data is not None
        super().__init__(raw_data=raw_data, requires_grad=requires_grad)

    def as_float(self) -> torch.Tensor:
        return self.data.to(torch.float32) / -self.min_value

    def sin(self) -> Self:
        out = self._new(torch.sin(self.as_float()*pi))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad * pi * torch.cos(self.as_float()*pi)
            out._backward = _backward
        return out

    def cos(self) -> Self:
        out = self._new(torch.cos(self.as_float()*pi))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad * -pi * torch.sin(self.as_float()*pi)
            out._backward = _backward
        return out

    def complex_mean(self, axis: int) -> Self:
        axis = axis % self.data.ndim  # Normalize negative axis to positive
        theta = self.as_float() * pi
        dir_x = torch.cos(theta).sum(dim=axis)
        dir_y = torch.sin(theta).sum(dim=axis)
        mean_real = torch.atan2(dir_y, dir_x) / pi
        out = self.__class__(data=mean_real.to(torch.float32), requires_grad=_rg(self))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):

                    cos_vals = torch.cos(theta)
                    sin_vals = torch.sin(theta)

                    dir_x_expanded = dir_x.unsqueeze(axis)
                    dir_y_expanded = dir_y.unsqueeze(axis)

                    denom = dir_x_expanded * dir_x_expanded + dir_y_expanded * dir_y_expanded + 1e-20
                    local_grad = (dir_y_expanded * sin_vals + dir_x_expanded * cos_vals) / denom

                    grad_factor = out._grad.unsqueeze(axis)
                    self._grad += grad_factor * local_grad

            out._backward = _backward
        return out

    @classmethod
    def rand(cls, shape: tuple[int, ...], requires_grad: bool = False) -> Self:
        return cls(raw_data=torch.randint(cls.min_value, cls.max_value, size=shape, dtype=cls.dtype, device=device), requires_grad=requires_grad)

    def real(self) -> "RealTensor":
        out = RealTensor(self.as_float(), requires_grad=_rg(self))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad
            out._backward = _backward
        return out


class RealTensor(Tensor):
    dtype = torch.float32
    min_value: float = -float('inf')
    max_value: float = float('inf')

    def __init__(self, data=None, *, raw_data: torch.Tensor | None = None, requires_grad: bool = False):
        if (data is not None) == (raw_data is not None):
            raise ValueError("Only one of data or raw_data can be provided")
        if data is not None:
            raw_data = torch.as_tensor(data, dtype=self.dtype, device=device)
        assert raw_data is not None
        super().__init__(raw_data=raw_data, requires_grad=requires_grad)

    def as_float(self) -> torch.Tensor:
        return self.data

    @convert_other
    def __mul__(self, other: Self) -> Self:
        out = self._new(self.data * other.data, requires_grad=_rg(self) or _rg(other))

        if not _no_grad:
            out._prev = {self, other}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad * other.data
                if _rg(other) and _rg(out):
                    other._grad += out._grad * self.data
            out._backward = _backward
        return out

    def __rmul__(self, other: Tensorlike) -> Self:
        return self * other


    @convert_other
    def __truediv__(self, other: Self) -> Self:
        return self * other ** -1

    @convert_other
    def __rtruediv__(self, other: Self) -> Self:
        return other * self ** -1

    @convert_other
    def __pow__(self, other: Self) -> Self:
        out = self._new(self.data ** other.data, requires_grad=_rg(self) or _rg(other))

        if not _no_grad:
            out._prev = {self, other}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad * other.data * self.data ** (other.data - 1)
                if _rg(other) and _rg(out):
                    other._grad += out._grad * torch.log(self.data) * self.data ** other.data
            out._backward = _backward
        return out

    @convert_other
    def __rpow__(self, other: Self) -> Self:
        return other ** self

    def abs(self) -> Self:
        out = self._new(torch.abs(self.data))

        if not _no_grad:
            out._prev = {self}

            def _backward():
                if _rg(self) and _rg(out):
                    self._grad += out._grad * self.sign
            out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        # Handle None axis case - torch uses no dim argument for mean all
        # For integer dtypes, convert to float32 first
        data_for_mean = self.data.to(torch.float32) if not torch.is_floating_point(self.data) else self.data
        if axis is None:
            result = data_for_mean.mean()
        else:
            result = data_for_mean.mean(dim=axis, keepdim=keepdims)
        out = self._new(result.to(self.dtype))

        if not _no_grad:
            out._prev = {self}

            # Store the original shape for backward
            original_shape = self.shape
            original_ndim = len(original_shape)

            def _backward():
                if _rg(self) and _rg(out):
                    g = out._grad / (self.size / out.size)  # scale by 1/N
                    if not keepdims and axis is not None:
                        # Handle both single axis and tuple of axes
                        axes = axis if isinstance(axis, tuple) else (axis,)
                        # Normalize negative indices to positive
                        axes = tuple(ax % original_ndim for ax in axes)
                        # Sort axes to insert in correct order
                        for ax in sorted(axes):
                            g = g.unsqueeze(ax)
                    self._grad += g.expand(self.shape)

            out._backward = _backward
        return out

    @convert_other
    def cross_entropy(self, other: Self) -> Self:
        logits = self.as_float().reshape(self.shape[0], -1)
        tgt = other.as_float().reshape(other.shape[0], -1)
        n = logits.shape[0]

        # numerically-stable soft-max
        shift = logits - logits.max(dim=1, keepdim=True)[0]
        exps  = torch.exp(shift)
        probs = exps / exps.sum(dim=1, keepdim=True)

        # cross-entropy loss
        loss_val = -torch.sum(tgt * torch.log(probs + 1e-20)) / n
        out = self._new(loss_val.reshape(()), requires_grad=_rg(self) or _rg(other))

        if not _no_grad:
            out._prev = {self, other}

            def _backward():
                if _rg(self) and _rg(out):
                    grad = out._grad * (probs - tgt) / n
                    if self.shape != grad.shape:
                        grad = grad.reshape(self.shape)
                    self._grad += grad

                if _rg(other) and _rg(out):
                    grad = out._grad * (-torch.log(probs + 1e-20) / n)
                    if other.shape != grad.shape:
                        grad = grad.reshape(other.shape)
                    other._grad += grad

            out._backward = _backward
        return out


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

class TensorWithGrad(Tensor):
    """Tensor with a gradient. Needed for type checking."""
    _grad: torch.Tensor

def _rg(tensor: "Tensor") -> TypeGuard[TensorWithGrad]:
    """Check if the tensor has a gradient and let the type checker know."""
    return tensor._grad is not None
