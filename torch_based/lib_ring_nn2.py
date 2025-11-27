from math import pi

from torch.nn import functional as F, Module, Parameter
from torch.autograd import Function
import torch

# compile the CUDA kernel
from torch.utils.cpp_extension import load
ring_conv2d_cuda2 = load(
    name="ring_conv2d_cuda2",
    sources=[
        "ring_conv2d_cuda2.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


def complex_mean(x: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    """Compute the mean angle, by summing the complex unit numbers and taking the resulting complex number's angle."""
    dir_x = torch.cos(x).sum(dim=dim)
    dir_y = torch.sin(x).sum(dim=dim)
    return torch.atan2(dir_y, dir_x)


class RingFF(Module):
    '''Simple fully-connected feed-forward ring neural network layer.'''
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.probe = Parameter(torch.empty(input_size, output_size).uniform_(-pi, pi))
        self.out = Parameter(torch.empty(input_size, output_size).uniform_(-pi, pi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute activations based on the difference between the input and the probe
        diff = x.unsqueeze(-1) - self.probe
        activations = torch.cos(diff)**3
        # take weighted complex mean
        out_x = (torch.cos(self.out) * activations).sum(dim=(-2,))
        out_y = (torch.sin(self.out) * activations).sum(dim=(-2,))
        return torch.atan2(out_y, out_x)


class RingConv2dFn2(Function):
    @staticmethod
    def forward(ctx, x, probe, out, stride, padding):
        output = ring_conv2d_cuda2.forward(x, probe, out, stride, padding)
        ctx.save_for_backward(x, probe, out)
        ctx.stride = stride
        ctx.padding = padding
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, probe, out = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_x, grad_probe, grad_out_param = ring_conv2d_cuda2.backward(
            grad_out.contiguous(), x, probe, out, stride, padding
        )
        return grad_x, grad_probe, grad_out_param, None, None


class RingConv2dCUDA(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        self.probe = Parameter(torch.empty(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size).uniform_(-pi, pi))
        self.out = Parameter(torch.empty(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size).uniform_(-pi, pi))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RingConv2dFn2.apply(x, self.probe, self.out, self.stride, self.padding)


class RingConv2dSimple(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        self.probe = Parameter(torch.empty(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size).uniform_(-pi, pi))
        self.out = Parameter(torch.empty(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size).uniform_(-pi, pi))
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            # For (B, C, H, W) tensor, pad the last two dimensions (H, W)
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        # Extract patches: (B, C, 1, H', W', kernel_h, kernel_w)
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unsqueeze(2)

        # compute activations based on the difference between the input and the probe
        diff = x - self.probe
        activations = torch.cos(diff)**3
        # take weighted complex mean
        out_x = (torch.cos(self.out) * activations).sum(dim=(1, 5, 6))
        out_y = (torch.sin(self.out) * activations).sum(dim=(1, 5, 6))
        return torch.atan2(out_y, out_x)


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
