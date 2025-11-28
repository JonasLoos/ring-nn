from math import pi

from torch.nn import functional as F, Module, Parameter
from torch.autograd import Function
import torch

# compile the CUDA kernel
import os
try:
    from torch.utils.cpp_extension import load
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ring_conv2d_simple_cuda = load(
        name="ring_conv2d_simple_cuda",
        sources=[
            os.path.join(current_dir, "ring_conv2d_simple_cuda.cu"),
        ],
        extra_include_paths=[current_dir],  # Include path for the .cuh file
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Warning: CUDA extension not available: {e}")
    CUDA_AVAILABLE = False
    ring_conv2d_simple_cuda = None


def complex_mean(x: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    """Compute the mean angle, by summing the complex unit numbers and taking the resulting complex number's angle."""
    dir_x = torch.cos(x).sum(dim=dim)
    dir_y = torch.sin(x).sum(dim=dim)
    return torch.atan2(dir_y, dir_x)


class RingFF(Module):
    '''Simple fully-connected feed-forward ring neural network layer.'''
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = Parameter(torch.empty(input_size, output_size).uniform_(-pi, pi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dir_x = (torch.cos(x).unsqueeze(-1) * self.weight).sum(dim=(-2,))
        dir_y = (torch.sin(x).unsqueeze(-1) * self.weight).sum(dim=(-2,))
        return torch.atan2(dir_y, dir_x)


class RingConv2dSimpleFn(Function):
    @staticmethod
    def forward(ctx, x, weight, stride, padding):
        # Apply padding first (matching Python implementation)
        if padding > 0:
            x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        
        # Use CUDA if available and tensor is on CUDA
        if CUDA_AVAILABLE and x.is_cuda:
            out = ring_conv2d_simple_cuda.forward(x, weight, stride, padding)
            ctx.save_for_backward(x, weight)  # Save padded x for CUDA backward
            ctx.stride = stride
            ctx.padding = padding
            ctx.use_cuda = True
        else:
            # Fallback to Python implementation
            # Extract patches: (B, C, 1, H', W', kernel_h, kernel_w)
            x_patches = x.unfold(2, weight.size(5), stride).unfold(3, weight.size(5), stride).unsqueeze(2)
            # compute activations based on the weighted complex mean
            dir_x = (torch.cos(x_patches) * weight).sum(dim=(1, 5, 6))
            dir_y = (torch.sin(x_patches) * weight).sum(dim=(1, 5, 6))
            out = torch.atan2(dir_y, dir_x)
            ctx.save_for_backward(x, weight)
            ctx.stride = stride
            ctx.padding = padding
            ctx.use_cuda = False
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        if ctx.use_cuda:
            grad_x_padded, grad_w = ring_conv2d_simple_cuda.backward(
                grad_out.contiguous(), x, weight, stride, padding
            )
            # Remove padding from grad_x to match original input shape
            if padding > 0:
                grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
            else:
                grad_x = grad_x_padded
        else:
            # Fallback to autograd (requires manual implementation or use autograd)
            # For now, return None to use autograd
            grad_x = None
            grad_w = None
        
        return grad_x, grad_w, None, None


class RingConv2dSimple(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        self.weight = Parameter(torch.empty(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size).uniform_(-pi, pi))
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use CUDA Function if available, otherwise fallback to Python
        if CUDA_AVAILABLE and x.is_cuda:
            return RingConv2dSimpleFn.apply(x, self.weight, self.stride, self.padding)
        else:
            # Python fallback
            if self.padding > 0:
                # For (B, C, H, W) tensor, pad the last two dimensions (H, W)
                x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

            # Extract patches: (B, C, 1, H', W', kernel_h, kernel_w)
            x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride).unsqueeze(2)

            # compute activations based on the weighted complex mean
            dir_x = (torch.cos(x) * self.weight).sum(dim=(1, 5, 6))
            dir_y = (torch.sin(x) * self.weight).sum(dim=(1, 5, 6))
            return torch.atan2(dir_y, dir_x)


class RingConv2dSimpleCUDA(Module):
    """CUDA-accelerated version of RingConv2dSimple."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Use RingConv2dSimple instead.")
        self.weight = Parameter(torch.empty(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size).uniform_(-pi, pi))
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("RingConv2dSimpleCUDA requires CUDA tensors")
        return RingConv2dSimpleFn.apply(x, self.weight, self.stride, self.padding)


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
