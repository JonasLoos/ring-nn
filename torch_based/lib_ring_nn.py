from torch.nn import functional as F, Module, Parameter
from torch.autograd import Function
import torch
from math import pi


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


class _RingConv2dFused(Function):
    @staticmethod
    def forward(ctx,
                x: torch.Tensor,
                weight: torch.Tensor,
                stride: int,
                padding: int):
        """
        x: (B, C, H, W)
        weight: (O, C, K, K)
        """
        B, C, H, W = x.shape
        O, Cw, K, Kw = weight.shape
        assert Cw == C and K == Kw

        if padding > 0:
            x_pad = F.pad(x, (padding, padding, padding, padding))
        else:
            x_pad = x

        # Unfold to patches: (B, C*K*K, L) with L = H_out * W_out
        patches = F.unfold(x_pad, kernel_size=K, stride=stride)  # (B, P, L)
        B, P, L = patches.shape
        # Collapse (C, K, K) into P = C*K*K
        w_flat = weight.view(O, P)  # (O, P)

        # Output buffer (B, O, L)
        out = x.new_empty(B, O, L)

        # Compute in chunks over out_channels to cap peak memory
        chunk = 8 if O > 8 else O  # tune as needed

        for s in range(0, O, chunk):
            e = min(s + chunk, O)
            wf = w_flat[s:e]  # (chunk, P)

            # Logical operation:
            # diff[b, o, p, l] = ringify(patches[b, p, l] - wf[o, p])
            # Then complex_mean over p.
            # Implemented in a fused way (but still vectorized per chunk).

            # (B, 1, P, L) - (1, chunk, P, 1) -> (B, chunk, P, L)
            diff = patches.unsqueeze(1) - wf.view(1, e - s, P, 1)
            diff = ringify(diff)

            # complex_mean over dim=(2,) == P (the flattened C,K,K)
            # dir_x: (B, chunk, L), dir_y: (B, chunk, L)
            dir_x = torch.cos(diff).sum(dim=2)
            dir_y = torch.sin(diff).sum(dim=2)

            out[:, s:e, :] = torch.atan2(dir_y, dir_x)

            # diff, dir_x, dir_y freed here — not saved.

        # Save minimal stuff for backward
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.kernel_size = K
        ctx.P = P

        # Reshape to (B, O, H_out, W_out)
        H_out = (x_pad.shape[2] - K) // stride + 1
        W_out = (x_pad.shape[3] - K) // stride + 1
        return out.view(B, O, H_out, W_out)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        K = ctx.kernel_size
        P = ctx.P

        B, C, H, W = x.shape
        O, _, _, _ = weight.shape

        if padding > 0:
            x_pad = F.pad(x, (padding, padding, padding, padding))
        else:
            x_pad = x

        # (B, P, L)
        patches = F.unfold(x_pad, kernel_size=K, stride=stride)
        B, P, L = patches.shape
        w_flat = weight.view(O, P)

        # Flatten grad_out to (B, O, L)
        Bgo, Ogo, H_out, W_out = grad_out.shape
        assert Bgo == B and Ogo == O
        grad_out_flat = grad_out.reshape(B, O, H_out * W_out)

        grad_patches = torch.zeros_like(patches)    # (B, P, L)
        grad_w_flat = torch.zeros_like(w_flat)      # (O, P)

        eps = 1e-8
        chunk = 8 if O > 8 else O

        for s in range(0, O, chunk):
            e = min(s + chunk, O)
            wf = w_flat[s:e]                        # (chunk, P)
            go = grad_out_flat[:, s:e, :]           # (B, chunk, L)

            # Recompute diff and complex_mean ingredients for this chunk
            # diff: (B, chunk, P, L)
            diff = patches.unsqueeze(1) - wf.view(1, e - s, P, 1)
            diff = ringify(diff)

            cosd = torch.cos(diff)
            sind = torch.sin(diff)

            # dir_x, dir_y as in forward
            dir_x = cosd.sum(dim=2)        # (B, chunk, L)
            dir_y = sind.sum(dim=2)        # (B, chunk, L)

            r2 = dir_x * dir_x + dir_y * dir_y + eps
            d_y_d_sx = -dir_y / r2                  # (B, chunk, L)
            d_y_d_sy =  dir_x / r2                  # (B, chunk, L)

            # f_x(diff) = ringify(cos(diff)) ~ cos(diff) for grad (ignoring jumps)
            # df_x/d(diff) ≈ -sin(diff)
            # f_y(diff) = sin(diff), df_y/d(diff) = cos(diff)

            # grad_diff = go * (d_y_d_sx * df_x + d_y_d_sy * df_y)
            #           = go * (d_y_d_sx * (-sin(diff)) + d_y_d_sy * cos(diff))
            tmp1 = d_y_d_sx.unsqueeze(2) * (-sind)  # (B, chunk, P, L)
            tmp2 = d_y_d_sy.unsqueeze(2) * cosd     # (B, chunk, P, L)
            grad_diff = go.unsqueeze(2) * (tmp1 + tmp2)

            # Accumulate grad wrt patches: sum over out_channels in this chunk
            # grad_patches[b, p, l] += sum_o grad_diff[b, o, p, l]
            grad_patches += grad_diff.sum(dim=1)

            # Grad wrt weights: diff = x_p - w => dL/dw = -sum dL/ddiff
            # grad_w_flat[o, p] -= sum_{b,l} grad_diff[b, o, p, l]
            grad_w_flat[s:e] -= grad_diff.sum(dim=(0, 3))

            # grad_diff, tmp1, tmp2, etc. freed here

        # Convert patch grads back to image grads
        H_pad = x_pad.shape[2]
        W_pad = x_pad.shape[3]
        grad_x_pad = F.fold(grad_patches, output_size=(H_pad, W_pad),
                            kernel_size=K, stride=stride)

        if padding > 0:
            grad_x = grad_x_pad[:, :, padding:-padding, padding:-padding]
        else:
            grad_x = grad_x_pad

        grad_weight = grad_w_flat.view_as(weight)

        return grad_x, grad_weight, None, None


class RingConv2dFused(Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, padding: int = 0):
        super().__init__()
        self.weight = Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RingConv2dFused.apply(
            x, self.weight, self.stride, self.padding
        ).clone()



class RingConv2dSimple(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.weight = Parameter(torch.randn(1, in_channels, out_channels, 1, 1, kernel_size, kernel_size))
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
