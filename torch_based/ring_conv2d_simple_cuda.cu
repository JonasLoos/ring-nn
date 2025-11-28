#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Include the kernel definitions
#include "ring_conv2d_simple_kernel.cuh"

torch::Tensor ring_conv2d_simple_forward_cuda(
    torch::Tensor x,
    torch::Tensor w,
    int stride,
    int padding
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "w must be float32");

    x = x.contiguous();
    w = w.contiguous();

    // x is (B, C, H, W) - already padded
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);  // Already includes padding
    const int W = x.size(3);  // Already includes padding
    
    // w is (1, C, O, 1, 1, K, K) - we'll reshape to (C, O, K, K)
    TORCH_CHECK(w.size(0) == 1, "w first dim must be 1");
    TORCH_CHECK(w.size(1) == C, "w second dim must match input channels");
    const int O = w.size(2);
    TORCH_CHECK(w.size(3) == 1 && w.size(4) == 1, "w dims 3 and 4 must be 1");
    const int K = w.size(5);
    TORCH_CHECK(w.size(6) == K, "w kernel size must be square");
    
    // Reshape weight from (1, C, O, 1, 1, K, K) to (C, O, K, K)
    auto w_flat = w.view({C, O, K, K});

    const int H_out = (H - K) / stride + 1;
    const int W_out = (W - K) / stride + 1;

    auto out = torch::empty({B, O, H_out, W_out}, x.options());

    int total = B * O * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    ring_conv2d_simple_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        O, K,
        stride, padding,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ring_conv2d_simple_forward_kernel launch failed");

    return out;
}

std::vector<torch::Tensor> ring_conv2d_simple_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor w,
    int stride,
    int padding
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA");

    grad_out = grad_out.contiguous();
    x = x.contiguous();
    w = w.contiguous();

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);  // Already includes padding
    const int W = x.size(3);  // Already includes padding
    const int O = w.size(2);
    const int K = w.size(5);
    const int H_out = grad_out.size(2);
    const int W_out = grad_out.size(3);

    // Reshape weight from (1, C, O, 1, 1, K, K) to (C, O, K, K) for kernel
    auto w_flat = w.view({C, O, K, K});

    auto grad_x = torch::zeros_like(x);
    auto grad_w_flat = torch::zeros({C, O, K, K}, w.options());

    int total = B * O * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ring_conv2d_simple_backward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w_flat.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_w_flat.data_ptr<float>(),
        B, C, H, W,
        O, K,
        stride, padding,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ring_conv2d_simple_backward_kernel launch failed");

    // Reshape grad_w back to original shape (1, C, O, 1, 1, K, K)
    auto grad_w = grad_w_flat.view({1, C, O, 1, 1, K, K});

    return {grad_x, grad_w};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ring_conv2d_simple_forward_cuda, "RingConv2dSimple forward (CUDA)");
    m.def("backward", &ring_conv2d_simple_backward_cuda, "RingConv2dSimple backward (CUDA)");
}

