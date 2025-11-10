#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Include the kernel definitions
#include "ring_conv2d_kernel.cuh"

torch::Tensor ring_conv2d_forward_cuda(
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

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int O = w.size(0);
    const int K = w.size(2);

    const int H_out = (H + 2 * padding - K) / stride + 1;
    const int W_out = (W + 2 * padding - K) / stride + 1;

    auto out = torch::empty({B, O, H_out, W_out}, x.options());

    int total = B * O * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ring_conv2d_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        O, K,
        stride, padding,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ring_conv2d_forward_kernel launch failed");

    return out;
}

std::vector<torch::Tensor> ring_conv2d_backward_cuda(
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
    const int H = x.size(2);
    const int W = x.size(3);
    const int O = w.size(0);
    const int K = w.size(2);
    const int H_out = grad_out.size(2);
    const int W_out = grad_out.size(3);

    auto grad_x = torch::zeros_like(x);
    auto grad_w = torch::zeros_like(w);

    int total = B * O * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ring_conv2d_backward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        B, C, H, W,
        O, K,
        stride, padding,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ring_conv2d_backward_kernel launch failed");

    return {grad_x, grad_w};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ring_conv2d_forward_cuda, "RingConv2d forward (CUDA)");
    m.def("backward", &ring_conv2d_backward_cuda, "RingConv2d backward (CUDA)");
}

