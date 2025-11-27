#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Include the kernel definitions
#include "ring_conv2d_kernel2.cuh"

torch::Tensor ring_conv2d_forward_cuda2(
    torch::Tensor x,
    torch::Tensor probe,
    torch::Tensor out,
    int stride,
    int padding
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(probe.is_cuda(), "probe must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(probe.scalar_type() == torch::kFloat32, "probe must be float32");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");

    x = x.contiguous();
    probe = probe.contiguous();
    out = out.contiguous();

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int O = probe.size(2);  // probe is (1, C, O, 1, 1, K, K)
    const int K = probe.size(6);

    const int H_out = (H + 2 * padding - K) / stride + 1;
    const int W_out = (W + 2 * padding - K) / stride + 1;

    auto output = torch::empty({B, O, H_out, W_out}, x.options());

    int total = B * O * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ring_conv2d_forward_kernel2<<<blocks, threads>>>(
        x.data_ptr<float>(),
        probe.data_ptr<float>(),
        out.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        O, K,
        stride, padding,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ring_conv2d_forward_kernel2 launch failed");

    return output;
}

std::vector<torch::Tensor> ring_conv2d_backward_cuda2(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor probe,
    torch::Tensor out,
    int stride,
    int padding
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(probe.is_cuda(), "probe must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");

    grad_out = grad_out.contiguous();
    x = x.contiguous();
    probe = probe.contiguous();
    out = out.contiguous();

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int O = probe.size(2);
    const int K = probe.size(6);
    const int H_out = grad_out.size(2);
    const int W_out = grad_out.size(3);

    auto grad_x = torch::zeros_like(x);
    auto grad_probe = torch::zeros_like(probe);
    auto grad_out_param = torch::zeros_like(out);

    int total = B * O * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    ring_conv2d_backward_kernel2<<<blocks, threads>>>(
        x.data_ptr<float>(),
        probe.data_ptr<float>(),
        out.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_probe.data_ptr<float>(),
        grad_out_param.data_ptr<float>(),
        B, C, H, W,
        O, K,
        stride, padding,
        H_out, W_out
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ring_conv2d_backward_kernel2 launch failed");

    return {grad_x, grad_probe, grad_out_param};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ring_conv2d_forward_cuda2, "RingConv2d forward (CUDA) v2");
    m.def("backward", &ring_conv2d_backward_cuda2, "RingConv2d backward (CUDA) v2");
}

