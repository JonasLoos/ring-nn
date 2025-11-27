#ifndef RING_CONV2D_KERNEL2_CUH
#define RING_CONV2D_KERNEL2_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>


__global__ void ring_conv2d_forward_kernel2(
    const float* __restrict__ x,       // (B, C, H, W)
    const float* __restrict__ probe,   // (1, C, O, 1, 1, K, K)
    const float* __restrict__ out,     // (1, C, O, 1, 1, K, K)
    float* __restrict__ output,        // (B, O, H_out, W_out)
    int B, int C, int H, int W,
    int O, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * O * H_out * W_out;
    if (idx >= total) return;

    int w_out_idx = idx % W_out;
    int tmp = idx / W_out;
    int h_out_idx = tmp % H_out;
    tmp /= H_out;
    int o = tmp % O;
    int b = tmp / O;

    int base_iy = h_out_idx * stride - padding;
    int base_ix = w_out_idx * stride - padding;

    float out_x = 0.0f;
    float out_y = 0.0f;

    // Sum weighted activations over C x K x K
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;

                float xv = 0.0f;  // Default to 0 for padding
                if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
                    int x_index = ((b * C + c) * H + iy) * W + ix;
                    xv = x[x_index];
                }

                // probe and out indices: (1, C, O, 1, 1, K, K) -> simplified: (c * O + o) * K * K + ky * K + kx
                int probe_index = (c * O + o) * K * K + ky * K + kx;
                int out_index = (c * O + o) * K * K + ky * K + kx;

                float probe_val = probe[probe_index];
                float out_val = out[out_index];

                float diff = xv - probe_val;
                float cos_diff = cosf(diff);
                float activation = cos_diff * cos_diff * cos_diff;  // cos^3(diff)

                float cos_out = cosf(out_val);
                float sin_out = sinf(out_val);

                out_x += cos_out * activation;
                out_y += sin_out * activation;
            }
        }
    }

    output[idx] = atan2f(out_y, out_x);
}

__global__ void ring_conv2d_backward_kernel2(
    const float* __restrict__ x,          // (B, C, H, W)
    const float* __restrict__ probe,      // (1, C, O, 1, 1, K, K)
    const float* __restrict__ out,        // (1, C, O, 1, 1, K, K)
    const float* __restrict__ grad_out,   // (B, O, H_out, W_out)
    float* __restrict__ grad_x,           // (B, C, H, W)
    float* __restrict__ grad_probe,       // (1, C, O, 1, 1, K, K)
    float* __restrict__ grad_out_param,   // (1, C, O, 1, 1, K, K)
    int B, int C, int H, int W,
    int O, int K,
    int stride, int padding,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * O * H_out * W_out;
    if (idx >= total) return;

    int w_out_idx = idx % W_out;
    int tmp = idx / W_out;
    int h_out_idx = tmp % H_out;
    tmp /= H_out;
    int o = tmp % O;
    int b = tmp / O;

    int base_iy = h_out_idx * stride - padding;
    int base_ix = w_out_idx * stride - padding;

    const float eps = 1e-8f;

    // First pass: compute out_x and out_y
    float out_x = 0.0f;
    float out_y = 0.0f;

    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;
            if ((unsigned)iy >= (unsigned)H) continue;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;
                if ((unsigned)ix >= (unsigned)W) continue;

                int x_index = ((b * C + c) * H + iy) * W + ix;
                int probe_index = (c * O + o) * K * K + ky * K + kx;
                int out_index = (c * O + o) * K * K + ky * K + kx;

                float xv = x[x_index];
                float probe_val = probe[probe_index];
                float out_val = out[out_index];

                float diff = xv - probe_val;
                float cos_diff = cosf(diff);
                float activation = cos_diff * cos_diff * cos_diff;

                float cos_out = cosf(out_val);
                float sin_out = sinf(out_val);

                out_x += cos_out * activation;
                out_y += sin_out * activation;
            }
        }
    }

    // Gradient of atan2(y, x) w.r.t. x and y
    float go = grad_out[idx];
    float r2 = out_x * out_x + out_y * out_y + eps;
    float d_atan2_d_x = -out_y / r2;
    float d_atan2_d_y = out_x / r2;

    // Second pass: accumulate grads via chain rule
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;
            if ((unsigned)iy >= (unsigned)H) continue;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;
                if ((unsigned)ix >= (unsigned)W) continue;

                int x_index = ((b * C + c) * H + iy) * W + ix;
                int probe_index = (c * O + o) * K * K + ky * K + kx;
                int out_index = (c * O + o) * K * K + ky * K + kx;

                float xv = x[x_index];
                float probe_val = probe[probe_index];
                float out_val = out[out_index];

                float diff = xv - probe_val;
                float cos_diff = cosf(diff);
                float sin_diff = sinf(diff);
                float cos2_diff = cos_diff * cos_diff;
                float activation = cos2_diff * cos_diff;  // cos^3(diff)

                float cos_out = cosf(out_val);
                float sin_out = sinf(out_val);

                // Gradient w.r.t. activation: dL/dactivation
                // dL/dout_x = go * d_atan2_d_x, dL/dout_y = go * d_atan2_d_y
                // dL/dactivation = dL/dout_x * cos_out + dL/dout_y * sin_out
                float grad_activation = go * (d_atan2_d_x * cos_out + d_atan2_d_y * sin_out);

                // Gradient w.r.t. diff: dL/ddiff = dL/dactivation * dactivation/ddiff
                // dactivation/ddiff = d(cos^3(diff))/ddiff = -3 * cos^2(diff) * sin(diff)
                float grad_diff = grad_activation * (-3.0f * cos2_diff * sin_diff);

                // diff = x - probe, so:
                // dL/dx = grad_diff
                // dL/dprobe = -grad_diff
                atomicAdd(&grad_x[x_index], grad_diff);
                atomicAdd(&grad_probe[probe_index], -grad_diff);

                // Gradient w.r.t. out parameter:
                // dL/dout = dL/dout_x * (-sin(out) * activation) + dL/dout_y * (cos(out) * activation)
                float grad_out_val = go * (d_atan2_d_x * (-sin_out * activation) + d_atan2_d_y * (cos_out * activation));
                atomicAdd(&grad_out_param[out_index], grad_out_val);
            }
        }
    }
}

#endif // RING_CONV2D_KERNEL2_CUH

