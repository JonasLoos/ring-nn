#ifndef RING_CONV2D_KERNEL_CUH
#define RING_CONV2D_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>


__global__ void ring_conv2d_forward_kernel(
    const float* __restrict__ x,       // (B, C, H, W)
    const float* __restrict__ w,       // (O, C, K, K)
    float* __restrict__ out,           // (B, O, H_out, W_out)
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

    float dir_x = 0.0f;
    float dir_y = 0.0f;

    // Sum complex unit vectors over C x K x K
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

                int w_index = (((o * C + c) * K + ky) * K) + kx;
                float wv = w[w_index];

                float diff = xv - wv;

                float cval = cosf(diff);
                float sval = sinf(diff);

                dir_x += cval;
                dir_y += sval;
            }
        }
    }

    out[idx] = atan2f(dir_y, dir_x);
}

__global__ void ring_conv2d_backward_kernel(
    const float* __restrict__ x,          // (B, C, H, W)
    const float* __restrict__ w,          // (O, C, K, K)
    const float* __restrict__ grad_out,   // (B, O, H_out, W_out)
    float* __restrict__ grad_x,           // (B, C, H, W)
    float* __restrict__ grad_w,           // (O, C, K, K)
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

    // First pass for this (b, o, h_out, w_out): compute diff, cos, sin, dir_x, dir_y.
    // For memory reasons, recompute in-place and immediately use it below.

    const float eps = 1e-8f;

    // You could do a two-pass loop; here we do one logical pass:
    // accumulate dir_x/dir_y, then a second loop to apply chain rule.
    // To avoid extra storage we recompute cos/sin in the second loop.

    float dir_x = 0.0f;
    float dir_y = 0.0f;

    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;
            if ((unsigned)iy >= (unsigned)H) continue;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;
                if ((unsigned)ix >= (unsigned)W) continue;

                int x_index = ((b * C + c) * H + iy) * W + ix;
                int w_index = (((o * C + c) * K + ky) * K) + kx;

                float xv = x[x_index];
                float wv = w[w_index];

                float diff = xv - wv;
                float cval = cosf(diff);
                float sval = sinf(diff);

                dir_x += cval;
                dir_y += sval;
            }
        }
    }

    float go = grad_out[idx];
    float r2 = dir_x * dir_x + dir_y * dir_y + eps;
    float d_y_d_sx = -dir_y / r2;
    float d_y_d_sy =  dir_x / r2;

    // Second pass: accumulate grads via chain rule
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;
            if ((unsigned)iy >= (unsigned)H) continue;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;
                if ((unsigned)ix >= (unsigned)W) continue;

                int x_index = ((b * C + c) * H + iy) * W + ix;
                int w_index = (((o * C + c) * K + ky) * K) + kx;

                float xv = x[x_index];
                float wv = w[w_index];

                float diff = xv - wv;
                float cval = cosf(diff);
                float sval = sinf(diff);

                // grad_diff = go * (d_y_d_sx * (-sin(diff)) + d_y_d_sy * cos(diff))
                float grad_diff = go * (d_y_d_sx * (-sval) + d_y_d_sy * cval);

                // diff = x - w
                // so dL/dx += grad_diff, dL/dw -= grad_diff
                atomicAdd(&grad_x[x_index], grad_diff);
                atomicAdd(&grad_w[w_index], -grad_diff);
            }
        }
    }
}

#endif // RING_CONV2D_KERNEL_CUH

