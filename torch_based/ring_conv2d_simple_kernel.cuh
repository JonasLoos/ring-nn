#ifndef RING_CONV2D_SIMPLE_KERNEL_CUH
#define RING_CONV2D_SIMPLE_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>


__global__ void ring_conv2d_simple_forward_kernel(
    const float* __restrict__ x,       // (B, C, H, W) - already padded
    const float* __restrict__ w,       // (1, C, O, 1, 1, K, K) flattened to (C, O, K, K)
    float* __restrict__ out,           // (B, O, H_out, W_out)
    int B, int C, int H, int W,        // H, W are dimensions after padding
    int O, int K,
    int stride, int padding,           // padding is for reference (H, W already include it)
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

    // Compute base position in padded input
    // After padding, the first valid pixel is at (padding, padding)
    int base_iy = h_out_idx * stride;
    int base_ix = w_out_idx * stride;

    float dir_x = 0.0f;
    float dir_y = 0.0f;

    // Sum cos(x) * w and sin(x) * w over C x K x K
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;

                float xv = 0.0f;  // Default to 0 for out-of-bounds (shouldn't happen if padding correct)
                if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
                    int x_index = ((b * C + c) * H + iy) * W + ix;
                    xv = x[x_index];
                }

                // Weight indexing: w is (C, O, K, K)
                int w_index = (((c * O + o) * K + ky) * K) + kx;
                float wv = w[w_index];

                float cos_x = cosf(xv);
                float sin_x = sinf(xv);

                dir_x += cos_x * wv;
                dir_y += sin_x * wv;
            }
        }
    }

    out[idx] = atan2f(dir_y, dir_x);
}

__global__ void ring_conv2d_simple_backward_kernel(
    const float* __restrict__ x,          // (B, C, H, W) - already padded
    const float* __restrict__ w,          // (1, C, O, 1, 1, K, K) flattened to (C, O, K, K)
    const float* __restrict__ grad_out,   // (B, O, H_out, W_out)
    float* __restrict__ grad_x,           // (B, C, H, W)
    float* __restrict__ grad_w,           // (C, O, K, K)
    int B, int C, int H, int W,           // H, W are dimensions after padding
    int O, int K,
    int stride, int padding,              // padding is for reference (H, W already include it)
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

    // Compute base position in padded input
    int base_iy = h_out_idx * stride;
    int base_ix = w_out_idx * stride;

    const float eps = 1e-8f;

    // First pass: compute dir_x and dir_y
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
                int w_index = (((c * O + o) * K + ky) * K) + kx;

                float xv = x[x_index];
                float wv = w[w_index];

                float cos_x = cosf(xv);
                float sin_x = sinf(xv);

                dir_x += cos_x * wv;
                dir_y += sin_x * wv;
            }
        }
    }

    float go = grad_out[idx];
    float r2 = dir_x * dir_x + dir_y * dir_y + eps;
    float d_y_d_sx = -dir_y / r2;  // d(atan2(dir_y, dir_x))/d(dir_x)
    float d_y_d_sy =  dir_x / r2;  // d(atan2(dir_y, dir_x))/d(dir_y)

    // Second pass: accumulate grads via chain rule
    // dir_x = sum(cos(x) * w), dir_y = sum(sin(x) * w)
    // d(dir_x)/dx = -sin(x) * w, d(dir_y)/dx = cos(x) * w
    // d(dir_x)/dw = cos(x), d(dir_y)/dw = sin(x)
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < K; ++ky) {
            int iy = base_iy + ky;
            if ((unsigned)iy >= (unsigned)H) continue;

            for (int kx = 0; kx < K; ++kx) {
                int ix = base_ix + kx;
                if ((unsigned)ix >= (unsigned)W) continue;

                int x_index = ((b * C + c) * H + iy) * W + ix;
                int w_index = (((c * O + o) * K + ky) * K) + kx;

                float xv = x[x_index];
                float wv = w[w_index];

                float cos_x = cosf(xv);
                float sin_x = sinf(xv);

                // dL/dx = go * (d_y_d_sx * d(dir_x)/dx + d_y_d_sy * d(dir_y)/dx)
                //       = go * (d_y_d_sx * (-sin_x * wv) + d_y_d_sy * (cos_x * wv))
                float grad_x_val = go * (d_y_d_sx * (-sin_x * wv) + d_y_d_sy * (cos_x * wv));

                // dL/dw = go * (d_y_d_sx * d(dir_x)/dw + d_y_d_sy * d(dir_y)/dw)
                //       = go * (d_y_d_sx * cos_x + d_y_d_sy * sin_x)
                float grad_w_val = go * (d_y_d_sx * cos_x + d_y_d_sy * sin_x);

                atomicAdd(&grad_x[x_index], grad_x_val);
                atomicAdd(&grad_w[w_index], grad_w_val);
            }
        }
    }
}

#endif // RING_CONV2D_SIMPLE_KERNEL_CUH

