// ═══════════════════════════════════════════════════════════════════
// OjasX — Normalization OpenCL Kernels
// LayerNorm, BatchNorm, RMSNorm (forward + backward)
// ═══════════════════════════════════════════════════════════════════

// ── LayerNorm forward ───────────────────────────────────────────────
// Input:  [M, N]  where M = batch * spatial, N = normalized_shape
// Weight: [N]     (gamma)
// Bias:   [N]     (beta)
// Output: [M, N]
// Mean:   [M]     (saved for backward)
// Rstd:   [M]     (saved for backward)  rstd = 1/sqrt(var+eps)
//
// Each work-item handles one row (one sample's normalization).
__kernel void layer_norm_f32(
    __global const float* input,
    __global const float* weight,   // gamma [N]
    __global const float* bias,     // beta  [N]
    __global float* output,
    __global float* mean_out,       // [M] — saved for backward
    __global float* rstd_out,       // [M] — saved for backward
    const int M,                    // number of rows
    const int N,                    // normalized dimension size
    const float eps
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int offset = row * N;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += input[offset + j];
    }
    float mu = sum / (float)N;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float diff = input[offset + j] - mu;
        var_sum += diff * diff;
    }
    float variance = var_sum / (float)N;
    float rstd = 1.0f / sqrt(variance + eps);

    // Save for backward
    mean_out[row] = mu;
    rstd_out[row] = rstd;

    // Pass 3: normalize, scale, shift
    for (int j = 0; j < N; j++) {
        float normed = (input[offset + j] - mu) * rstd;
        output[offset + j] = normed * weight[j] + bias[j];
    }
}

// ── LayerNorm backward ──────────────────────────────────────────────
// Computes grad_input, grad_weight, grad_bias from grad_output.
// Each work-item handles one row for grad_input.
// grad_weight and grad_bias need atomic adds (accumulated across rows).
__kernel void layer_norm_backward_f32(
    __global const float* grad_out,  // [M, N]
    __global const float* input,     // [M, N]
    __global const float* weight,    // [N]
    __global const float* mean,      // [M]
    __global const float* rstd,      // [M]
    __global float* grad_input,      // [M, N]
    const int M,
    const int N
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int offset = row * N;
    float mu = mean[row];
    float rs = rstd[row];

    // Compute ds = sum(grad_out * (x - mean) * rstd * weight)
    // Compute db = sum(grad_out * weight)
    float ds = 0.0f;
    float db = 0.0f;
    for (int j = 0; j < N; j++) {
        float g = grad_out[offset + j];
        float x_hat = (input[offset + j] - mu) * rs;
        ds += g * weight[j] * x_hat;
        db += g * weight[j];
    }

    // grad_input[i,j] = rstd * (grad_out[i,j] * weight[j]
    //                   - (x_hat * ds + db) / N)
    float inv_N = 1.0f / (float)N;
    for (int j = 0; j < N; j++) {
        float x_hat = (input[offset + j] - mu) * rs;
        float g = grad_out[offset + j];
        grad_input[offset + j] = rs * (g * weight[j] - inv_N * (x_hat * ds + db));
    }
}

// Inline atomic float addition using compare-and-swap CAS loop
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

inline void atomic_add_float(volatile __global float* addr, float val) {
    union { unsigned int u32; float f32; } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atom_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// ── Accumulate grad_weight and grad_bias (parallelized 2D kernel) ──
// Reduces across the M dimension in parallel.
// gid.0 = column j, gid.1 = row i.
__kernel void layer_norm_grad_weight_bias_f32(
    __global const float* grad_out,  // [M, N]
    __global const float* input,     // [M, N]
    __global const float* mean,      // [M]
    __global const float* rstd,      // [M]
    __global float* grad_weight,     // [N]
    __global float* grad_bias,       // [N]
    const int M,
    const int N
) {
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (j >= N || i >= M) return;

    int idx = i * N + j;
    float g = grad_out[idx];
    float x_hat = (input[idx] - mean[i]) * rstd[i];

    atomic_add_float(&grad_weight[j], g * x_hat);
    atomic_add_float(&grad_bias[j], g);
}

// ── BatchNorm forward (training mode) ───────────────────────────────
// Input:  [N, C, spatial]  (flattened spatial dims)
// Weight: [C] (gamma)
// Bias:   [C] (beta)
// Each work-item handles one channel.
__kernel void batch_norm_f32(
    __global const float* input,      // [batch, C, spatial]
    __global const float* weight,     // [C]
    __global const float* bias,       // [C]
    __global float* output,           // [batch, C, spatial]
    __global float* mean_out,         // [C]
    __global float* var_out,          // [C]
    const int batch_size,
    const int C,
    const int spatial,                // H * W
    const float eps,
    const float momentum              // for running stats (unused here — done on CPU)
) {
    int c = get_global_id(0);
    if (c >= C) return;

    int total = batch_size * spatial;

    // Compute mean for this channel
    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial; s++) {
            sum += input[n * C * spatial + c * spatial + s];
        }
    }
    float mu = sum / (float)total;

    // Compute variance for this channel
    float var_sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial; s++) {
            float diff = input[n * C * spatial + c * spatial + s] - mu;
            var_sum += diff * diff;
        }
    }
    float variance = var_sum / (float)total;

    mean_out[c] = mu;
    var_out[c] = variance;

    float rstd = 1.0f / sqrt(variance + eps);

    // Normalize, scale, shift
    for (int n = 0; n < batch_size; n++) {
        for (int s = 0; s < spatial; s++) {
            int idx = n * C * spatial + c * spatial + s;
            float normed = (input[idx] - mu) * rstd;
            output[idx] = normed * weight[c] + bias[c];
        }
    }
}

// ── RMSNorm forward ─────────────────────────────────────────────────
// Used in LLaMA-style models. No mean subtraction, just RMS scaling.
// out = x / sqrt(mean(x^2) + eps) * weight
// Each work-item handles one row.
__kernel void rms_norm_f32(
    __global const float* input,
    __global const float* weight,     // [N]
    __global float* output,
    __global float* rrms_out,         // [M] — saved for backward
    const int M,
    const int N,
    const float eps
) {
    int row = get_global_id(0);
    if (row >= M) return;

    int offset = row * N;

    // Compute mean(x^2)
    float sq_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float v = input[offset + j];
        sq_sum += v * v;
    }
    float rms = sqrt(sq_sum / (float)N + eps);
    float rrms = 1.0f / rms;

    rrms_out[row] = rrms;

    for (int j = 0; j < N; j++) {
        output[offset + j] = input[offset + j] * rrms * weight[j];
    }
}
