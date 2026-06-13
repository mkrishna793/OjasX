// ═══════════════════════════════════════════════════════════════════
// TorchCL — Activation Function Kernels
// ReLU, Sigmoid, Tanh, GELU, LeakyReLU, SiLU, Softmax helper
// ═══════════════════════════════════════════════════════════════════

__kernel void relu_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = fmax(a[gid], 0.0f);
}

__kernel void relu_backward_f32(__global const float* grad_out,
                                __global const float* input,
                                __global float* grad_in,
                                const int n) {
    int gid = get_global_id(0);
    if (gid < n) grad_in[gid] = (input[gid] > 0.0f) ? grad_out[gid] : 0.0f;
}

__kernel void sigmoid_f32(__global const float* a,
                          __global float* c,
                          const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = 1.0f / (1.0f + exp(-a[gid]));
}

__kernel void sigmoid_backward_f32(__global const float* grad_out,
                                   __global const float* output,
                                   __global float* grad_in,
                                   const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float s = output[gid];
        grad_in[gid] = grad_out[gid] * s * (1.0f - s);
    }
}

__kernel void tanh_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = tanh(a[gid]);
}

__kernel void tanh_backward_f32(__global const float* grad_out,
                                __global const float* output,
                                __global float* grad_in,
                                const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float t = output[gid];
        grad_in[gid] = grad_out[gid] * (1.0f - t * t);
    }
}

__kernel void gelu_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float x = a[gid];
        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        c[gid] = x * cdf;
    }
}

__kernel void leaky_relu_f32(__global const float* a,
                             const float neg_slope,
                             __global float* c,
                             const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = (a[gid] >= 0.0f) ? a[gid] : neg_slope * a[gid];
}

__kernel void silu_f32(__global const float* a,
                       __global float* c,
                       const int n) {
    int gid = get_global_id(0);
    if (gid < n) c[gid] = a[gid] / (1.0f + exp(-a[gid]));
}

// ── Backward kernels for GELU, SiLU, LeakyReLU ─────────────────────

__kernel void gelu_backward_f32(__global const float* grad_out,
                                __global const float* input,
                                __global float* grad_in,
                                const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float x = input[gid];
        // GELU'(x) using tanh approximation derivative
        float k = 0.7978845608f;  // sqrt(2/pi)
        float c1 = 0.044715f;
        float inner = k * (x + c1 * x * x * x);
        float t = tanh(inner);
        float cdf = 0.5f * (1.0f + t);
        float sech2 = 1.0f - t * t;
        float dinput = 0.5f * (1.0f + t) + 0.5f * x * sech2 * k * (1.0f + 3.0f * c1 * x * x);
        grad_in[gid] = grad_out[gid] * dinput;
    }
}

__kernel void silu_backward_f32(__global const float* grad_out,
                                __global const float* input,
                                __global float* grad_in,
                                const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        float x = input[gid];
        float sig = 1.0f / (1.0f + exp(-x));
        // SiLU'(x) = sig(x) + x * sig(x) * (1 - sig(x))
        //          = sig(x) * (1 + x * (1 - sig(x)))
        float dsilu = sig * (1.0f + x * (1.0f - sig));
        grad_in[gid] = grad_out[gid] * dsilu;
    }
}

__kernel void leaky_relu_backward_f32(__global const float* grad_out,
                                      __global const float* input,
                                      const float neg_slope,
                                      __global float* grad_in,
                                      const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        grad_in[gid] = (input[gid] >= 0.0f) ? grad_out[gid] : neg_slope * grad_out[gid];
    }
}
