// ═══════════════════════════════════════════════════════════════════
// OjasX Liquid — Stateful Elementwise Kernels (CKT)
// Each kernel reads persistent state S(t), integrates dS/dt, writes S(t+dt)
// ═══════════════════════════════════════════════════════════════════

// ── Continuous ReLU: dS/dt = (-S + max(0,X)) / tau ──────────────────
__kernel void liquid_relu_f32(
    __global const float* input,
    __global float* state,
    __global float* output,
    __global float* error_out,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float x = input[gid];
    float s = state[gid];
    float target = fmax(x, 0.0f);

    // RK2 (Heun's method)
    float f1 = (-s + target) / tau;
    float s_euler = s + dt * f1;
    float f2 = (-s_euler + target) / tau;
    float s_new = s + 0.5f * dt * (f1 + f2);

    // Error estimate: |RK2 - Euler|
    float err = fabs(s_new - s_euler);

    state[gid] = s_new;
    output[gid] = s_new;

    // Per-workgroup max error (thread 0 writes)
    // Simplified: each thread writes its own error
    error_out[gid] = err;
}

// ── Continuous Sigmoid ──────────────────────────────────────────────
__kernel void liquid_sigmoid_f32(
    __global const float* input,
    __global float* state,
    __global float* output,
    __global float* error_out,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float x = input[gid];
    float s = state[gid];
    float target = 1.0f / (1.0f + exp(-x));

    float f1 = (-s + target) / tau;
    float s_euler = s + dt * f1;
    float f2 = (-s_euler + target) / tau;
    float s_new = s + 0.5f * dt * (f1 + f2);

    state[gid] = s_new;
    output[gid] = s_new;
    error_out[gid] = fabs(s_new - s_euler);
}

// ── Continuous Tanh ─────────────────────────────────────────────────
__kernel void liquid_tanh_f32(
    __global const float* input,
    __global float* state,
    __global float* output,
    __global float* error_out,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float x = input[gid];
    float s = state[gid];
    float target = tanh(x);

    float f1 = (-s + target) / tau;
    float s_euler = s + dt * f1;
    float f2 = (-s_euler + target) / tau;
    float s_new = s + 0.5f * dt * (f1 + f2);

    state[gid] = s_new;
    output[gid] = s_new;
    error_out[gid] = fabs(s_new - s_euler);
}

// ── Continuous GELU ─────────────────────────────────────────────────
__kernel void liquid_gelu_f32(
    __global const float* input,
    __global float* state,
    __global float* output,
    __global float* error_out,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float x = input[gid];
    float s = state[gid];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    float target = x * cdf;

    float f1 = (-s + target) / tau;
    float s_euler = s + dt * f1;
    float f2 = (-s_euler + target) / tau;
    float s_new = s + 0.5f * dt * (f1 + f2);

    state[gid] = s_new;
    output[gid] = s_new;
    error_out[gid] = fabs(s_new - s_euler);
}

// ── Generic continuous add (accumulator) ────────────────────────────
__kernel void liquid_add_f32(
    __global const float* a,
    __global const float* b,
    __global float* state,
    __global float* output,
    __global float* error_out,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;

    float s = state[gid];
    float target = a[gid] + b[gid];

    float f1 = (-s + target) / tau;
    float s_euler = s + dt * f1;
    float f2 = (-s_euler + target) / tau;
    float s_new = s + 0.5f * dt * (f1 + f2);

    state[gid] = s_new;
    output[gid] = s_new;
    error_out[gid] = fabs(s_new - s_euler);
}
