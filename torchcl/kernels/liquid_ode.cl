// ═══════════════════════════════════════════════════════════════════
// OjasX Liquid — ODE Integration Kernels
// Euler, RK2, RK4, Adaptive RK23 with error control
// ═══════════════════════════════════════════════════════════════════

// Generic ODE: dS/dt = F(S, X) where F = (-S + X) / tau
// These kernels work on arbitrary state vectors.

__kernel void ode_euler_f32(
    __global const float* target,
    __global float* state,
    __global float* output,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float s = state[gid];
    float f = (-s + target[gid]) / tau;
    float s_new = s + dt * f;
    state[gid] = s_new;
    output[gid] = s_new;
}

__kernel void ode_rk2_f32(
    __global const float* target,
    __global float* state,
    __global float* output,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float s = state[gid];
    float t = target[gid];
    float f1 = (-s + t) / tau;
    float s_mid = s + 0.5f * dt * f1;
    float f2 = (-s_mid + t) / tau;
    float s_new = s + dt * f2;
    state[gid] = s_new;
    output[gid] = s_new;
}

__kernel void ode_rk4_f32(
    __global const float* target,
    __global float* state,
    __global float* output,
    const float dt,
    const float tau,
    const int n
) {
    int gid = get_global_id(0);
    if (gid >= n) return;
    float s = state[gid];
    float t = target[gid];

    float k1 = (-s + t) / tau;
    float k2 = (-(s + 0.5f * dt * k1) + t) / tau;
    float k3 = (-(s + 0.5f * dt * k2) + t) / tau;
    float k4 = (-(s + dt * k3) + t) / tau;

    float s_new = s + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    state[gid] = s_new;
    output[gid] = s_new;
}

// Adaptive RK23: computes both RK2 and RK3, error = |RK3 - RK2|
__kernel void ode_adaptive_f32(
    __global const float* target,
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
    float t = target[gid];

    // RK2
    float k1 = (-s + t) / tau;
    float s_half = s + 0.5f * dt * k1;
    float k2 = (-s_half + t) / tau;
    float s_rk2 = s + dt * k2;

    // RK3 (Bogacki-Shampine style)
    float s_mid = s + 0.5f * dt * k1;
    float k2b = (-s_mid + t) / tau;
    float s_end = s + dt * k2b;
    float k3 = (-s_end + t) / tau;
    float s_rk3 = s + (dt / 6.0f) * (k1 + 4.0f * k2b + k3);

    float err = fabs(s_rk3 - s_rk2);

    // Use higher-order result
    state[gid] = s_rk3;
    output[gid] = s_rk3;
    error_out[gid] = err;
}
