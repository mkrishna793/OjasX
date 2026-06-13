// ═══════════════════════════════════════════════════════════════════
// OjasX — Embedding OpenCL Kernels
// Gather (forward) and scatter-add (backward) for vocabulary lookups.
// ═══════════════════════════════════════════════════════════════════

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// ── Embedding lookup (forward) ──────────────────────────────────────
__kernel void embedding_lookup_f32(
    __global const float* weight,   // [V, D]  embedding table
    __global const float* indices,  // [N]     token indices (as float → int)
    __global float* output,         // [N, D]
    const int N,                    // number of indices
    const int D                     // embedding dimension
) {
    int gid = get_global_id(0);
    int total = N * D;
    if (gid >= total) return;

    int n = gid / D;
    int d = gid % D;
    int idx = (int)indices[n];

    output[n * D + d] = weight[idx * D + d];
}

// Inline atomic float addition using compare-and-swap CAS loop
inline void atomic_add_float(volatile __global float* addr, float val) {
    union { unsigned int u32; float f32; } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atom_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

// ── Embedding backward (scatter-add) ────────────────────────────────
// Accumulate gradients back into the embedding table.
// Each work-item handles one (n, d) position.
// Uses atomic additions to prevent data races on duplicate indices.
__kernel void embedding_backward_f32(
    __global const float* grad_out,   // [N, D]
    __global const float* indices,    // [N]
    __global float* grad_weight,      // [V, D] — accumulated
    const int N,
    const int D
) {
    int gid = get_global_id(0);
    int total = N * D;
    if (gid >= total) return;

    int n = gid / D;
    int d = gid % D;
    int idx = (int)indices[n];

    // Atomically accumulate to avoid data races
    atomic_add_float(&grad_weight[idx * D + d], grad_out[n * D + d]);
}
