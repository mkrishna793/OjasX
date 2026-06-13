// ═══════════════════════════════════════════════════════════════════
// OjasX Liquid — Adaptive Precision Streaming (APS) Kernels
// ═══════════════════════════════════════════════════════════════════

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void pack_fp32_to_fp16(
    __global const float* input,
    __global half* output,
    const int n
) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] = (half)input[gid];
    }
}

__kernel void unpack_fp16_to_fp32(
    __global const half* input,
    __global float* output,
    const int n
) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] = (float)input[gid];
    }
}
