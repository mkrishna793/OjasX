// ═══════════════════════════════════════════════════════════════════
// OjasX Liquid — Adaptive Workgroup Morphing (AWM) Kernels
// ═══════════════════════════════════════════════════════════════════

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel void awm_relu_f32(
    __global const float* input,
    __global float* output,
    __global float* prev_output,
    __global int* work_queue,
    __local int* local_converged,
    const float epsilon,
    const int n_base,
    const int total_work
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    if (lid == 0) {
        *local_converged = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool is_converged = false;
    bool active = (gid < n_base);
    float y = 0.0f;

    if (active) {
        float x = input[gid];
        y = x > 0.0f ? x : 0.0f;
        float prev = prev_output[gid];

        float diff = fabs(y - prev);
        float denom = fabs(prev) + 1e-7f;
        if (diff / denom < epsilon) {
            is_converged = true;
            atom_inc(local_converged);
        }

        // Divergent workload simulation:
        // If not converged (hard task), run a heavy loop.
        // If converged (easy task), skip loop to steal work early.
        if (!is_converged) {
            float val = y;
            for (int i = 0; i < 200; i++) {
                val = native_cos(val) * native_sin(val) + 0.1f;
            }
            if (val == 9999.0f) y = val; // prevent optimization
        }

        output[gid] = y;
        prev_output[gid] = y;
    } else {
        // Out of bounds threads of the rounded workgroup are counted as converged
        is_converged = true;
        atom_inc(local_converged);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // If workgroup is mostly converged (>50%), converged threads help steal and process shared work
    int converged_count = *local_converged;
    if (converged_count > (int)(0.5f * local_size)) {
        if (is_converged) {
            while (true) {
                int stolen_idx = atom_inc(work_queue);
                if (stolen_idx >= total_work) break;

                float sx = input[stolen_idx];
                float sy = sx > 0.0f ? sx : 0.0f;
                output[stolen_idx] = sy;
                prev_output[stolen_idx] = sy;
            }
        }
    }
}

__kernel void awm_sigmoid_f32(
    __global const float* input,
    __global float* output,
    __global float* prev_output,
    __global int* work_queue,
    __local int* local_converged,
    const float epsilon,
    const int n_base,
    const int total_work
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    if (lid == 0) {
        *local_converged = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool is_converged = false;
    bool active = (gid < n_base);
    float y = 0.0f;

    if (active) {
        float x = input[gid];
        y = 1.0f / (1.0f + exp(-x));
        float prev = prev_output[gid];

        float diff = fabs(y - prev);
        float denom = fabs(prev) + 1e-7f;
        if (diff / denom < epsilon) {
            is_converged = true;
            atom_inc(local_converged);
        }

        if (!is_converged) {
            float val = y;
            for (int i = 0; i < 200; i++) {
                val = native_cos(val) * native_sin(val) + 0.1f;
            }
            if (val == 9999.0f) y = val;
        }

        output[gid] = y;
        prev_output[gid] = y;
    } else {
        is_converged = true;
        atom_inc(local_converged);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int converged_count = *local_converged;
    if (converged_count > (int)(0.5f * local_size)) {
        if (is_converged) {
            while (true) {
                int stolen_idx = atom_inc(work_queue);
                if (stolen_idx >= total_work) break;

                float sx = input[stolen_idx];
                float sy = 1.0f / (1.0f + exp(-sx));
                output[stolen_idx] = sy;
                prev_output[stolen_idx] = sy;
            }
        }
    }
}
