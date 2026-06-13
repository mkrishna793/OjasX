// ═══════════════════════════════════════════════════════════════════
// OjasX — Fused Attention (FlashAttention equivalent) OpenCL Kernel
// ═══════════════════════════════════════════════════════════════════

__kernel void flash_attention_f32(
    __global const float* Q,       // [B * H * M, D]
    __global const float* K,       // [B * H * N, D]
    __global const float* V,       // [B * H * N, D]
    __global float* Out,           // [B * H * M, D]
    const int B,                   // Batch size
    const int H,                   // Heads
    const int M,                   // Query seq len
    const int N,                   // Key/value seq len
    const int D,                   // Head dimension
    const float scale
) {
    // Each workgroup handles one row of Q (representing one query token).
    // The number of workgroups is B * H * M.
    int q_row = get_group_id(0);
    if (q_row >= B * H * M) return;

    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    // Compute batch/head indices
    int bh = q_row / M; // Batch * Head index
    int q_idx = q_row % M;

    // Query offset
    int q_offset = q_row * D;
    // Key/Value offset for this batch and head
    int kv_offset = bh * N * D;

    // Local memory buffers for parallel reduction
    // We support local workgroup sizes up to 256 threads.
    __local float local_max[256];
    __local float local_sum[256];

    // Local buffer for attention scores (scaled dot products)
    // Supports sequence lengths up to 1024.
    #define MAX_N 1024
    __local float scores[MAX_N];

    // Step 1: Compute query dot product with all keys
    for (int j = lid; j < N; j += lsize) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            sum += Q[q_offset + k] * K[kv_offset + j * D + k];
        }
        scores[j] = sum * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Softmax over scores[0...N-1] in local memory
    // Find local max
    float my_max = -1e20f;
    for (int j = lid; j < N; j += lsize) {
        if (scores[j] > my_max) {
            my_max = scores[j];
        }
    }
    local_max[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce to find global max for this row
    if (lid == 0) {
        float g_max = local_max[0];
        for (int i = 1; i < lsize; i++) {
            if (local_max[i] > g_max) {
                g_max = local_max[i];
            }
        }
        local_max[0] = g_max;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float row_max = local_max[0];

    // Compute sum of exp(score - row_max)
    float my_sum = 0.0f;
    for (int j = lid; j < N; j += lsize) {
        scores[j] = exp(scores[j] - row_max);
        my_sum += scores[j];
    }
    local_sum[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce to find global sum
    if (lid == 0) {
        float g_sum = 0.0f;
        for (int i = 0; i < lsize; i++) {
            g_sum += local_sum[i];
        }
        local_sum[0] = g_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float row_sum = local_sum[0];

    // Normalize scores to get probabilities
    for (int j = lid; j < N; j += lsize) {
        scores[j] /= (row_sum + 1e-6f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Compute weighted sum of values: Out = Prob * V
    for (int k = lid; k < D; k += lsize) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += scores[j] * V[kv_offset + j * D + k];
        }
        Out[q_offset + k] = sum;
    }
}
