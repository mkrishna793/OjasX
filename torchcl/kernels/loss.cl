// ═══════════════════════════════════════════════════════════════════
// OjasX — Loss Function OpenCL Kernels
// Cross-entropy (numerically stable), MSE
// ═══════════════════════════════════════════════════════════════════

// ── Cross-entropy loss ──────────────────────────────────────────────
// Computes: loss = -sum(target * log_softmax(logits)) / batch
// Uses the numerically stable log-softmax: log_softmax = x - max - log(sum(exp(x-max)))
//
// logits:  [batch, num_classes]
// targets: [batch]  (class indices as float — truncated to int)
// loss:    [1]  (scalar output, accumulated across batch)
// Each work-item handles one sample.
__kernel void cross_entropy_forward_f32(
    __global const float* logits,    // [batch, C]
    __global const float* targets,   // [batch] — class indices (as float)
    __global float* loss_per_sample, // [batch] — per-sample loss
    __global float* log_softmax_out, // [batch, C] — saved for backward
    const int batch_size,
    const int C                      // num_classes
) {
    int n = get_global_id(0);
    if (n >= batch_size) return;

    int offset = n * C;
    int target_class = (int)targets[n];

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int j = 0; j < C; j++) {
        max_val = fmax(max_val, logits[offset + j]);
    }

    // Compute log(sum(exp(x - max)))
    float sum_exp = 0.0f;
    for (int j = 0; j < C; j++) {
        sum_exp += exp(logits[offset + j] - max_val);
    }
    float log_sum_exp = log(sum_exp);

    // Compute log_softmax for all classes (saved for backward)
    for (int j = 0; j < C; j++) {
        log_softmax_out[offset + j] = logits[offset + j] - max_val - log_sum_exp;
    }

    // Loss for this sample: -log_softmax[target_class]
    if (target_class >= 0 && target_class < C) {
        loss_per_sample[n] = -(logits[offset + target_class] - max_val - log_sum_exp);
    } else {
        loss_per_sample[n] = 0.0f;
    }
}

// ── Cross-entropy backward ──────────────────────────────────────────
// grad_logits = softmax(logits) - one_hot(target)
// Each work-item handles one sample.
__kernel void cross_entropy_backward_f32(
    __global const float* log_softmax,  // [batch, C]
    __global const float* targets,      // [batch]
    __global float* grad_logits,        // [batch, C]
    const int batch_size,
    const int C,
    const float inv_batch               // 1.0 / batch_size for mean reduction
) {
    int n = get_global_id(0);
    if (n >= batch_size) return;

    int offset = n * C;
    int target_class = (int)targets[n];

    for (int j = 0; j < C; j++) {
        float softmax_val = exp(log_softmax[offset + j]);
        float one_hot = (j == target_class) ? 1.0f : 0.0f;
        grad_logits[offset + j] = (softmax_val - one_hot) * inv_batch;
    }
}

// ── MSE loss forward ────────────────────────────────────────────────
// loss_per_element = (pred - target)^2
// Each work-item handles one element.
__kernel void mse_forward_f32(
    __global const float* pred,
    __global const float* target,
    __global float* loss_per_element,   // (pred - target)^2
    const int n
) {
    int gid = get_global_id(0);
    if (gid < n) {
        float diff = pred[gid] - target[gid];
        loss_per_element[gid] = diff * diff;
    }
}

// ── MSE backward ────────────────────────────────────────────────────
// grad_pred = 2 * (pred - target) / n
__kernel void mse_backward_f32(
    __global const float* pred,
    __global const float* target,
    __global float* grad_pred,
    const int n
) {
    int gid = get_global_id(0);
    if (gid < n) {
        grad_pred[gid] = 2.0f * (pred[gid] - target[gid]) / (float)n;
    }
}
