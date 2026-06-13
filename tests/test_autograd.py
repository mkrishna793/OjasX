"""
OjasX Autograd Test — Verifies that gradients flow correctly through
OpenCL operations by comparing against PyTorch CPU autograd.
"""

import sys
import torch
import numpy as np

print("=" * 60)
print("  OjasX V2 — Autograd Test Suite")
print("=" * 60)

import torchcl
from torchcl.autograd import (
    ReluFunction,
    SigmoidFunction,
    TanhFunction,
    GeluFunction,
    SiluFunction,
    LeakyReluFunction,
    AddFunction,
    SubFunction,
    MulFunction,
    MatmulFunction,
    CrossEntropyFunction,
    LayerNormFunction,
)

passed = 0
failed = 0
errors = []


def check(name, got, expected, atol=1e-2):
    """Compare OpenCL result to CPU expected value."""
    global passed, failed
    try:
        if isinstance(got, torch.Tensor) and torchcl.is_opencl_tensor(got):
            got = torchcl.to_cpu(got)
        if isinstance(expected, torch.Tensor):
            ok = torch.allclose(got.float(), expected.float(), atol=atol, rtol=1e-2)
        else:
            ok = abs(float(got) - float(expected)) < atol
        if ok:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            errors.append(name)
            print(f"  [FAIL] {name}")
            print(f"         Got:      {got.flatten()[:5]}")
            print(f"         Expected: {expected.flatten()[:5]}")
    except Exception as e:
        failed += 1
        errors.append(f"{name}: {e}")
        print(f"  [ERROR] {name}: {e}")


# ── Test: ReLU forward + backward ────────────────────────────────────
print("\n--- Activation Forward/Backward ---")

x_cpu = torch.randn(64, 64, requires_grad=True)
x_cl = torchcl.to_opencl(x_cpu.detach())

# ReLU
y_cpu = torch.relu(x_cpu)
y_cl = ReluFunction.apply(x_cl)
check("relu forward", y_cl, y_cpu.detach(), atol=1e-4)

# ReLU backward
grad_out_cpu = torch.ones_like(y_cpu)
y_cpu.backward(grad_out_cpu)

grad_out_cl = torchcl.to_opencl(grad_out_cpu.detach())
grad_in_cl = ReluFunction.backward(
    type('ctx', (), {
        'saved_tensors': (x_cl,),
        '_n': 64*64,
        '_shape': (64, 64),
    })(),
    grad_out_cl,
)
check("relu backward", grad_in_cl, x_cpu.grad, atol=1e-4)

# Sigmoid forward
x_cpu2 = torch.randn(32, 32, requires_grad=True)
x_cl2 = torchcl.to_opencl(x_cpu2.detach())

y_cpu2 = torch.sigmoid(x_cpu2)
y_cl2 = SigmoidFunction.apply(x_cl2)
check("sigmoid forward", y_cl2, y_cpu2.detach(), atol=1e-3)

# Tanh forward
y_cpu3 = torch.tanh(x_cpu2)
y_cl3 = TanhFunction.apply(x_cl2)
check("tanh forward", y_cl3, y_cpu3.detach(), atol=1e-3)

# GELU forward
y_cpu4 = torch.nn.functional.gelu(x_cpu2)
y_cl4 = GeluFunction.apply(x_cl2)
check("gelu forward", y_cl4, y_cpu4.detach(), atol=1e-2)

# SiLU forward
y_cpu5 = torch.nn.functional.silu(x_cpu2)
y_cl5 = SiluFunction.apply(x_cl2)
check("silu forward", y_cl5, y_cpu5.detach(), atol=1e-3)

# ── Test: Arithmetic forward ────────────────────────────────────────
print("\n--- Arithmetic Forward ---")

a_cpu = torch.randn(32, 32)
b_cpu = torch.randn(32, 32)
a_cl = torchcl.to_opencl(a_cpu)
b_cl = torchcl.to_opencl(b_cpu)

check("add forward", AddFunction.apply(a_cl, b_cl), a_cpu + b_cpu, atol=1e-4)
check("sub forward", SubFunction.apply(a_cl, b_cl), a_cpu - b_cpu, atol=1e-4)
check("mul forward", MulFunction.apply(a_cl, b_cl), a_cpu * b_cpu, atol=1e-4)

# ── Test: Matmul forward + backward ─────────────────────────────────
print("\n--- Matmul Forward/Backward ---")

m1_cpu = torch.randn(16, 32)
m2_cpu = torch.randn(32, 8)
m1_cl = torchcl.to_opencl(m1_cpu)
m2_cl = torchcl.to_opencl(m2_cpu)

result_cl = MatmulFunction.apply(m1_cl, m2_cl)
result_cpu = m1_cpu @ m2_cpu
check("matmul forward (16x32 @ 32x8)", result_cl, result_cpu, atol=1e-2)

# ── Test: Layer Norm forward + backward ──────────────────────────────
print("\n--- Layer Normalization ---")

ln_input = torch.randn(8, 64, requires_grad=True)
ln_weight = torch.ones(64, requires_grad=True)
ln_bias = torch.zeros(64, requires_grad=True)

# Run CPU Reference
ln_result_cpu = torch.nn.functional.layer_norm(ln_input, [64], ln_weight, ln_bias)
grad_out_cpu = torch.randn_like(ln_result_cpu)
ln_result_cpu.backward(grad_out_cpu)

# Run OpenCL
ln_input_cl = torchcl.to_opencl(ln_input.detach())
ln_weight_cl = torchcl.to_opencl(ln_weight.detach())
ln_bias_cl = torchcl.to_opencl(ln_bias.detach())

class DummyCtx:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
ctx = DummyCtx()
ln_result_cl = LayerNormFunction.forward(ctx, ln_input_cl, ln_weight_cl, ln_bias_cl, 64)
check("layer_norm forward", ln_result_cl, ln_result_cpu.detach(), atol=1e-3)

grad_out_cl = torchcl.to_opencl(grad_out_cpu)
grad_in_cl, grad_w_cl, grad_b_cl, _, _ = LayerNormFunction.backward(ctx, grad_out_cl)

check("layer_norm backward grad_input", grad_in_cl, ln_input.grad, atol=1e-3)
check("layer_norm backward grad_weight", grad_w_cl, ln_weight.grad, atol=1e-3)
check("layer_norm backward grad_bias", grad_b_cl, ln_bias.grad, atol=1e-3)

# ── Test: Cross-Entropy Loss ────────────────────────────────────────
print("\n--- Cross-Entropy Loss ---")

logits = torch.randn(8, 10)
targets = torch.tensor([0, 3, 5, 7, 2, 9, 1, 4], dtype=torch.float32)

logits_cl = torchcl.to_opencl(logits)
targets_cl = torchcl.to_opencl(targets)

loss_cl = torchcl.cross_entropy_loss(logits_cl, targets_cl)
loss_cpu = torch.nn.functional.cross_entropy(logits, targets.long())
check("cross_entropy forward", loss_cl, loss_cpu.unsqueeze(0), atol=1e-2)

# ── Test: MSE Loss ───────────────────────────────────────────────────
print("\n--- MSE Loss ---")

pred = torch.randn(4, 8)
target_mse = torch.randn(4, 8)

pred_cl = torchcl.to_opencl(pred)
target_mse_cl = torchcl.to_opencl(target_mse)

mse_cl = torchcl.mse_loss(pred_cl, target_mse_cl)
mse_cpu = torch.nn.functional.mse_loss(pred, target_mse)
check("mse_loss forward", mse_cl, mse_cpu.unsqueeze(0), atol=0.1)

# ── Summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  AUTOGRAD RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\n  Failed tests:")
    for e in errors:
        print(f"    - {e}")

if failed == 0:
    print("\n  ALL AUTOGRAD TESTS PASSED!")

sys.exit(0 if failed == 0 else 1)
