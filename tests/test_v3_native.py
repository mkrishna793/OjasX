"""
OjasX V3 — Native PyTorch Integration Test
Tests the __torch_dispatch__ and tensor subclass architecture.
"""
import sys
import torch
import torch.nn as nn
import time

import torchcl

print("=" * 60)
print("  OjasX V3 — Native Integration Test")
print("=" * 60)

passed = failed = 0

def check(name, ok):
    global passed, failed
    if ok:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

# 1. Native Tensor Creation and Movement
print("\n--- Native Syntax ---")
t = torch.randn(512, 512)
t_ocl = t.to("opencl")
check("tensor.to('opencl') subclassing", type(t_ocl).__name__ == "OjasXTensor")
check("tensor.device shows as 'opencl'", str(t_ocl.device) == "opencl")

# 2. Operator Overloading (OjasX Dispatch)
print("\n--- Operator Overloading ---")
start = time.time()
res1 = t_ocl + t_ocl
add_time = time.time() - start

check("a + b operator overloading", type(res1).__name__ == "OjasXTensor")
check("addition correct", torch.allclose(res1.to("cpu"), (t + t)))

start = time.time()
res2 = t_ocl @ t_ocl
mm_time = time.time() - start

check("a @ b (matmul) overloading", type(res2).__name__ == "OjasXTensor")
print(f"  [INFO] Native @ matmul time: {mm_time*1000:.1f} ms")

# 3. CPU Fallback
print("\n--- Automatic CPU Fallback ---")
# Try an operation OjasX hasn't implemented in OpenCL (like log10)
try:
    res3 = torch.log10(t_ocl.abs())
    check("CPU fallback executed gracefully", type(res3).__name__ == "OjasXTensor")
except Exception as e:
    print(f"Fallback error: {e}")
    check("CPU fallback executed gracefully", False)

# 4. Native nn.Module Support
print("\n--- Native nn.Module Support ---")
# Standard PyTorch model (NO ocl_nn imports!)
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
# Magically move it to OpenCL
model = model.to("opencl")

check("model.to('opencl') converts parameters", type(model[0].weight).__name__ == "OjasXTensor")

x = torch.randn(32, 512).to("opencl")
out = model(x)
check("Standard nn.Sequential forward pass", type(out).__name__ == "OjasXTensor")
check("Forward pass shape correct", out.shape == (32, 10))

# ── Summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  V3 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
