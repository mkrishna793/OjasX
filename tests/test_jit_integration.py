import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import torchcl
import torchcl.api as api
from torchcl.runtime.memory import get_buffer_pool
from torchcl.jit.cache import get_kernel_cache

print("=" * 60)
print("  TorchCL API JIT Fusion Integration Test")
print("=" * 60)

passed = 0
failed = 0

def check(name, got, expected, atol=1e-4):
    global passed, failed
    if isinstance(got, torch.Tensor) and torchcl.is_opencl_tensor(got):
        got = torchcl.to_cpu(got)
    if isinstance(expected, torch.Tensor):
        expected = expected.float()
    
    if np.allclose(got, expected, atol=atol):
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")
        print(f"         Got:      {got.flatten()[:5]}")
        print(f"         Expected: {expected.flatten()[:5]}")

# ── Test 1: Unary Fusion Chain ──
print("\n--- Test 1: Unary Fusion Chain (relu -> sigmoid -> tanh) ---")

x = torch.randn(1024)
x_cl = torchcl.to_opencl(x)

# Without JIT fusion, this would run 3 kernels.
# With JIT fusion, it returns a lazy tensor and compiles/runs 1 fused kernel upon materialization (e.g. to_cpu)
y_cl = torchcl.tanh_(torchcl.sigmoid(torchcl.relu(x_cl)))

# Before materialization, y_cl should be lazy (no buffer in pool active list yet or not in opencl_buffers)
assert api._is_lazy(y_cl), "Expected y_cl to be lazy"

# Materialize it by pulling to CPU
y_cpu = torchcl.to_cpu(y_cl)

expected = torch.tanh(torch.sigmoid(torch.relu(x)))
check("relu -> sigmoid -> tanh result", y_cpu, expected)

# ── Test 2: Binary + Unary Fusion ──
print("\n--- Test 2: Binary + Unary Fusion (relu(a + b)) ---")

a = torch.randn(512, 512)
b = torch.randn(512, 512)
a_cl = torchcl.to_opencl(a)
b_cl = torchcl.to_opencl(b)

res_cl = torchcl.relu(torchcl.add(a_cl, b_cl))
assert api._is_lazy(res_cl), "Expected res_cl to be lazy"

res_cpu = torchcl.to_cpu(res_cl)
expected = torch.relu(a + b)
check("relu(a + b) result", res_cpu, expected)

# ── Test 3: Complex Multi-Fusion Cache Hit ──
print("\n--- Test 3: Cache Hit Verification ---")

# We run the same relu(a + b) chain again with different data.
# It should trigger a cache hit in the JIT compiler and run instantly.
a2 = torch.randn(512, 512)
b2 = torch.randn(512, 512)
a2_cl = torchcl.to_opencl(a2)
b2_cl = torchcl.to_opencl(b2)

cache = get_kernel_cache()
initial_hits = cache.stats()["hits"]

res2_cl = torchcl.relu(torchcl.add(a2_cl, b2_cl))
res2_cpu = torchcl.to_cpu(res2_cl)

expected2 = torch.relu(a2 + b2)
check("relu(a2 + b2) result", res2_cpu, expected2)

final_hits = cache.stats()["hits"]
print(f"  Initial hits: {initial_hits}, Final hits: {final_hits}")
if final_hits > initial_hits:
    passed += 1
    print("  [PASS] Cache hit triggered successfully!")
else:
    failed += 1
    print("  [FAIL] Cache hit was not triggered.")

# ── Summary ──
print()
print("=" * 60)
print(f"  INTEGRATION JIT RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

import sys
sys.exit(0 if failed == 0 else 1)
