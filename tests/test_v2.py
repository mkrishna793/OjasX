"""OjasX V2 Test — Autograd, nn.Module, Conv2d, MaxPool2d"""
import sys, torch, numpy as np
print("=" * 60)
print("  OjasX V2 — Comprehensive Test")
print("=" * 60)

import torchcl
from torchcl import nn as ocl_nn
from torchcl.autograd import ocl_matmul, ocl_relu, ocl_linear, ocl_softmax
from torchcl.api import to_opencl, to_cpu

passed = failed = 0

def check(name, ok):
    global passed, failed
    if ok:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

# ── Autograd tests ───────────────────────────────────────────────────
print("\n--- Autograd ---")

a = to_opencl(torch.randn(32, 64))
b = to_opencl(torch.randn(64, 16))
c = ocl_matmul(a, b)
check("autograd matmul forward shape", to_cpu(c).shape == (32, 16))

x = to_opencl(torch.randn(100))
y = ocl_relu(x)
check("autograd relu forward", to_cpu(y).min().item() >= 0)

# ── nn.Linear test ───────────────────────────────────────────────────
print("\n--- nn.Module Layers ---")

linear = ocl_nn.Linear(784, 256)
inp = to_opencl(torch.randn(32, 784))
out = linear(inp)
check("nn.Linear(784->256) shape", to_cpu(out).shape == (32, 256))
check("nn.Linear has params", len(linear.parameters()) == 2)

# ── nn.Sequential MLP ────────────────────────────────────────────────
print("\n--- Sequential MLP ---")

mlp = ocl_nn.Sequential(
    ocl_nn.Linear(784, 256),
    ocl_nn.ReLU(),
    ocl_nn.Linear(256, 128),
    ocl_nn.ReLU(),
    ocl_nn.Linear(128, 10),
    ocl_nn.Softmax(),
)
print(mlp)
inp = to_opencl(torch.randn(16, 784))
out = mlp(inp)
out_cpu = to_cpu(out)
check("MLP output shape", out_cpu.shape == (16, 10))
check("MLP softmax sums to 1", torch.allclose(out_cpu.sum(dim=1), torch.ones(16), atol=0.01))
check("MLP total params", len(mlp.parameters()) == 6)

# ── Conv2d test ──────────────────────────────────────────────────────
print("\n--- Conv2d ---")

conv = ocl_nn.Conv2d(1, 8, kernel_size=3, padding=1)
img = to_opencl(torch.randn(4, 1, 28, 28))  # Batch of 4 MNIST images
out = conv(img)
out_cpu = to_cpu(out)
check("Conv2d(1->8, k=3, p=1) shape", out_cpu.shape == (4, 8, 28, 28))

conv2 = ocl_nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
out2 = conv2(out)
out2_cpu = to_cpu(out2)
check("Conv2d(8->16, k=3, s=2) shape", out2_cpu.shape == (4, 16, 13, 13))

# ── MaxPool2d test ───────────────────────────────────────────────────
print("\n--- MaxPool2d ---")

pool = ocl_nn.MaxPool2d(2)
pooled = pool(out)
pooled_cpu = to_cpu(pooled)
check("MaxPool2d(2) shape", pooled_cpu.shape == (4, 8, 14, 14))

# ── BatchNorm1d test ─────────────────────────────────────────────────
print("\n--- BatchNorm1d ---")

bn = ocl_nn.BatchNorm1d(256)
x = to_opencl(torch.randn(32, 256))
y = bn(x)
y_cpu = to_cpu(y)
check("BatchNorm1d output shape", y_cpu.shape == (32, 256))
check("BatchNorm1d near zero mean", abs(y_cpu.mean().item()) < 0.5)

# ── Dropout test ─────────────────────────────────────────────────────
print("\n--- Dropout ---")

drop = ocl_nn.Dropout(0.5)
x = to_opencl(torch.ones(1000))
y_train = to_cpu(drop(x))
check("Dropout zeros some values", (y_train == 0).sum().item() > 100)
drop.eval()
y_eval = to_cpu(drop(x))
check("Dropout eval passes through", torch.allclose(y_eval, torch.ones(1000)))

# ── Full CNN test ────────────────────────────────────────────────────
print("\n--- Full CNN (Conv->ReLU->Pool->Flatten->Linear) ---")

cnn = ocl_nn.Sequential(
    ocl_nn.Conv2d(1, 8, 3, padding=1),
    ocl_nn.ReLU(),
    ocl_nn.MaxPool2d(2),
    ocl_nn.Conv2d(8, 16, 3, padding=1),
    ocl_nn.ReLU(),
    ocl_nn.MaxPool2d(2),
    ocl_nn.Flatten(),
    ocl_nn.Linear(16 * 7 * 7, 128),
    ocl_nn.ReLU(),
    ocl_nn.Linear(128, 10),
    ocl_nn.Softmax(),
)
print(cnn)
batch = to_opencl(torch.randn(8, 1, 28, 28))

import time
start = time.time()
result = cnn(batch)
elapsed = (time.time() - start) * 1000

result_cpu = to_cpu(result)
check("CNN output shape", result_cpu.shape == (8, 10))
check("CNN softmax valid", torch.allclose(result_cpu.sum(dim=1), torch.ones(8), atol=0.01))
print(f"  [INFO] CNN forward time: {elapsed:.0f} ms")

# ── Summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  V2 RESULTS: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
