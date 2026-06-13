# OjasX Build Guide — C++ PrivateUse1 Extension

OjasX features a hybrid runtime architecture designed to support both high-performance native compilation and seamless fallback execution on standard laptops.

---

## ⚡ Architecture Overview

1. **C++ PrivateUse1 Path (High Performance)**
   - Compiles a native PyTorch extension (`torchcl._C`) that registers a custom `c10::Allocator` on the `PrivateUse1` dispatch key.
   - Bridges allocations made via PyTorch's native C++ allocator directly to the Python-managed OpenCL buffer pool.
   - Bypasses Python interpreter overhead for tensor allocations.

2. **Pure-Python Fallback (Zero Setup)**
   - Automatically active if the C++ extension `_C` cannot be loaded (e.g. on laptops without build tools).
   - Intercepts operations using PyTorch's `__torch_dispatch__` mechanism with the `OjasXTensor` wrapper subclass and monkeypatches `.to("opencl")`.
   - Offers 100% API compatibility and enables testing/development anywhere.

---

## 🛠️ Prerequisites

To compile the C++ extension, ensure you have the following installed:

### 1. System Compiler
* **Windows**: Visual Studio 2019/2022 (MSVC) with the **"Desktop development with C++"** workload.
* **Linux**: GCC/G++ version 9 or higher (with C++17 support).

### 2. Build Tools & Dependencies
Install the required build dependencies using pip:
```bash
pip install -r requirements-cpp.txt
```
> [!NOTE]
> `ninja` is highly recommended on Windows for fast parallel compilation and compatibility with PyTorch's extension builder.

---

## 🚀 Building the Extension

### In-Place Compilation (Recommended for Development)
To build the C++ extension in-place within the `torchcl/` directory:
```bash
python setup.py build_ext --inplace
```

### Installing as a Editable Package
To build and install OjasX globally as an editable package:
```bash
pip install -e .
```

---

## 🔍 Verification

Verify whether the C++ extension has compiled successfully:

```bash
python -c "import torchcl.runtime.privateuse1 as pu1; print('C++ Extension Loaded:', pu1.HAS_CPP_EXTENSION)"
```

* **If C++ extension loaded successfully**: Outputs `C++ Extension Loaded: True` (tensor allocations and operators run natively via PrivateUse1).
* **If running python fallback**: Outputs `C++ Extension Loaded: False` (runs in fallback mode using `OjasXTensor` wrapper subclass).

---

## ❓ Troubleshooting

### 1. `ninja` not found or compiler errors on Windows
On Windows, Python might fail to locate `cl.exe` (the MSVC compiler). To resolve this:
1. Open the **"Developer Command Prompt for VS"** or **"Developer PowerShell for VS"** from the Start Menu.
2. Navigate to your project directory.
3. Run the compile/build command from within that prompt.

### 2. PyTorch and C++ Standard Mismatch
Ensure your Python environment uses a consistent version of PyTorch. The compiler must support `C++17` as required by LibTorch/PyTorch headers.
