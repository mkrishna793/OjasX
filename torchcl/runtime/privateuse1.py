import torch
import numpy as np
import weakref
from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.api import (
    _opencl_buffers,
    _wrap_output,
    _get_buf,
    _get_shape,
    _get_dtype,
)

HAS_CPP_EXTENSION = False

# ── Memory callbacks for C++ Allocator ───────────────────────────────

def cpp_allocate(size: int) -> int:
    """Invoked by C++ c10::Allocator to allocate OpenCL buffer."""
    cl_buf = get_buffer_pool().allocate(size)
    ptr = cl_buf.buffer.int_ptr
    _opencl_buffers[ptr] = cl_buf
    return ptr

def cpp_free(ptr: int) -> None:
    """Invoked by C++ c10::Allocator to free OpenCL buffer."""
    cl_buf = _opencl_buffers.pop(ptr, None)
    if cl_buf is not None:
        get_buffer_pool().free(cl_buf)

# ── Attempt to load C++ extension ────────────────────────────────────

try:
    import torchcl._C as _C
    torch._C._rename_privateuse1_backend("opencl")
    _C.register_allocator(cpp_allocate, cpp_free)
    HAS_CPP_EXTENSION = True
except ImportError:
    pass

# ── Operator implementations for PrivateUse1 dispatch key ─────────────

# Create PyTorch Library for PrivateUse1 dispatch key implementations
try:
    my_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
except Exception:
    my_lib = None

if my_lib is not None:
    def cl_copy_(self: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """Handles CPU <-> GPU copy operations for opencl tensors."""
        engine = get_engine()
        
        # Self is on opencl, src is on CPU
        if self.device.type == "privateuseone" and src.device.type == "cpu":
            self_buf = _opencl_buffers[self.data_ptr()]
            # Ensure src is contiguous and float32
            src_np = src.detach().cpu().numpy().astype(np.float32)
            get_buffer_pool().host_to_device(src_np, self_buf)
            
        # Self is on CPU, src is on opencl
        elif self.device.type == "cpu" and src.device.type == "privateuseone":
            src_buf = _opencl_buffers[src.data_ptr()]
            self_np = np.empty(src.shape, dtype=np.float32)
            get_buffer_pool().device_to_host(self_np, src_buf)
            # Copy data back to self
            self.copy_(torch.from_numpy(self_np))
            
        # Both self and src are on opencl
        elif self.device.type == "privateuseone" and src.device.type == "privateuseone":
            self_buf = _opencl_buffers[self.data_ptr()]
            src_buf = _opencl_buffers[src.data_ptr()]
            # Run simple elementwise copy kernel
            n = int(np.prod(src.shape))
            engine.run_elementwise_unary("copy_f32", src_buf, self_buf, n)
            
        return self

    my_lib.impl("copy_", cl_copy_)

    def cl_empty(size, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
        """Standard tensor allocation, handled automatically by the C++ allocator."""
        pass

    my_lib.impl("empty.memory_format", cl_empty)

    # Unary implementations
    def _register_unary(op_name, kernel_name):
        def unary_op(a: torch.Tensor) -> torch.Tensor:
            engine = get_engine()
            out = torch.empty_like(a)
            a_buf = _opencl_buffers[a.data_ptr()]
            out_buf = _opencl_buffers[out.data_ptr()]
            n = int(np.prod(a.shape))
            engine.run_elementwise_unary(kernel_name, a_buf, out_buf, n)
            return out
        my_lib.impl(op_name, unary_op)

    _register_unary("relu", "relu_f32")
    _register_unary("sigmoid", "sigmoid_f32")
    _register_unary("tanh", "tanh_f32")
    _register_unary("gelu", "gelu_f32")
    _register_unary("silu", "silu_f32")
    _register_unary("neg", "neg_f32")
    _register_unary("abs", "abs_f32")
    _register_unary("exp", "exp_f32")
    _register_unary("log", "log_f32")
    _register_unary("sqrt", "sqrt_f32")

    # Binary implementations
    def _register_binary(op_name, kernel_name):
        def binary_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            engine = get_engine()
            out = torch.empty_like(a)
            a_buf = _opencl_buffers[a.data_ptr()]
            b_buf = _opencl_buffers[b.data_ptr()]
            out_buf = _opencl_buffers[out.data_ptr()]
            n = int(np.prod(a.shape))
            engine.run_elementwise_binary(kernel_name, a_buf, b_buf, out_buf, n)
            return out
        my_lib.impl(op_name, binary_op)

    _register_binary("add.Tensor", "add_f32")
    _register_binary("sub.Tensor", "sub_f32")
    _register_binary("mul.Tensor", "mul_f32")
    _register_binary("div.Tensor", "div_f32")

    # Matrix multiplication
    def cl_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        engine = get_engine()
        out_shape = (a.shape[0], b.shape[1])
        out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
        a_buf = _opencl_buffers[a.data_ptr()]
        b_buf = _opencl_buffers[b.data_ptr()]
        out_buf = _opencl_buffers[out.data_ptr()]
        engine.run_matmul(a_buf, b_buf, out_buf, a.shape[0], a.shape[1], b.shape[1])
        return out

    my_lib.impl("mm", cl_mm)
