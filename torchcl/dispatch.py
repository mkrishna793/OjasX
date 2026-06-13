"""
OjasX V3 — Native Dispatcher
Routes PyTorch ATen operations to OjasX OpenCL kernels or falls back to CPU.
"""

import torch
import torchcl
import torchcl.autograd as autograd

# Dictionary mapping PyTorch ATen operations to our OjasX implementation
DISPATCH_TABLE = {
    torch.ops.aten.add.Tensor: autograd.ocl_add if hasattr(autograd, 'ocl_add') else torchcl.add,
    torch.ops.aten.mul.Tensor: autograd.ocl_mul if hasattr(autograd, 'ocl_mul') else torchcl.mul,
    torch.ops.aten.mm.default: autograd.ocl_matmul,
    torch.ops.aten.relu.default: autograd.ocl_relu,
    torch.ops.aten.sigmoid.default: autograd.ocl_sigmoid,
    torch.ops.aten.tanh.default: autograd.ocl_tanh if hasattr(autograd, 'ocl_tanh') else torchcl.tanh_,
    # PyTorch linear often uses addmm
}

# Special ops that need custom mapping or shape handling can go here.
# For example, ATen addmm: out = beta * input + alpha * (mat1 @ mat2)
def ocl_addmm(input, mat1, mat2, *, beta=1, alpha=1):
    # Just a simple implementation for nn.Linear support
    mm_res = autograd.ocl_matmul(mat1, mat2)
    if alpha != 1:
        # We don't have scalar mul yet, fallback or implement
        pass 
    res = autograd.ocl_add(input, mm_res) # ignores alpha/beta for basic usage
    return res

# We'll rely heavily on CPU fallback for unhandled ops to guarantee 100% compatibility.

def execute_on_cpu(func, *args, **kwargs):
    """
    Automatic CPU Fallback:
    If OjasX doesn't have an OpenCL kernel for this specific PyTorch operation yet,
    we seamlessly move the tensors to the CPU, run the PyTorch native operation,
    and then bring the result back to OpenCL.
    """
    from torchcl.tensor import OjasXTensor
    import torchcl
    
    # 1. Unwrap OjasXTensors and pull the data to CPU
    def unwrap_to_cpu(x):
        if isinstance(x, OjasXTensor):
            return torchcl.to_cpu(x._elem)
        return x
        
    cpu_args = torch.utils._pytree.tree_map(unwrap_to_cpu, args)
    cpu_kwargs = torch.utils._pytree.tree_map(unwrap_to_cpu, kwargs)
    
    # 2. Execute natively on PyTorch CPU
    res = func(*cpu_args, **cpu_kwargs)
    
    # 3. Wrap the result back into OjasXTensor (moves back to OpenCL)
    def wrap_to_opencl(x):
        if isinstance(x, torch.Tensor):
            return OjasXTensor(torchcl.to_opencl(x.contiguous()))
        return x
        
    return torch.utils._pytree.tree_map(wrap_to_opencl, res)


def dispatch_op(func, *args, **kwargs):
    """Routes the ATen operation either to OpenCL or falls back to CPU."""
    if func in DISPATCH_TABLE:
        try:
            # Try to run on OpenCL
            # Unpack OjasXTensors for our internal API
            from torchcl.tensor import OjasXTensor
            def unwrap(x):
                return x._elem if isinstance(x, OjasXTensor) else x
            
            ocl_args = torch.utils._pytree.tree_map(unwrap, args)
            ocl_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
            
            res = DISPATCH_TABLE[func](*ocl_args, **ocl_kwargs)
            
            # Wrap result back into OjasXTensor
            def wrap(x):
                return OjasXTensor(x) if isinstance(x, torch.Tensor) else x
            return torch.utils._pytree.tree_map(wrap, res)
            
        except Exception as e:
            # If our OpenCL kernel fails for some edge case (e.g., weird broadcasting),
            # gracefully fall back to CPU instead of crashing.
            print(f"[OjasX] OpenCL dispatch failed for {func.__name__}, falling back to CPU. ({e})")
            return execute_on_cpu(func, *args, **kwargs)
    
    # Operation not yet supported natively on OpenCL, use CPU fallback
    return execute_on_cpu(func, *args, **kwargs)
