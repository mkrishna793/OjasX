"""
OjasX V3 — Native Tensor Subclass
Hooks into PyTorch's __torch_dispatch__ to intercept operations
and provide the 'device="opencl"' native feel.
"""

import torch
from .dispatch import dispatch_op

class OjasXTensor(torch.Tensor):
    """
    A PyTorch Tensor subclass that intercepts all operations.
    The actual data buffer lives on the OpenCL device, but PyTorch's
    internal metadata tracker thinks it's on the CPU. This bypasses
    the need for a C++ compiler to register the device.
    """
    @staticmethod
    def __new__(cls, elem):
        # Create a wrapper tensor. We tell PyTorch the device is CPU
        # so it doesn't crash trying to find an unregistered C++ device backend.
        real_size = getattr(elem, "_torchcl_shape", elem.size())
        dummy = torch.empty(real_size)
        real_strides = dummy.stride()
        
        r = torch.Tensor._make_wrapper_subclass(
            cls, 
            size=real_size, 
            strides=real_strides, 
            storage_offset=0,
            dtype=elem.dtype, 
            layout=elem.layout, 
            device=torch.device("cpu"), 
            requires_grad=elem.requires_grad
        )
        # Store the actual underlying tensor (which holds the OpenCL buffer)
        r._elem = elem
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        The magic method. Every time PyTorch tries to do something
        like `a + b` or `torch.matmul(a, b)` on an OjasXTensor, it gets routed here.
        """
        kwargs = kwargs or {}
        return dispatch_op(func, *args, **kwargs)

    @property
    def device(self):
        # Trick PyTorch into thinking the device is opencl
        class FakeDevice:
            def __init__(self):
                self.type = 'opencl'
            def __str__(self):
                return "opencl"
            def __repr__(self):
                return "device(type='opencl')"
            def __eq__(self, getattr):
                return str(getattr) == 'opencl' or (hasattr(getattr, 'type') and getattr.type == 'opencl')
        return FakeDevice()

    def __repr__(self):
        return f"OjasXTensor({self._elem}, device='opencl')"


# =====================================================================
# Native `tensor.to("opencl")` Monkeypatch
# =====================================================================

# Save the original PyTorch methods
_original_tensor_to = torch.Tensor.to
_original_module_to = torch.nn.Module.to

def _custom_tensor_to(self, *args, **kwargs):
    """Intercepts tensor.to('opencl')"""
    target_device = None
    if len(args) > 0 and isinstance(args[0], str):
        target_device = args[0]
    elif kwargs.get("device") is not None:
        target_device = str(kwargs.get("device"))
        
    if target_device == "opencl" or target_device == "device(type='opencl')":
        # Physically move data to OpenCL via OjasX V1 API
        import torchcl
        ocl_tensor = torchcl.to_opencl(self)
        # Wrap it in our Native dispatcher
        return OjasXTensor(ocl_tensor)
        
    if (target_device == "cpu" or target_device == "device(type='cpu')") and isinstance(self, OjasXTensor):
        import torchcl
        return torchcl.to_cpu(self._elem)
        
    # Not opencl? Let PyTorch handle it normally
    return _original_tensor_to(self, *args, **kwargs)

def _custom_module_to(self, *args, **kwargs):
    """Intercepts model.to('opencl')"""
    target_device = None
    if len(args) > 0 and isinstance(args[0], str):
        target_device = args[0]
    elif kwargs.get("device") is not None:
        target_device = str(kwargs.get("device"))
        
    if target_device == "opencl" or target_device == "device(type='opencl')":
        # Convert all parameters and buffers to OjasXTensor
        def _convert(t):
            return OjasXTensor(t.contiguous())
            
        for name, param in self.named_parameters(recurse=False):
            if param is not None:
                new_param = torch.nn.Parameter(_convert(param), requires_grad=param.requires_grad)
                self.register_parameter(name, new_param)
                
        for name, buf in self.named_buffers(recurse=False):
            if buf is not None:
                self.register_buffer(name, _convert(buf))
                
        for child in self.children():
            child.to("opencl")
            
        return self
        
    # Let PyTorch handle normally
    return _original_module_to(self, *args, **kwargs)

def apply_monkeypatches():
    """Applies the patches. Called automatically in __init__.py"""
    torch.Tensor.to = _custom_tensor_to
    torch.nn.Module.to = _custom_module_to
