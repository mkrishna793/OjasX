import torch

class OjasXTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        # The storage and tensor metadata are kept on CPU
        # But we intercept operations
        r = torch.Tensor._make_wrapper_subclass(
            cls, size=elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, layout=elem.layout, device=torch.device("cpu"), requires_grad=elem.requires_grad
        )
        r._elem = elem
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        print(f"Intercepted: {func.__name__}")
        
        def unwrap(x):
            return x._elem if isinstance(x, OjasXTensor) else x
            
        def wrap(x):
            return OjasXTensor(x) if isinstance(x, torch.Tensor) else x

        cpu_args = torch.utils._pytree.tree_map(unwrap, args)
        cpu_kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
        
        res = func(*cpu_args, **cpu_kwargs)
        
        return torch.utils._pytree.tree_map(wrap, res)

original_to = torch.Tensor.to
def custom_to(self, *args, **kwargs):
    if len(args) > 0 and args[0] == "opencl":
        return OjasXTensor(self)
    if kwargs.get("device") == "opencl":
        return OjasXTensor(self)
    return original_to(self, *args, **kwargs)

torch.Tensor.to = custom_to

t = torch.randn(3, 3)
t_ocl = t.to("opencl")
print("Converted to OjasXTensor:", type(t_ocl))
result = t_ocl + t_ocl
print("Result type:", type(result))
