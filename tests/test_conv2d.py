"""
Test suite for OpenCLConv2d forward and backward passes.
"""

import numpy as np
import torch
import torchcl
from torchcl.nn import OpenCLConv2d
from torchcl.autograd import Conv2dFunction


def test_conv2d_direct_3x3():
    print("\n--- Test OpenCLConv2d Direct 3x3 ---")
    N, C_in, C_out = 2, 3, 4
    H, W = 8, 8
    stride, padding = (1, 1), (1, 1)
    
    conv_cl = OpenCLConv2d(C_in, C_out, kernel_size=3, padding=padding, bias=True)
    conv_cpu = torch.nn.Conv2d(C_in, C_out, kernel_size=3, padding=padding, bias=True)
    
    # Load same weights
    conv_cpu.weight.data.copy_(conv_cl.weight.data)
    conv_cpu.bias.data.copy_(conv_cl.bias.data)
    
    x_cpu = torch.randn(N, C_in, H, W, requires_grad=True)
    x_cl = torchcl.to_opencl(x_cpu.detach())
    
    # Forward
    y_cl = conv_cl(x_cl)
    y_cpu = conv_cpu(x_cpu)
    
    y_cl_cpu = torchcl.to_cpu(y_cl)
    assert np.allclose(y_cl_cpu.detach().numpy(), y_cpu.detach().numpy(), atol=1e-3)
    print("  Forward pass match")
    
    # Backward
    grad_out = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out)
    
    grad_out_cl = torchcl.to_opencl(grad_out)
    
    # Call backward directly with mock ctx
    mock_ctx = type('ctx', (), {
        'saved_tensors': (x_cl, conv_cl._weight_gpu, conv_cl._bias_gpu),
        '_strategy': 'direct',
        '_params': (stride, padding, y_cl_cpu.shape, x_cpu.shape, conv_cpu.weight.shape)
    })()
    
    grad_in_cl, grad_w_cl, grad_b_cl, _, _ = Conv2dFunction.backward(mock_ctx, grad_out_cl)
    
    # Verify input grad
    assert np.allclose(torchcl.to_cpu(grad_in_cl).numpy(), x_cpu.grad.numpy(), atol=1e-3)
    # Verify weight grad
    assert np.allclose(torchcl.to_cpu(grad_w_cl).numpy(), conv_cpu.weight.grad.numpy(), atol=1e-3)
    # Verify bias grad
    assert np.allclose(torchcl.to_cpu(grad_b_cl).numpy(), conv_cpu.bias.grad.numpy(), atol=1e-3)
    print("  Backward pass match")
    print("  [PASS] OpenCLConv2d Direct 3x3")


def test_conv2d_im2col():
    print("\n--- Test OpenCLConv2d im2col fallback ---")
    N, C_in, C_out = 2, 2, 3
    H, W = 6, 6
    stride, padding = (1, 1), (0, 0)
    
    # Using 2x2 kernel size triggers im2col fallback
    conv_cl = OpenCLConv2d(C_in, C_out, kernel_size=2, padding=padding, bias=True)
    conv_cpu = torch.nn.Conv2d(C_in, C_out, kernel_size=2, padding=padding, bias=True)
    
    conv_cpu.weight.data.copy_(conv_cl.weight.data)
    conv_cpu.bias.data.copy_(conv_cl.bias.data)
    
    x_cpu = torch.randn(N, C_in, H, W, requires_grad=True)
    x_cl = torchcl.to_opencl(x_cpu.detach())
    
    # Forward
    y_cl = conv_cl(x_cl)
    y_cpu = conv_cpu(x_cpu)
    
    y_cl_cpu = torchcl.to_cpu(y_cl)
    assert np.allclose(y_cl_cpu.detach().numpy(), y_cpu.detach().numpy(), atol=1e-3)
    print("  Forward pass match")
    
    # Backward
    grad_out = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out)
    
    grad_out_cl = torchcl.to_opencl(grad_out)
    
    # Call backward directly with mock ctx
    mock_ctx = type('ctx', (), {
        'saved_tensors': (x_cl, conv_cl._weight_gpu, conv_cl._bias_gpu),
        '_strategy': 'im2col',
        '_cols_buf': Conv2dFunction._cols_buf if hasattr(Conv2dFunction, '_cols_buf') else None,
        '_params': (stride, padding, y_cl_cpu.shape, x_cpu.shape, conv_cpu.weight.shape)
    })()
    
    # We must retrieve cols_buf that was saved on the last forward pass
    # Since Conv2dFunction is a singleton class pattern, let's fetch the actual saved ctx
    # Or simpler, we can let the backward logic recompute cols_buf if _cols_buf is None or mock it.
    # We already made our backward logic recompute it if strategy == 'direct', let's also let it recompute if _cols_buf is None!
    # Wait, in Conv2dFunction.backward:
    # "if ctx._strategy == 'direct' or getattr(ctx, '_cols_buf', None) is None:"
    # Let's verify if our Conv2dFunction.backward does that. It does:
    # "if ctx._strategy == 'direct':"
    # Let's modify autograd.py's Conv2dFunction.backward slightly to also accept None _cols_buf:
    # "if ctx._strategy == 'direct' or getattr(ctx, '_cols_buf', None) is None:"
    # That is extremely robust!
    
    # Let's prepare mock_ctx with _cols_buf = None so it recomputes it
    mock_ctx._cols_buf = None
    
    grad_in_cl, grad_w_cl, grad_b_cl, _, _ = Conv2dFunction.backward(mock_ctx, grad_out_cl)
    
    # Verify input grad
    assert np.allclose(torchcl.to_cpu(grad_in_cl).numpy(), x_cpu.grad.numpy(), atol=1e-3)
    # Verify weight grad
    assert np.allclose(torchcl.to_cpu(grad_w_cl).numpy(), conv_cpu.weight.grad.numpy(), atol=1e-3)
    print("  Backward pass match")
    print("  [PASS] OpenCLConv2d im2col")


if __name__ == "__main__":
    test_conv2d_direct_3x3()
    test_conv2d_im2col()
    print("\nAll conv2d tests completed successfully!")
