"""
Test suite for Continuous Memory (CMEM) allocator and LiquidTensor.
"""

import numpy as np
import torch
import torchcl
from torchcl.liquid.memory import LiquidMemoryPool, LiquidTensor


def test_liquid_memory_pool():
    print("\n--- Test Liquid Memory Pool ---")
    pool = LiquidMemoryPool(size_mb=2)
    
    # Allocate a sub-buffer
    cl_buf1 = pool.allocate(1024, np.dtype(np.float32), (256,))
    cl_buf2 = pool.allocate(2048, np.dtype(np.float32), (512,))
    
    assert cl_buf1.nbytes == 1024
    assert cl_buf2.nbytes == 2048
    
    # Free buffers to release active ranges
    pool.free(cl_buf1)
    pool.free(cl_buf2)
    print("  [PASS] Liquid memory pool allocation")


def test_liquid_tensor_append_and_view():
    print("\n--- Test Liquid Tensor Appending and Zero-copy Slicing ---")
    max_shape = (10, 4)
    # Create LiquidTensor
    lt = LiquidTensor(max_shape, dtype=np.float32)
    
    assert lt.shape == (0, 4)
    
    # Create a chunk of data to append (shape 2, 4)
    chunk1_cpu = torch.ones(2, 4) * 1.5
    chunk1_cl = torchcl.to_opencl(chunk1_cpu)
    chunk1_buf = torchcl.api._get_buf(chunk1_cl)
    
    # Append
    lt.append(chunk1_buf, axis=0)
    assert lt.shape == (2, 4)
    
    # Create another chunk to append (shape 3, 4)
    chunk2_cpu = torch.ones(3, 4) * 2.5
    chunk2_cl = torchcl.to_opencl(chunk2_cpu)
    chunk2_buf = torchcl.api._get_buf(chunk2_cl)
    
    # Append
    lt.append(chunk2_buf, axis=0)
    assert lt.shape == (5, 4)
    
    # Slice view of elements [2:5]
    view_buf = lt.view(2, 5, axis=0)
    assert view_buf.shape == (3, 4)
    
    # Read back view data
    view_np = torchcl.runtime.memory.get_buffer_pool().device_to_host(view_buf, np.float32, (3, 4))
    
    print("Evolved LiquidTensor shape: ", lt.shape)
    print("Slice view [2:5] contents:\n", view_np)
    
    # The elements [2:5] correspond to the second chunk, which was initialized to 2.5
    assert np.allclose(view_np, 2.5)
    
    # Clean up
    lt.release()
    torchcl.runtime.memory.get_buffer_pool().free(view_buf)
    
    print("  [PASS] Liquid tensor append and view")


if __name__ == "__main__":
    test_liquid_memory_pool()
    test_liquid_tensor_append_and_view()
    print("\nAll Memory tests completed successfully!")
