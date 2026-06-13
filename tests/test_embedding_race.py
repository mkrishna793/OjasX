"""
Test suite to verify that embedding backward is race-free with duplicate indices.
"""

import numpy as np
import torch
import torchcl
from torchcl.nn import OpenCLEmbedding


def test_embedding_backward_race():
    print("\n--- Test Embedding Backward Duplicate Indices (Race Free) ---")
    num_embeddings = 10
    embedding_dim = 4
    
    # Create layers
    emb_cl = OpenCLEmbedding(num_embeddings, embedding_dim)
    emb_cpu = torch.nn.Embedding(num_embeddings, embedding_dim)
    
    # Load same weights
    emb_cpu.weight.data.copy_(emb_cl.weight.data)
    
    # Create duplicate indices (all referencing index 3)
    indices_cpu = torch.tensor([3, 3, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    
    # Forward pass
    y_cl = emb_cl(indices_cpu)
    y_cpu = emb_cpu(indices_cpu)
    
    # Backward pass
    grad_out = torch.ones_like(y_cpu)
    y_cpu.backward(grad_out)
    
    # Call backward directly for CL
    from torchcl.autograd import _get_buf, _wrap_output
    engine = torchcl.ops.engine.get_engine()
    pool = torchcl.runtime.memory.get_buffer_pool()
    
    indices_float = indices_cpu.float()
    indices_cl = torchcl.to_opencl(indices_float)
    
    grad_out_cl = torchcl.to_opencl(grad_out)
    
    grad_weight_buf = pool.allocate(num_embeddings * embedding_dim * 4, np.dtype(np.float32), (num_embeddings, embedding_dim))
    pool.zero_fill(grad_weight_buf)
    
    engine.run_embedding_backward(
        _get_buf(grad_out_cl),
        _get_buf(indices_cl),
        grad_weight_buf,
        len(indices_cpu),
        embedding_dim
    )
    
    grad_weight_cl_cpu = pool.device_to_host(grad_weight_buf, np.float32, (num_embeddings, embedding_dim))
    pool.free(grad_weight_buf)
    
    # Check that CL results match CPU results (accumulated sum of 8.0 for index 3)
    print("CL grad weight row 3:  ", grad_weight_cl_cpu[3])
    print("CPU grad weight row 3: ", emb_cpu.weight.grad[3].numpy())
    
    assert np.allclose(grad_weight_cl_cpu, emb_cpu.weight.grad.numpy(), atol=1e-5)
    print("  [PASS] Embedding backward race-free accumulation")


if __name__ == "__main__":
    test_embedding_backward_race()
    print("\nAll embedding backward race tests completed successfully!")
