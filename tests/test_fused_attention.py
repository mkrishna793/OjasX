import torch
import numpy as np
import torchcl

def test_fused_attention_correctness():
    print("\n--- Test Fused Attention Correctness ---")
    
    # Dimensions: Batch=2, Heads=4, SeqLen=32, HeadDim=64
    B, H, M, N, D = 2, 4, 32, 32, 64
    
    # Create Q, K, V tensors
    q = torch.randn(B, H, M, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    
    # Run CPU Reference
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(probs, v)
    
    # Move inputs to OpenCL
    q_cl = torchcl.to_opencl(q)
    k_cl = torchcl.to_opencl(k)
    v_cl = torchcl.to_opencl(v)
    
    # Run Fused Attention on OpenCL
    cl_out = torchcl.fused_attention(q_cl, k_cl, v_cl)
    cl_out_cpu = torchcl.to_cpu(cl_out)
    
    # Compute error
    max_diff = torch.max(torch.abs(cl_out_cpu - ref_out)).item()
    print(f"Max absolute difference: {max_diff:.6f}")
    
    # Assert correctness
    assert torch.allclose(cl_out_cpu, ref_out, atol=1e-3, rtol=1e-3)
    print("  [PASS] Fused scaled dot-product attention matches PyTorch CPU reference.")

if __name__ == "__main__":
    test_fused_attention_correctness()
    print("\nAll Fused Attention tests completed successfully!")
