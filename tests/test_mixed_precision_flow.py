import torch
import numpy as np
import torchcl
from torchcl.liquid.dispatch import get_dispatcher
from torchcl.liquid.cost_model import KernelConfig

def test_mixed_precision_matmul():
    print("\n--- Test Mixed Precision Matmul (FP16 Flow) ---")
    
    # Force dispatcher to recommend half-precision tiled matmul
    dispatcher = get_dispatcher()
    original_dispatch = dispatcher.dispatch
    
    # Mock dispatch to return FP16 configuration
    def mock_dispatch(op, *args, **kwargs):
        return KernelConfig(workgroup_size=256, tile_size=16, strategy="tiled", precision="half")
    
    dispatcher.dispatch = mock_dispatch
    
    try:
        # Define matrix shapes
        M, K, N = 32, 32, 32
        
        # Create input matrices
        a_cpu = torch.randn(M, K)
        b_cpu = torch.randn(K, N)
        
        # Move to OpenCL
        a_cl = torchcl.to_opencl(a_cpu)
        b_cl = torchcl.to_opencl(b_cpu)
        
        # Run matmul (this will execute the FP16 flow: pack -> matmul_fp16 -> unpack)
        res_cl = torchcl.matmul(a_cl, b_cl)
        
        # Fetch back to CPU
        res_cpu = torchcl.to_cpu(res_cl)
        
        # Compute expected result
        expected = torch.matmul(a_cpu, b_cpu)
        
        # Verify output
        max_diff = torch.max(torch.abs(res_cpu - expected)).item()
        print(f"Max absolute difference: {max_diff:.6f}")
        
        # FP16 has lower precision, so atol=5e-2 is standard and acceptable
        assert torch.allclose(res_cpu, expected, atol=5e-2, rtol=1e-2)
        print("  [PASS] Dynamic mixed precision matmul (FP16 flow) is numerically correct.")
        
    finally:
        # Restore dispatcher
        dispatcher.dispatch = original_dispatch

if __name__ == "__main__":
    test_mixed_precision_matmul()
    print("\nAll mixed precision tests completed successfully!")
