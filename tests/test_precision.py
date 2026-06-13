"""
Test suite for Adaptive Precision Streaming (APS).
"""

import numpy as np
import torch
import torchcl
from torchcl.liquid.precision import AdaptivePrecision


def test_precision_analysis():
    print("\n--- Test Precision Analysis ---")
    ap = AdaptivePrecision()
    
    # Large range values -> float32 recommended
    data_large = torchcl.to_opencl(torch.randn(100) * 10000.0)
    # Small range values -> float16/int8 recommended
    data_small = torchcl.to_opencl(torch.randn(100) * 0.1)
    
    _, _, prec_large = ap.analyze_precision(torchcl.api._get_buf(data_large), 100)
    _, _, prec_small = ap.analyze_precision(torchcl.api._get_buf(data_small), 100)
    
    print(f"Large range suggested precision: {prec_large}")
    print(f"Small range suggested precision: {prec_small}")
    
    assert prec_large == "float32"
    assert prec_small in ("float16", "int8")
    print("  [PASS] Precision analysis suggestion")


def test_precision_packing_roundtrip():
    print("\n--- Test FP32 to FP16 Packing Roundtrip ---")
    ap = AdaptivePrecision()
    n = 128
    
    # Original FP32 data
    orig_cpu = torch.randn(n) * 2.0
    orig_cl = torchcl.to_opencl(orig_cpu)
    orig_buf = torchcl.api._get_buf(orig_cl)
    
    # Pack to FP16 on GPU
    packed_buf = ap.pack_to_fp16(orig_buf, n)
    
    # Unpack back to FP32 on GPU
    unpacked_buf = ap.unpack_from_fp16(packed_buf, n)
    
    # Read back to CPU
    unpacked_np = torchcl.runtime.memory.get_buffer_pool().device_to_host(unpacked_buf, np.float32, (n,))
    
    # Verify values are close (FP16 tolerance is about 1e-3 for these ranges)
    assert np.allclose(unpacked_np, orig_cpu.numpy(), atol=5e-3, rtol=1e-3)
    
    # Cleanup
    torchcl.runtime.memory.get_buffer_pool().free(packed_buf)
    torchcl.runtime.memory.get_buffer_pool().free(unpacked_buf)
    
    print("Original first 5: ", orig_cpu.numpy()[:5])
    print("Unpacked first 5: ", unpacked_np[:5])
    print("  [PASS] Precision packing FP32 <-> FP16")


if __name__ == "__main__":
    test_precision_analysis()
    test_precision_packing_roundtrip()
    print("\nAll Precision tests completed successfully!")
