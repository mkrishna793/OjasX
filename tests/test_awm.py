"""
Test suite for Adaptive Workgroup Morphing (AWM).
"""

import numpy as np
import torch
import torchcl
from torchcl.liquid.awm import AWMEngine


def test_awm_relu():
    print("\n--- Test AWM Relu ---")
    awm = AWMEngine(convergence_tol=1e-3, efficiency_threshold=0.75)
    n = 256
    
    # Inputs
    input_cpu = torch.randn(n) * 2.0
    input_cpu[:n//2] = -1.0  # Force convergence in first half (relu(-1) = 0, prev = 0)
    input_cl = torchcl.to_opencl(input_cpu)
    input_buf = torchcl.api._get_buf(input_cl)
    
    # Buffers for previous output (initialized to 0) and output
    prev_cl = torchcl.zeros(n)
    prev_buf = torchcl.api._get_buf(prev_cl)
    
    output_cl = torchcl.zeros(n)
    output_buf = torchcl.api._get_buf(output_cl)
    
    # Run morphed op
    stolen_count = awm.run_morphed_op("relu", input_buf, prev_buf, output_buf, n)
    
    # Read output
    output_np = torchcl.to_cpu(output_cl).numpy()
    expected_np = torch.relu(input_cpu).numpy()
    
    print(f"Stolen work items processed: {stolen_count}")
    print("AWM output first 5: ", output_np[:5])
    print("Expected first 5:   ", expected_np[:5])
    
    # Output should exactly match Relu
    assert np.allclose(output_np, expected_np, atol=1e-5)
    assert stolen_count > 0
    print("  [PASS] AWM Relu correctness")


def test_awm_sigmoid():
    print("\n--- Test AWM Sigmoid ---")
    awm = AWMEngine(convergence_tol=1e-3, efficiency_threshold=0.75)
    n = 256
    
    input_cpu = torch.randn(n)
    input_cpu[:n//2] = -10.0  # Force convergence in first half (sigmoid(-10) approx 0, prev = 0)
    input_cl = torchcl.to_opencl(input_cpu)
    input_buf = torchcl.api._get_buf(input_cl)
    
    prev_cl = torchcl.zeros(n)
    prev_buf = torchcl.api._get_buf(prev_cl)
    
    output_cl = torchcl.zeros(n)
    output_buf = torchcl.api._get_buf(output_cl)
    
    stolen_count = awm.run_morphed_op("sigmoid", input_buf, prev_buf, output_buf, n)
    
    output_np = torchcl.to_cpu(output_cl).numpy()
    expected_np = torch.sigmoid(input_cpu).numpy()
    
    print(f"Stolen work items processed: {stolen_count}")
    assert np.allclose(output_np, expected_np, atol=1e-5)
    assert stolen_count > 0
    print("  [PASS] AWM Sigmoid correctness")


if __name__ == "__main__":
    test_awm_relu()
    test_awm_sigmoid()
    print("\nAll AWM tests completed successfully!")
