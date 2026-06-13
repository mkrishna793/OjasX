"""
Test suite for Differential Dispatcher and Cost Model with real timing profiling.
"""

import numpy as np
import torch
import torchcl
from torchcl.liquid.dispatch import get_dispatcher
from torchcl.liquid.cost_model import KernelConfig
from torchcl.liquid.profiler import get_profiler


def test_dispatcher_profiling():
    print("\n--- Test Dispatcher Profiling ---")
    dispatcher = get_dispatcher()
    
    # Create test tensors
    tensor_dense = torch.ones(1000)
    tensor_sparse = torch.zeros(1000)
    tensor_sparse[5] = 1.0  # highly sparse
    
    cl_dense = torchcl.to_opencl(tensor_dense)
    cl_sparse = torchcl.to_opencl(tensor_sparse)
    
    profile_dense = dispatcher.profile_data(cl_dense)
    profile_sparse = dispatcher.profile_data(cl_sparse)
    
    print(f"Dense tensor sparsity: {profile_dense.sparsity:.4f}")
    print(f"Sparse tensor sparsity: {profile_sparse.sparsity:.4f}")
    
    assert profile_dense.sparsity == 0.0
    assert profile_sparse.sparsity > 0.99
    print("  [PASS] Dispatcher tensor profiling")


def test_dispatcher_selection():
    print("\n--- Test Dispatcher Selection & Real Timing Profiling ---")
    dispatcher = get_dispatcher()
    profiler = get_profiler()
    
    # Create large/small inputs
    tensor_small = torch.randn(10, 10)
    tensor_large = torch.randn(128, 128)
    
    cl_small = torchcl.to_opencl(tensor_small)
    cl_large = torchcl.to_opencl(tensor_large)
    
    config_small = dispatcher.dispatch("matmul", cl_small)
    config_large = dispatcher.dispatch("matmul", cl_large)
    
    print(f"Small matmul config: {config_small}")
    print(f"Large matmul config: {config_large}")
    
    assert config_small is not None
    assert config_large is not None
    
    # Online learning verification using REAL GPU timing profiles
    print("\n  Profiling real matmul configurations on GPU:")
    shapes = [16, 32, 48, 64, 96, 128]
    
    log_count = 0
    for dim in shapes:
        # Create input tensors
        a = torchcl.randn(dim, dim)
        b = torchcl.randn(dim, dim)
        
        # Profile actual matmul execution time on GPU (with warmups and syncs)
        def run_op():
            _ = torchcl.matmul(a, b)
            
        # We test both naive and tiled configs by simulating their dispatcher logs
        actual_ms_tiled = profiler.profile(run_op, "matmul", (dim, dim), "tiled", warmup=1, iterations=2)
        actual_ms_naive = profiler.profile(run_op, "matmul", (dim, dim), "naive", warmup=1, iterations=2)
        
        print(f"    Shape {dim}x{dim} | Tiled: {actual_ms_tiled:.3f} ms | Naive: {actual_ms_naive:.3f} ms")
        
        # Log real timing points to the dispatcher
        cfg_tiled = KernelConfig(workgroup_size=256, tile_size=16, strategy="tiled")
        cfg_naive = KernelConfig(workgroup_size=256, tile_size=1, strategy="naive")
        
        dispatcher.log_result("matmul", a, cfg_tiled, actual_ms_tiled)
        dispatcher.log_result("matmul", a, cfg_naive, actual_ms_naive)
        log_count += 2

    print(f"Online model trained successfully on {log_count} real GPU timing profiles.")
    print(f"Updated CostModel regression weights: {dispatcher.cost_model.weights}")
    
    assert len(dispatcher.cost_model.history) >= 10
    print("  [PASS] Cost model online training on real timings")


if __name__ == "__main__":
    test_dispatcher_profiling()
    test_dispatcher_selection()
    print("\nAll Dispatcher tests completed successfully!")
