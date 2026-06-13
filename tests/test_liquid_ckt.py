"""
Test suite for Continuous Kernel Time (CKT) execution engine and stateful kernels.
"""

import numpy as np
import torch
import torchcl
from torchcl.liquid.ckt_engine import get_ckt_engine
from torchcl.liquid.state import get_state_manager


def test_ckt_stateful_relu():
    print("\n--- Test CKT Stateful ReLU ---")
    ckt = get_ckt_engine(tau=1.0, eps=1e-3, method="rk2")
    shape = (100,)
    
    # Create persistent state
    state = ckt.create_state(shape, initial_dt=0.1)
    
    # Input tensor
    input_data = torch.ones(shape) * 2.0
    input_cl = torchcl.to_opencl(input_data)
    
    # Evolve state for multiple steps
    output_buf = ckt.evolve("relu", torchcl.api._get_buf(input_cl), state, max_steps=200, tol=1e-3)
    
    # Read output
    output_np = torchcl.runtime.memory.get_buffer_pool().device_to_host(output_buf, np.float32, shape)
    
    print(f"Initial state (zero): {state.read_host()[:5]}")
    print(f"Final evolved output: {output_np[:5]} (expected close to 2.0)")
    
    # Verify the state evolved towards the target value (2.0)
    assert np.allclose(output_np, 2.0, atol=2e-2)
    assert state.step_count > 0
    
    # Cleanup state
    get_state_manager().release(state._id)
    torchcl.runtime.memory.get_buffer_pool().free(output_buf)
    print("  [PASS] CKT stateful ReLU")


def test_ckt_ode_integration():
    print("\n--- Test CKT General-purpose ODE Integration ---")
    ckt = get_ckt_engine(tau=1.5, eps=1e-3, method="adaptive")
    shape = (50,)
    
    state = ckt.create_state(shape, initial_dt=0.2)
    
    target_data = torch.ones(shape) * 5.0
    target_cl = torchcl.to_opencl(target_data)
    
    # Run a single adaptive step
    output_cl = torchcl.zeros(shape)
    err = ckt.step_ode(
        torchcl.api._get_buf(target_cl),
        state,
        torchcl.api._get_buf(output_cl),
        method="adaptive"
    )
    
    print(f"Adaptive step error: {err:.6f}")
    assert err >= 0.0
    assert state.step_count == 1
    
    # Clean up
    get_state_manager().release(state._id)
    print("  [PASS] CKT general ODE integration")


if __name__ == "__main__":
    test_ckt_stateful_relu()
    test_ckt_ode_integration()
    print("\nAll CKT tests completed successfully!")
