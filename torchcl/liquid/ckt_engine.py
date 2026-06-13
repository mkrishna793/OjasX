"""
CKT Engine — Continuous Kernel Time execution controller.

Launches stateful OpenCL kernels that maintain persistent GPU state
across invocations. Manages adaptive step-size via error feedback.
"""
from __future__ import annotations
import numpy as np
import pyopencl as cl
from torchcl.runtime.context import get_queue
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.kernels.registry import get_kernel_registry
from torchcl.liquid.state import LiquidState, get_state_manager
from torchcl.ops.engine import get_engine

_LIQUID_OPS = {
    "relu", "sigmoid", "tanh", "gelu",
}

_ODE_METHODS = {"euler", "rk2", "rk4", "adaptive"}


class CKTEngine:
    """Continuous Kernel Time execution engine.

    Instead of stateless kernel launches, CKT kernels evolve a persistent
    state S(t) via ODE integration: dS/dt = F(S, X, θ).
    """

    def __init__(self, tau: float = 1.0, eps: float = 1e-4,
                 method: str = "rk2"):
        self._tau = tau
        self._eps = eps
        self._method = method
        self._registry = get_kernel_registry()
        self._pool = get_buffer_pool()
        self._state_mgr = get_state_manager()

    def create_state(self, shape: tuple, initial_dt: float = 0.1) -> LiquidState:
        """Create a new persistent state buffer."""
        return self._state_mgr.create(shape, np.float32, initial_dt)

    def step(self, op: str, input_buf: CLBuffer, state: LiquidState,
             output_buf: CLBuffer) -> float:
        """Execute one CKT step: evolve state from S(t) to S(t+dt).

        Args:
            op: Operation name ('relu', 'sigmoid', 'tanh', 'gelu')
            input_buf: Current input X(t)
            state: Persistent LiquidState
            output_buf: Output buffer for y(t+dt)

        Returns:
            Max error from this step (for adaptive dt control).
        """
        if op not in _LIQUID_OPS:
            raise ValueError(f"Unknown liquid op: {op}. Available: {_LIQUID_OPS}")

        queue = get_queue()
        n = state.numel
        kernel_name = f"liquid_{op}_f32"
        kernel = self._registry.get_kernel("liquid_elementwise.cl", kernel_name)

        # Allocate error buffer
        error_buf = self._pool.allocate(n * 4, np.float32, (n,))

        global_size = (self._round_up(n, 256),)
        local_size = (min(256, n),) if n >= 256 else None

        kernel(queue, global_size, local_size,
               input_buf.buffer,
               state.buffer.buffer,
               output_buf.buffer,
               error_buf.buffer,
               np.float32(state.dt),
               np.float32(self._tau),
               np.int32(n))

        # Run max reduction on GPU
        engine = get_engine()
        max_error_gpu = self._pool.allocate(4, np.float32, (1,))
        engine.run_reduction("max_f32", error_buf, max_error_gpu, n)

        # Read back only a single float
        max_error = float(self._pool.device_to_host(max_error_gpu, np.float32, (1,))[0])
        self._pool.free(max_error_gpu)
        self._pool.free(error_buf)

        # Update adaptive dt
        state.update_dt(max_error, self._eps)

        # Check convergence
        if max_error < self._eps * 0.01:
            state.mark_converged()

        return max_error

    def step_ode(self, target_buf: CLBuffer, state: LiquidState,
                 output_buf: CLBuffer, method: str | None = None) -> float:
        """Execute one ODE integration step using the specified method.

        This is the general-purpose ODE integrator that can evolve any
        state toward any target function.
        """
        method = method or self._method
        if method not in _ODE_METHODS:
            raise ValueError(f"Unknown ODE method: {method}")

        queue = get_queue()
        n = state.numel

        if method == "adaptive":
            kernel = self._registry.get_kernel("liquid_ode.cl", "ode_adaptive_f32")
            error_buf = self._pool.allocate(n * 4, np.float32, (n,))
            global_size = (self._round_up(n, 256),)
            local_size = (min(256, n),) if n >= 256 else None

            kernel(queue, global_size, local_size,
                   target_buf.buffer, state.buffer.buffer,
                   output_buf.buffer, error_buf.buffer,
                   np.float32(state.dt), np.float32(self._tau), np.int32(n))

            # Run max reduction on GPU
            engine = get_engine()
            max_error_gpu = self._pool.allocate(4, np.float32, (1,))
            engine.run_reduction("max_f32", error_buf, max_error_gpu, n)

            # Read back only a single float
            max_error = float(self._pool.device_to_host(max_error_gpu, np.float32, (1,))[0])
            self._pool.free(max_error_gpu)
            self._pool.free(error_buf)
            state.update_dt(max_error, self._eps)
            return max_error
        else:
            kernel_name = f"ode_{method}_f32"
            kernel = self._registry.get_kernel("liquid_ode.cl", kernel_name)
            global_size = (self._round_up(n, 256),)
            local_size = (min(256, n),) if n >= 256 else None

            kernel(queue, global_size, local_size,
                   target_buf.buffer, state.buffer.buffer,
                   output_buf.buffer,
                   np.float32(state.dt), np.float32(self._tau), np.int32(n))

            state.step_count += 1
            return 0.0  # No error estimate for fixed-step methods

    def evolve(self, op: str, input_buf: CLBuffer, state: LiquidState,
               max_steps: int = 100, tol: float = 1e-4) -> CLBuffer:
        """Evolve state until convergence (total time >= 5 * tau) or max_steps.

        Returns the final output buffer.
        """
        n = state.numel
        output_buf = self._pool.allocate(n * 4, np.float32, state.shape)

        target_time = 5.0 * self._tau
        t_accum = 0.0
        steps = 0

        while t_accum < target_time and steps < max_steps:
            dt = state.dt
            self.step(op, input_buf, state, output_buf)
            t_accum += dt
            steps += 1

        return output_buf

    @staticmethod
    def _round_up(n: int, m: int) -> int:
        return ((n + m - 1) // m) * m


_global_ckt: CKTEngine | None = None

def get_ckt_engine(**kwargs) -> CKTEngine:
    global _global_ckt
    if _global_ckt is None:
        _global_ckt = CKTEngine(**kwargs)
    return _global_ckt
