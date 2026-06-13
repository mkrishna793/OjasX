"""
Adaptive Workgroup Morphing (AWM) — Dynamic thread and workgroup rebalancing.
"""

from __future__ import annotations
import numpy as np
import pyopencl as cl
from torchcl.runtime.context import get_queue
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.kernels.registry import get_kernel_registry


class AWMEngine:
    """AWM Engine manages dynamic workgroups and work-stealing for convergent operations."""

    def __init__(self, convergence_tol: float = 1e-3, efficiency_threshold: float = 0.5) -> None:
        self.convergence_tol = convergence_tol
        self.efficiency_threshold = efficiency_threshold
        self.registry = get_kernel_registry()
        self.pool = get_buffer_pool()

    def run_morphed_op(self, op: str, input_buf: CLBuffer, prev_output_buf: CLBuffer,
                      output_buf: CLBuffer, n: int) -> int:
        """Run an operation using Adaptive Workgroup Morphing.

        Splits workload into 50% private work and 50% shared work.
        Fast threads that converge early steal from the shared work queue.

        Returns:
            Number of work items processed via work-stealing.
        """
        queue = get_queue()
        
        # Split work: first 50% is private/base work, remaining is shared
        n_base = n // 2
        
        # Allocate global work queue counter, initialized to the base work size (n_base)
        work_queue_buf = self.pool.allocate(4, np.int32, (1,))
        cl.enqueue_copy(queue, work_queue_buf.buffer, np.array([n_base], dtype=np.int32))

        kernel_name = f"awm_{op}_f32"
        kernel = self.registry.get_kernel("awm_ops.cl", kernel_name)

        # Launch parameters: only launch threads for the private work (n_base)
        global_size = (self._round_up(n_base, 256),)
        local_size = (min(256, n_base),) if n_base >= 256 else (n_base,)

        # Local memory for tracking converged work items (size of int)
        local_converged_size = 4

        kernel(
            queue, global_size, local_size,
            input_buf.buffer,
            output_buf.buffer,
            prev_output_buf.buffer,
            work_queue_buf.buffer,
            cl.LocalMemory(local_converged_size),
            np.float32(self.convergence_tol),
            np.int32(n_base),
            np.int32(n)
        )

        # Read back work queue final counter to see how many were stolen
        stolen_end = np.empty(1, dtype=np.int32)
        cl.enqueue_copy(queue, stolen_end, work_queue_buf.buffer)
        queue.finish()

        self.pool.free(work_queue_buf)
        
        # Stolen items = final counter - n_base (clipped to total shared work)
        stolen_count = int(stolen_end[0] - n_base)
        stolen_count = min(n - n_base, stolen_count)
        return max(0, stolen_count)

    @staticmethod
    def _round_up(n: int, m: int) -> int:
        return ((n + m - 1) // m) * m
