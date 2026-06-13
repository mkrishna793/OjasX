"""
Profiler module for timing OpenCL kernel variants and profiling input data.
"""

from __future__ import annotations
import time
import numpy as np
import pyopencl as cl
from torchcl.runtime.context import get_queue, synchronize


class MicroProfiler:
    """Micro-profiler for timing kernel executions and caching performance characteristics."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, tuple, str], float] = {}

    def profile(self, run_fn, name: str, data_shape: tuple, config_str: str,
                warmup: int = 1, iterations: int = 3) -> float:
        """Profile a callable kernel execution and return execution time in milliseconds."""
        key = (name, data_shape, config_str)
        if key in self.cache:
            return self.cache[key]

        # Warmup
        for _ in range(warmup):
            run_fn()
        synchronize()

        # Run iterations and measure
        durations = []
        for _ in range(iterations):
            start = time.perf_counter()
            run_fn()
            synchronize()
            durations.append(time.perf_counter() - start)

        elapsed_ms = float(np.mean(durations)) * 1000.0
        self.cache[key] = elapsed_ms
        return elapsed_ms

    def clear(self):
        """Clear the timing cache."""
        self.cache.clear()


KernelProfiler = MicroProfiler



_global_profiler: MicroProfiler | None = None

def get_profiler() -> MicroProfiler:
    """Get the global micro-profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MicroProfiler()
    return _global_profiler
