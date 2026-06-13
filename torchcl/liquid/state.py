"""
LiquidState — Persistent GPU state buffers for Continuous Kernel Time.

A LiquidState holds a GPU buffer that persists across kernel launches,
enabling stateful computation (ODE integration, recurrence, etc.)
"""
from __future__ import annotations
import threading
import numpy as np
import pyopencl as cl
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.runtime.context import get_queue


class LiquidState:
    """Persistent GPU state for a continuous kernel."""
    __slots__ = ("buffer", "dt", "step_count", "error_history",
                 "shape", "dtype", "_converged", "_id")
    _counter = 0
    _lock = threading.Lock()

    def __init__(self, shape: tuple, dtype=np.float32, initial_dt: float = 0.1):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.dt = initial_dt
        self.step_count = 0
        self.error_history: list[float] = []
        self._converged = False
        n_bytes = int(np.prod(shape)) * self.dtype.itemsize
        self.buffer = get_buffer_pool().allocate(n_bytes, self.dtype, shape)
        # Zero-initialize
        get_buffer_pool().zero_fill(self.buffer)
        with LiquidState._lock:
            LiquidState._counter += 1
            self._id = LiquidState._counter

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def converged(self) -> bool:
        return self._converged

    def mark_converged(self):
        self._converged = True

    def update_dt(self, error: float, eps: float = 1e-4,
                  s_min: float = 0.2, s_max: float = 5.0, order: int = 2):
        """Adaptive step-size control (embedded RK method)."""
        self.error_history.append(error)
        if error < 1e-12:
            self.dt = min(self.dt * s_max, 10.0)
        else:
            factor = 0.9 * (eps / error) ** (1.0 / order)
            factor = max(s_min, min(s_max, factor))
            self.dt *= factor
        self.step_count += 1

    def read_host(self) -> np.ndarray:
        """Read state buffer back to CPU."""
        return get_buffer_pool().device_to_host(
            self.buffer, self.dtype, self.shape)

    def write_host(self, data: np.ndarray):
        """Write CPU data into state buffer."""
        data = np.ascontiguousarray(data, dtype=self.dtype)
        cl.enqueue_copy(get_queue(), self.buffer.buffer, data)

    def __repr__(self):
        return (f"LiquidState(id={self._id}, shape={self.shape}, "
                f"dt={self.dt:.4f}, steps={self.step_count})")


class StateManager:
    """Pool of LiquidStates with lifecycle tracking."""

    def __init__(self):
        self._states: dict[int, LiquidState] = {}
        self._lock = threading.Lock()

    def create(self, shape: tuple, dtype=np.float32,
               initial_dt: float = 0.1) -> LiquidState:
        state = LiquidState(shape, dtype, initial_dt)
        with self._lock:
            self._states[state._id] = state
        return state

    def get(self, state_id: int) -> LiquidState | None:
        return self._states.get(state_id)

    def release(self, state_id: int):
        with self._lock:
            s = self._states.pop(state_id, None)
        if s:
            get_buffer_pool().free(s.buffer)

    def release_all(self):
        with self._lock:
            for s in self._states.values():
                get_buffer_pool().free(s.buffer)
            self._states.clear()

    @property
    def active_count(self) -> int:
        return len(self._states)


_global_state_mgr: StateManager | None = None

def get_state_manager() -> StateManager:
    global _global_state_mgr
    if _global_state_mgr is None:
        _global_state_mgr = StateManager()
    return _global_state_mgr
