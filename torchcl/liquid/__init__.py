"""
OjasX Liquid Compute Engine
============================

The breakthrough subsystem that transforms OjasX from a static OpenCL wrapper
into a dynamically adaptive compute engine. Six pillars:

1. Continuous Kernel Time (CKT)     — Stateful kernels with adaptive ODE stepping
2. Differential Dispatch (DD)       — Data-aware kernel selection via cost model
3. Adaptive Workgroup Morphing (AWM) — Dynamic thread rebalancing via work-stealing
4. Adaptive Precision Streaming (APS)— Per-region mixed precision
5. Continuous Memory (CMEM)         — Ring-buffer dynamic tensors
6. Liquid Graph                     — Runtime-morphing computation graphs
"""

from torchcl.liquid.state import LiquidState, StateManager
from torchcl.liquid.ckt_engine import CKTEngine
from torchcl.liquid.dispatch import DifferentialDispatcher, DataProfile, KernelConfig
from torchcl.liquid.cost_model import CostModel
from torchcl.liquid.profiler import KernelProfiler
from torchcl.liquid.precision import AdaptivePrecision, PrecisionMap
from torchcl.liquid.awm import AWMEngine
from torchcl.liquid.memory import LiquidMemoryPool, LiquidTensor

__all__ = [
    "LiquidState", "StateManager",
    "CKTEngine",
    "DifferentialDispatcher", "DataProfile", "KernelConfig",
    "CostModel",
    "KernelProfiler",
    "AdaptivePrecision", "PrecisionMap",
    "AWMEngine",
    "LiquidMemoryPool", "LiquidTensor",
]
