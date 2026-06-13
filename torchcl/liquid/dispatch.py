"""
Differential Dispatcher — Dynamic kernel configuration selection.
Analyzes data profile via O(1) sampling and hardware profile to pick the optimal execution parameters.
"""

from __future__ import annotations
import numpy as np
import torch
import pyopencl as cl

from torchcl.runtime.context import get_queue, get_device_info
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.api import _get_buf, _get_shape, _get_dtype, is_opencl_tensor
from torchcl.liquid.cost_model import CostModel, DataProfile, HardwareProfile, KernelConfig

# Map PyTorch dtypes to NumPy dtypes
TORCH_TO_NUMPY = {
    torch.float32: np.dtype(np.float32),
    torch.float16: np.dtype(np.float16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
}


class DifferentialDispatcher:
    """Dynamically tunes and dispatches operations based on data and hardware properties."""

    def __init__(self) -> None:
        self.cost_model = CostModel()
        self._hw_profile: HardwareProfile | None = None

    @property
    def hardware_profile(self) -> HardwareProfile:
        if self._hw_profile is None:
            info = get_device_info()
            # Extract fields safely, providing defaults if not present
            self._hw_profile = HardwareProfile(
                compute_units=info.get("max_compute_units", 1),
                local_mem_kb=info.get("local_mem_size_kb", 4),
                max_workgroup=info.get("max_work_group_size", 256),
                global_mem_mb=info.get("global_mem_size_mb", 512),
                preferred_vector_width=info.get("preferred_vector_width_float", 1)
            )
        return self._hw_profile

    def profile_data(self, tensor: torch.Tensor | CLBuffer) -> DataProfile:
        """Sample data to construct a DataProfile in O(1) time."""
        if isinstance(tensor, torch.Tensor):
            if is_opencl_tensor(tensor):
                buf = _get_buf(tensor)
                shape = _get_shape(tensor)
                torch_dtype = _get_dtype(tensor)
                np_dtype = TORCH_TO_NUMPY.get(torch_dtype, np.dtype(np.float32))
            else:
                # CPU tensor
                shape = tuple(tensor.shape)
                np_dtype = TORCH_TO_NUMPY.get(tensor.dtype, np.dtype(np.float32))
                # Sample CPU tensor directly
                arr = tensor.detach().cpu().numpy().ravel()
                sample_size = min(256, len(arr))
                if sample_size > 0:
                    sample = arr[:sample_size]
                    sparsity = float(np.mean(sample == 0))
                    mean = float(np.mean(sample))
                    std = float(np.std(sample))
                else:
                    sparsity, mean, std = 0.0, 0.0, 0.0
                return DataProfile(shape, sparsity, mean, std, np_dtype)
        else:
            buf = tensor
            shape = buf.shape or (buf.nbytes // 4,)
            np_dtype = buf.dtype or np.dtype(np.float32)

        n = int(np.prod(shape))
        sample_size = min(256, n)
        if sample_size <= 0:
            return DataProfile(shape, 0.0, 0.0, 0.0, np_dtype)

        # Copy sample from GPU to CPU
        queue = get_queue()
        sample = np.empty(sample_size, dtype=np_dtype)
        try:
            cl.enqueue_copy(queue, sample, buf.buffer)
            queue.finish()
            sparsity = float(np.mean(sample == 0))
            mean = float(np.mean(sample))
            std = float(np.std(sample))
        except Exception:
            sparsity, mean, std = 0.0, 0.0, 0.0

        return DataProfile(shape, sparsity, mean, std, np_dtype)

    def get_candidate_configs(self, op: str, data: DataProfile) -> list[KernelConfig]:
        """Generate a list of candidate KernelConfigs for the given operation."""
        hw = self.hardware_profile
        candidates = []

        # Match candidate precision to input data type
        prec = "half" if data.dtype in (np.float16, np.dtype(np.float16)) else "float"

        if op == "matmul":
            # Strategies: naive, tiled
            strategies = ["naive"]
            if hw.local_mem_kb >= 4:
                strategies.append("tiled")

            # Tile sizes
            tile_sizes = [4, 8]
            if hw.local_mem_kb >= 16:
                tile_sizes.append(16)

            # Workgroup sizes
            wg_sizes = [64, 128, 256]

            for strat in strategies:
                for wg in wg_sizes:
                    if wg <= hw.max_workgroup:
                        if strat == "tiled":
                            for tile in tile_sizes:
                                candidates.append(KernelConfig(wg, tile, strat, prec))
                        else:
                            candidates.append(KernelConfig(wg, 1, strat, prec))
        else:
            # Elementwise / Activation operations
            # Workgroup sizes: power of 2 up to max_workgroup
            wg_sizes = [64, 128, 256, 512]
            for wg in wg_sizes:
                if wg <= hw.max_workgroup:
                    candidates.append(KernelConfig(wg, 1, "default", "float"))

        if not candidates:
            candidates.append(KernelConfig(64, 1, "default", "float"))

        return candidates

    def dispatch(self, op: str, *tensors, next_op: str | None = None) -> KernelConfig:
        """Select the best configuration for the operation based on input tensors."""
        if not tensors:
            # Fallback configuration
            return KernelConfig(256, 1, "default", "float")

        # Profile the first primary input tensor
        data = self.profile_data(tensors[0])
        hw = self.hardware_profile
        candidates = self.get_candidate_configs(op, data)
        return self.cost_model.select_best(op, data, hw, candidates)

    def log_result(self, op: str, tensor: torch.Tensor | CLBuffer, config: KernelConfig, actual_ms: float):
        """Log actual execution performance for online learning."""
        data = self.profile_data(tensor)
        hw = self.hardware_profile
        self.cost_model.log_timing(op, data, hw, config, actual_ms)


_global_dispatcher: DifferentialDispatcher | None = None

def get_dispatcher() -> DifferentialDispatcher:
    """Get the global dispatcher instance."""
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = DifferentialDispatcher()
    return _global_dispatcher
