"""
Adaptive Precision Streaming (APS) — Dynamic mixed precision management and packing/unpacking utilities.
"""

from __future__ import annotations
import numpy as np
import pyopencl as cl
import torch

from torchcl.runtime.context import get_queue
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.kernels.registry import get_kernel_registry
from torchcl.api import _get_buf, _get_shape, _wrap_output, is_opencl_tensor


class PrecisionMap:
    """Holds precision layout metadata for a tensor."""

    def __init__(self, shape: tuple, base_precision: str = "float32") -> None:
        self.shape = shape
        self.base_precision = base_precision
        # Precision mapping: can specify block-level or tensor-level precision
        self.precision_tags: dict[int, str] = {0: base_precision}

    def set_precision_for_block(self, block_idx: int, prec: str):
        self.precision_tags[block_idx] = prec

    def get_precision_for_block(self, block_idx: int) -> str:
        return self.precision_tags.get(block_idx, self.base_precision)


class AdaptivePrecision:
    """Manages analysis of dynamic range and packing/unpacking of data on the GPU."""

    def __init__(self) -> None:
        self.registry = get_kernel_registry()
        self.pool = get_buffer_pool()

    def analyze_precision(self, cl_buf: CLBuffer, n: int) -> tuple[float, float, str]:
        """Analyze the range of values in a buffer and suggest the optimal precision.

        Returns:
            (min_val, max_val, suggested_precision)
        """
        queue = get_queue()

        # Host-side fallback for analysis (fast sampling)
        # In a real environment, we do a quick device-side min/max reduction or read a sample
        sample_size = min(256, n)
        sample = np.empty(sample_size, dtype=np.float32)
        cl.enqueue_copy(queue, sample, cl_buf.buffer)
        queue.finish()

        min_val = float(np.min(sample)) if len(sample) > 0 else 0.0
        max_val = float(np.max(sample)) if len(sample) > 0 else 0.0
        val_range = max_val - min_val

        # Select precision based on dynamic range
        if val_range < 2.0:
            suggested = "int8"
        elif val_range < 1000.0:
            suggested = "float16"
        else:
            suggested = "float32"

        return min_val, max_val, suggested

    def pack_to_fp16(self, input_buf: CLBuffer, n: int) -> CLBuffer:
        """Pack FP32 buffer to FP16 (half) buffer on GPU."""
        queue = get_queue()
        kernel = self.registry.get_kernel("precision.cl", "pack_fp32_to_fp16")

        # Output half elements use 2 bytes each
        output_buf = self.pool.allocate(n * 2, np.dtype(np.float16), input_buf.shape)

        global_size = (((n + 255) // 256) * 256,)
        local_size = (min(256, n),) if n >= 256 else (n,)

        kernel(
            queue, global_size, local_size,
            input_buf.buffer,
            output_buf.buffer,
            np.int32(n)
        )
        return output_buf

    def unpack_from_fp16(self, input_buf: CLBuffer, n: int) -> CLBuffer:
        """Unpack FP16 buffer back to FP32 buffer on GPU."""
        queue = get_queue()
        kernel = self.registry.get_kernel("precision.cl", "unpack_fp16_to_fp32")

        output_buf = self.pool.allocate(n * 4, np.dtype(np.float32), input_buf.shape)

        global_size = (((n + 255) // 256) * 256,)
        local_size = (min(256, n),) if n >= 256 else (n,)

        kernel(
            queue, global_size, local_size,
            input_buf.buffer,
            output_buf.buffer,
            np.int32(n)
        )
        return output_buf
