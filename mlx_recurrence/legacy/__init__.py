"""mlx_recurrence.legacy — v1 token-loop Metal kernels.

These are the original (v0.1) kernels: a Metal forward that materialises
the full per-timestep state tensor h_all and a backward that reads it back.
They are kept for backwards compatibility and as a simple, readable
reference. New work should prefer the v2 chassis-based kernels
(``mlx_recurrence.ssd``, ``mlx_recurrence.gla``, ``mlx_recurrence.rglru``),
which use segment checkpointing + recompute to slash DRAM traffic.

Public API (unchanged from v0.1):
    selective_scan_metal, selective_scan_chunked
    gla_scan_metal, gla_scan_chunked
"""

from .ssm_scan import (
    selective_scan_metal,
    selective_scan_chunked,
    _ssm_forward_kernel,
    _ssm_backward_chunked,
    _ssm_backward_metal,
)
from .gla_scan import (
    gla_scan_metal,
    gla_scan_chunked,
    _gla_forward_kernel,
    _gla_backward_chunked,
    _gla_backward_metal,
)

__all__ = [
    "selective_scan_metal",
    "selective_scan_chunked",
    "gla_scan_metal",
    "gla_scan_chunked",
    "_ssm_forward_kernel",
    "_ssm_backward_chunked",
    "_ssm_backward_metal",
    "_gla_forward_kernel",
    "_gla_backward_chunked",
    "_gla_backward_metal",
]
