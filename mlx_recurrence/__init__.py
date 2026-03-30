"""mlx_recurrence — Fused Metal kernels for linear recurrence on Apple Silicon."""

from .ssm_scan import (
    selective_scan_metal,
    selective_scan_chunked,
    _ssm_forward_kernel,
    _ssm_backward_chunked,
)
from .gla_scan import (
    gla_scan_metal,
    gla_scan_chunked,
    _gla_forward_kernel,
    _gla_backward_chunked,
)

__all__ = [
    "selective_scan_metal",
    "selective_scan_chunked",
    "gla_scan_metal",
    "gla_scan_chunked",
    # lower-level kernels (public for testing / advanced use)
    "_ssm_forward_kernel",
    "_ssm_backward_chunked",
    "_gla_forward_kernel",
    "_gla_backward_chunked",
]

__version__ = "0.1.0"
