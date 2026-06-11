"""mlx_recurrence — A plug-in framework for linear-recurrence Metal kernels
on Apple Silicon ("flash-linear-attention for MLX").

Each kernel is a self-contained plug-in built on a shared chassis
(:mod:`mlx_recurrence._chassis`) that supplies the segment-checkpoint +
recompute backward pattern, shape validation, and a parity-test helper. The
Metal source for each recurrence stays in its own module, readable per-kernel.

v2 kernels (checkpoint + recompute, fused simd reductions, chunked-prefill
final-state variants):
    ssd     — Mamba-2-style head-wise SSD selective scan
    gla     — Gated Linear Attention recurrence
    rglru   — RG-LRU diagonal recurrence (Griffin / RecurrentGemma)
    rotlru  — rotational LRU: complex-diagonal scan over (u, w) pairs

The original v0.1 token-loop kernels remain available under
``mlx_recurrence.legacy`` and are re-exported at top level for backwards
compatibility (``selective_scan_metal``, ``gla_scan_metal``, ...).
"""

# --- v2 chassis-based kernels ---------------------------------------------
from .ssd import (
    ssd_scan,
    ssd_scan_with_state,
    ssd_scan_reference,
)
from .gla import (
    gla_scan,
    gla_scan_with_state,
    gla_scan_reference,
)
from .rglru import (
    rglru_scan,
    rglru_scan_with_state,
    rglru_scan_reference,
)
from .rotlru import (
    rotlru_scan,
    rotlru_scan_with_state,
    rotlru_scan_reference,
)

# --- shared chassis (public for building new plug-in kernels) -------------
from ._chassis import (
    DEFAULT_SEG,
    get_or_build_kernel,
    check_segment_shape,
    parity_check,
)

# --- legacy v0.1 kernels (backwards compatibility) ------------------------
from . import legacy
from .legacy import (
    selective_scan_metal,
    selective_scan_chunked,
    gla_scan_metal,
    gla_scan_chunked,
)

__all__ = [
    # v2 SSD
    "ssd_scan",
    "ssd_scan_with_state",
    "ssd_scan_reference",
    # v2 GLA
    "gla_scan",
    "gla_scan_with_state",
    "gla_scan_reference",
    # v2 RG-LRU
    "rglru_scan",
    "rglru_scan_with_state",
    "rglru_scan_reference",
    # v2 rotational LRU
    "rotlru_scan",
    "rotlru_scan_with_state",
    "rotlru_scan_reference",
    # chassis
    "DEFAULT_SEG",
    "get_or_build_kernel",
    "check_segment_shape",
    "parity_check",
    # legacy subpackage + re-exports
    "legacy",
    "selective_scan_metal",
    "selective_scan_chunked",
    "gla_scan_metal",
    "gla_scan_chunked",
]

__version__ = "0.3.0"
