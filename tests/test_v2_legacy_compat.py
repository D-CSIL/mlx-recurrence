"""Backwards-compatibility smoke tests for the legacy v0.1 API.

Verifies the top-level re-exports and the ``mlx_recurrence.legacy``
subpackage still import and run on tiny shapes.
"""

import mlx.core as mx


def test_toplevel_legacy_reexports_importable():
    from mlx_recurrence import (
        selective_scan_metal,
        selective_scan_chunked,
        gla_scan_metal,
        gla_scan_chunked,
    )
    assert callable(selective_scan_metal)
    assert callable(selective_scan_chunked)
    assert callable(gla_scan_metal)
    assert callable(gla_scan_chunked)


def test_legacy_subpackage_importable():
    from mlx_recurrence import legacy
    from mlx_recurrence.legacy import (
        selective_scan_metal,
        gla_scan_metal,
    )
    assert callable(legacy.selective_scan_metal)
    assert callable(selective_scan_metal)
    assert callable(gla_scan_metal)


def test_legacy_ssm_runs_tiny():
    from mlx_recurrence import selective_scan_metal

    B, L, inner, N = 1, 16, 8, 4
    mx.random.seed(0)
    u     = mx.random.normal((B, L, inner)) * 0.5
    delta = mx.abs(mx.random.normal((B, L, inner))) * 0.05 + 0.001
    B_in  = mx.random.normal((B, L, N)) * 0.5
    C_in  = mx.random.normal((B, L, N)) * 0.5
    A_neg = -mx.abs(mx.random.normal((inner, N))) - 0.1

    y = selective_scan_metal(u, delta, B_in, C_in, A_neg)
    mx.eval(y)
    assert y.shape == (B, L, inner)


def test_legacy_gla_runs_tiny():
    from mlx_recurrence import gla_scan_metal

    B, L, H, Dh = 1, 16, 2, 8
    mx.random.seed(0)
    q     = mx.random.normal((B, L, H, Dh)) * (Dh ** -0.5)
    k     = mx.random.normal((B, L, H, Dh)) * 0.5
    v     = mx.random.normal((B, L, H, Dh)) * 0.5
    gates = mx.sigmoid(mx.random.normal((B, L, H)) + 2.0)

    y = gla_scan_metal(q, k, v, gates)
    mx.eval(y)
    assert y.shape == (B, L, H, Dh)


def test_version_matches_pyproject():
    """__version__ must stay in sync with pyproject.toml (single-source-of-
    truth check — catches one-sided release bumps)."""
    import re
    from pathlib import Path

    import mlx_recurrence

    pyproject = Path(mlx_recurrence.__file__).parent.parent / "pyproject.toml"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert m, "version not found in pyproject.toml"
    assert mlx_recurrence.__version__ == m.group(1)
