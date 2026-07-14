"""Direct-CDF transition rows for state-conditioned shock parameters (v1).

The transition kernel of a continuous stochastic process is discretized on a set of
**fixed common nodes** (built once from ``grid_sigma``), while the regime-dependent
``sigma`` enters only the transition *CDF*. The row is evaluated **directly at the
actual from-value** — never by interpolating precomputed node rows, which is O(1) wrong
for a low ``sigma`` on wide cells (pro-comp-method audit 2026-07-14, finding F1).

Supported families in v1: CDF-binned IID normal and Tauchen AR(1) (``sigma`` sits in
their transition CDF). Rouwenhorst is intentionally excluded: its transition depends on
``rho`` only, so fixing the nodes removes the sole ``sigma`` channel (audit F2).

These builders reduce exactly to ``NormalIIDProcess(gauss_hermite=False)`` /
``TauchenAR1Process`` rows when evaluated at a node with ``sigma == grid_sigma``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats.norm import cdf

from lcm.typing import Float1D, ScalarFloat

_Scalar = float | int | ScalarFloat


def _row_from_edge_cdf(cdf_at_edges: Float1D) -> Float1D:
    """Assemble an ``(n,)`` probability row from CDF values at the ``(n-1)`` bin edges.

    First bin is the lower tail, last bin the upper tail, interior bins the CDF diffs —
    the same binning pylcm uses, so the row sums to one by construction.
    """
    first = cdf_at_edges[:1]
    interior = jnp.diff(cdf_at_edges)
    last = 1.0 - cdf_at_edges[-1:]
    return jnp.concatenate([first, interior, last])


def iid_normal_row(nodes: Float1D, mu: _Scalar, sigma: _Scalar) -> Float1D:
    """CDF-binned ``N(mu, sigma**2)`` on FIXED ``nodes``.

    IID: the row does not depend on the current value. Binning on the midpoints of the
    fixed common nodes (never on moving ``sigma``-specific nodes — audit F6/RT6).
    """
    edges = (nodes[:-1] + nodes[1:]) / 2.0
    return _row_from_edge_cdf(cdf((edges - mu) / sigma))


def tauchen_row(
    nodes: Float1D,
    rho: _Scalar,
    sigma: _Scalar,
    from_value: _Scalar,
) -> Float1D:
    """Conditional AR(1) row on FIXED ``nodes``, innovation ``~ N(0, sigma**2)``.

    Evaluated DIRECTLY at ``from_value`` (audit F1) rather than by interpolating node
    rows. The denominator is the innovation ``sigma`` (conditional std of ``y' | y``),
    matching ``TauchenAR1Process.compute_transition_probs``.
    """
    edges = (nodes[:-1] + nodes[1:]) / 2.0
    return _row_from_edge_cdf(cdf((edges - rho * from_value) / sigma))
