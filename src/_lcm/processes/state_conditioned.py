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

from collections.abc import Mapping
from typing import Literal

import jax.numpy as jnp
from jax.scipy.stats.norm import cdf

from _lcm.grids import DiscreteGrid
from _lcm.processes.base import StateConditioned
from lcm.typing import Float1D, ScalarFloat, ScalarInt

__all__ = [
    "StateConditioned",
    "conditioned_row",
    "gather_sigma",
    "iid_normal_row",
    "sigma_array_by_code",
    "tauchen_row",
]

_Scalar = float | int | ScalarFloat

# Process families whose transition CDF carries `sigma` (v1-supported). Rouwenhorst is
# intentionally absent: its transition is `rho`-only, so a fixed node grid leaves no
# channel for a state-conditioned `sigma` (audit F2).
Family = Literal["iid_normal", "tauchen"]


def sigma_array_by_code(cond_grid: DiscreteGrid, by: Mapping[str, float]) -> Float1D:
    """Order the per-category `sigma` values by the categorical's integer **code**.

    Stacking by `Mapping` insertion order silently permutes regimes when the code order
    differs (audit F5/RT5); indexing the returned array by the conditioning state's code
    is therefore correct by construction. `by` must name exactly the categories of
    `cond_grid` — an extra key is a typo or a stale category, never a no-op.
    """
    cats, codes = cond_grid.categories, cond_grid.codes
    missing = set(cats) - set(by)
    if missing:
        msg = f"StateConditioned.by is missing categories {sorted(missing)}"
        raise ValueError(msg)
    extra = set(by) - set(cats)
    if extra:
        msg = (
            f"StateConditioned.by has categories {sorted(extra)} that are not in the "
            f"conditioning grid {sorted(cats)}"
        )
        raise ValueError(msg)
    # Runtime indexing (`gather_sigma`) uses the conditioning state's code directly,
    # which is safe because `@categorical` assigns contiguous 0..n-1 codes and
    # `DiscreteGrid` accepts only such classes — so position == code here.
    ordered = sorted(zip(codes, cats, strict=True))  # by integer code
    return jnp.asarray([by[name] for _code, name in ordered])


def conditioned_row(
    *,
    family: Family,
    nodes: Float1D,
    sigma: _Scalar,
    from_value: _Scalar,
    mu: _Scalar,
    rho: _Scalar | None = None,
) -> Float1D:
    """Dispatch to the direct-CDF row builder for the given process `family`.

    `nodes` are the FIXED common nodes (from `grid_sigma`); `sigma` is the current
    regime's innovation std (already gathered by code). `mu` is the process's fixed
    location (IID mean / AR(1) intercept) — dropping it misplaces the entire row
    (code-review F2). `rho` is required for Tauchen.
    """
    if family == "iid_normal":
        return iid_normal_row(nodes, mu=mu, sigma=sigma)
    if family == "tauchen":
        if rho is None:
            msg = "conditioned_row(family='tauchen') requires rho"
            raise ValueError(msg)
        return tauchen_row(nodes, rho=rho, sigma=sigma, from_value=from_value, mu=mu)
    msg = f"unsupported family {family!r} (v1: 'iid_normal' | 'tauchen')"
    raise ValueError(msg)


def gather_sigma(sigma_by_code: Float1D, code: int | ScalarInt) -> ScalarFloat:
    """Select the current regime's `sigma` from the code-ordered array (audit F5)."""
    return sigma_by_code[code]


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
    mu: _Scalar = 0.0,
) -> Float1D:
    """Conditional AR(1) row for ``y' = mu + rho*y + eps``, ``eps ~ N(0, sigma**2)``.

    Evaluated DIRECTLY at ``from_value`` (audit F1) rather than by interpolating node
    rows. The denominator is the innovation ``sigma`` (conditional std of ``y' | y``),
    matching ``TauchenAR1Process.compute_transition_probs``.

    ``nodes`` and ``from_value`` are in PHYSICAL units — i.e. the axis
    ``TauchenAR1Process.compute_gridpoints`` returns, centred on ``mu/(1-rho)`` —
    so the conditional mean is ``mu + rho*from_value``. Stock pylcm builds the same
    row in demeaned coordinates, where the intercept vanishes; here it does not, and
    omitting it misplaces every row unless ``mu == 0`` (code-review F2).
    """
    edges = (nodes[:-1] + nodes[1:]) / 2.0
    return _row_from_edge_cdf(cdf((edges - mu - rho * from_value) / sigma))
