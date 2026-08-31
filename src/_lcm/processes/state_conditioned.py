"""Direct-CDF transition rows for state-conditioned shock parameters.

The transition kernel is discretized on fixed common nodes, built once from the
process's own scalar `sigma`, while the per-category `sigma` enters only the transition
CDF. Each row is evaluated directly at the actual time-$t$ value.

Interpolating precomputed node rows instead is wrong by an amount that does not shrink
as the grid refines: for a `sigma` small relative to the cell width, essentially all of
the conditional mass sits inside a single cell, and a row read off the neighbouring
nodes and blended puts mass where the true kernel has none.

Supported families are the CDF-binned IID normal and Tauchen AR(1), whose transition
CDFs carry `sigma`. Rouwenhorst is excluded: its transition depends on `rho` only, so
fixing the nodes removes the sole `sigma` channel.

Evaluated at a node with the per-category `sigma` equal to the node-placing one, these
builders reproduce the `NormalIIDProcess(gauss_hermite=False)` and `TauchenAR1Process`
rows exactly.
"""

from collections.abc import Mapping
from typing import Literal

import jax.numpy as jnp
from jax.scipy.stats.norm import cdf

from _lcm.grids import DiscreteGrid
from _lcm.processes.base import StateConditioned
from lcm.typing import Float1D, ScalarFloat, ScalarInt

__all__ = [
    "Family",
    "StateConditioned",
    "conditioned_row",
    "gather_sigma",
    "iid_normal_row",
    "sigma_array_by_code",
    "tauchen_row",
]

_Scalar = float | int | ScalarFloat

# Process families whose transition CDF carries `sigma`. Rouwenhorst is absent: its
# transition is `rho`-only, so a fixed node grid leaves no channel for a
# state-conditioned `sigma`.
Family = Literal["iid_normal", "tauchen"]


def sigma_array_by_code(*, cond_grid: DiscreteGrid, by: Mapping[str, float]) -> Float1D:
    """Order the per-category `sigma` values by the categorical's integer **code**.

    Stacking by `Mapping` insertion order silently permutes categories whenever the code
    order differs from the insertion order; indexing the returned array by the
    conditioning state's code is therefore correct by construction. `by` must name
    exactly the categories of `cond_grid` — an extra key is a typo or a stale category,
    never a no-op.
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

    `nodes` are the fixed common nodes, placed by the process's own scalar `sigma`;
    `sigma` here is the innovation std the conditioning state selects, already gathered
    by code. `mu` is the process's fixed location (IID mean, AR(1) intercept) —
    dropping it misplaces the entire row. `rho` is required for Tauchen.
    """
    if family == "iid_normal":
        return iid_normal_row(nodes=nodes, mu=mu, sigma=sigma)
    if family == "tauchen":
        if rho is None:
            msg = "conditioned_row(family='tauchen') requires rho"
            raise ValueError(msg)
        return tauchen_row(
            nodes=nodes, rho=rho, sigma=sigma, from_value=from_value, mu=mu
        )
    msg = f"unsupported family {family!r} (v1: 'iid_normal' | 'tauchen')"
    raise ValueError(msg)


def gather_sigma(*, sigma_by_code: Float1D, code: int | ScalarInt) -> ScalarFloat:
    """Select the `sigma` the time-$t$ conditioning state selects, by its code."""
    return sigma_by_code[code]


def _row_from_edge_cdf(cdf_at_edges: Float1D) -> Float1D:
    """Assemble an `(n,)` probability row from CDF values at the `(n-1)` bin edges.

    First bin is the lower tail, last bin the upper tail, interior bins the CDF diffs —
    the same binning pylcm uses, so the row sums to one by construction.
    """
    first = cdf_at_edges[:1]
    interior = jnp.diff(cdf_at_edges)
    last = 1.0 - cdf_at_edges[-1:]
    return jnp.concatenate([first, interior, last])


def iid_normal_row(*, nodes: Float1D, mu: _Scalar, sigma: _Scalar) -> Float1D:
    r"""CDF-binned $N(\mu, \sigma_{s_t}^2)$ on the fixed `nodes`.

    IID: the row does not depend on the time-$t$ value. Binned on the midpoints of the
    fixed common nodes, never on nodes that move with the selected `sigma`.
    """
    edges = (nodes[:-1] + nodes[1:]) / 2.0
    return _row_from_edge_cdf(cdf((edges - mu) / sigma))


def tauchen_row(
    *,
    nodes: Float1D,
    rho: _Scalar,
    sigma: _Scalar,
    from_value: _Scalar,
    mu: _Scalar = 0.0,
) -> Float1D:
    r"""Conditional AR(1) row for $y_{t+1} = \mu + \rho y_t + \varepsilon_{t+1}$.

    With $\varepsilon_{t+1} \sim N(0, \sigma_{s_t}^2)$, evaluated directly at
    `from_value` — the time-$t$ value — rather than by interpolating node rows. The
    denominator is the innovation `sigma`, the conditional std of $y_{t+1} \mid y_t$,
    matching `TauchenAR1Process.compute_transition_probs`.

    `nodes` and `from_value` are in physical units: the axis
    `TauchenAR1Process.compute_gridpoints` returns, centred on $\mu/(1-\rho)$, so the
    conditional mean is $\mu + \rho y_t$. Stock pylcm builds the same row in demeaned
    coordinates, where the intercept vanishes. Here it does not, and dropping it
    misplaces every row unless $\mu = 0$.
    """
    edges = (nodes[:-1] + nodes[1:]) / 2.0
    return _row_from_edge_cdf(cdf((edges - mu - rho * from_value) / sigma))
