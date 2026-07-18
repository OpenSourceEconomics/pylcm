"""Vectorized golden-section maximization with explicit endpoint safeguards.

The continuous-outer solver refines an outer-action optimum *inside a bracket
identified on a global candidate mesh* — never on the full domain, because no
global unimodality is assumed. This module supplies the bracket-local
primitive: a fixed-iteration, array-valued golden-section maximizer that

- runs under `jax.lax.fori_loop` with a static trip count (JIT/vmap/scan
  compatible; one objective evaluation per iteration via the classic
  mirror-point reuse);
- never branches in Python on traced values;
- **evaluates both endpoints explicitly** and returns the best of
  {endpoints, final interior points} — golden section alone can only converge
  *toward* a boundary, never onto it, so a boundary maximum would otherwise be
  systematically missed;
- breaks exact ties toward the **smaller** abscissa (the canonical outer
  tie rule: choose the smaller outer action), deterministically;
- treats NaN objective values as `-inf` (an infeasible probe must not poison
  the bracket — the same `fmax` semantics as the finite outer collapse);
- accepts degenerate brackets (`lower == upper`) and per-cell invalid masks.

Maximization only: the solver's objectives are values. Minimize by negating.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt

import jax
import jax.numpy as jnp
from jaxtyping import Int, Scalar

from lcm.typing import BoolND, FloatND

# `jax.lax.fori_loop` materializes its counter at JAX's default int dtype
# (weak int64 under x64), not pylcm's canonical int32 — so the loop-body index
# needs a dtype-agnostic scalar-int hint rather than `ScalarInt`.
type _LoopIndex = Int[Scalar, ""]

_INV_PHI = (sqrt(5.0) - 1.0) / 2.0  # 1/phi ~ 0.618...
_INV_PHI_SQ = (3.0 - sqrt(5.0)) / 2.0  # 1/phi^2 ~ 0.382...


@dataclass(frozen=True, kw_only=True)
class GoldenSectionResult:
    """Outcome of a vectorized bracket-local golden-section maximization."""

    x: FloatND
    """The selected abscissa per cell — the best of the endpoint and final
    interior candidates, ties toward the smaller abscissa."""

    value: FloatND
    """The (sanitized) objective at `x`; `-inf` on invalid cells."""

    lower: FloatND
    """Final bracket lower edge per cell (diagnostic: bracket width)."""

    upper: FloatND
    """Final bracket upper edge per cell (diagnostic: bracket width)."""

    iterations: int
    """The static iteration count the search ran with."""

    converged: BoolND
    """Whether the final bracket width met `width_tolerance` (always `True`
    on valid cells when no tolerance was requested: with a fixed budget the
    deterministic shrink *is* the contract)."""

    valid: BoolND
    """Which cells carried a real bracket; invalid cells report `-inf`."""


def maximize_golden_section(
    objective: Callable[[FloatND], FloatND],
    *,
    lower: FloatND,
    upper: FloatND,
    iterations: int,
    valid: BoolND | None = None,
    width_tolerance: float | None = None,
) -> GoldenSectionResult:
    """Maximize a vectorized scalar objective inside per-cell brackets.

    Args:
        objective: Vectorized map from an abscissa array (one probe per cell)
            to objective values of the same shape. Must accept any abscissa
            inside `[lower, upper]`; NaN results are treated as `-inf`.
        lower: Per-cell bracket lower edges.
        upper: Per-cell bracket upper edges; `upper >= lower` where valid.
        iterations: Static golden-section iteration count (`>= 0`). The final
            interior bracket width is `(upper - lower) * INV_PHI**iterations`.
        valid: Optional per-cell mask; cells `False` are not searched and
            report `x = lower`, `value = -inf`, `valid = False`.
        width_tolerance: Optional absolute width the final bracket must meet
            for `converged`; `None` means the fixed budget is the contract
            and `converged == valid`.

    Returns:
        The per-cell maximizer candidates and bracket diagnostics.

    Raises:
        ValueError: If `iterations` is negative.

    """
    if iterations < 0:
        msg = f"iterations must be >= 0, got {iterations}"
        raise ValueError(msg)
    lower_arr = jnp.asarray(lower)
    upper_arr = jnp.asarray(upper)
    lower_arr, upper_arr = jnp.broadcast_arrays(lower_arr, upper_arr)
    is_valid = (
        jnp.broadcast_to(jnp.asarray(valid), lower_arr.shape)
        if valid is not None
        else jnp.ones(lower_arr.shape, dtype=bool)
    )
    is_valid = (
        is_valid
        & jnp.isfinite(lower_arr)
        & jnp.isfinite(upper_arr)
        & (upper_arr >= lower_arr)
    )
    # Invalid cells still flow through the arithmetic (no Python branching on
    # traced data); their probes are pinned to `lower` so the objective is
    # never asked outside a real bracket, and their outputs are masked at the
    # end.
    safe_lower = jnp.where(is_valid, lower_arr, 0.0)
    safe_upper = jnp.where(is_valid, upper_arr, 0.0)

    def probe(x: FloatND) -> FloatND:
        raw = objective(x)
        return jnp.where(jnp.isnan(raw), -jnp.inf, raw)

    width = safe_upper - safe_lower
    x1 = safe_lower + _INV_PHI_SQ * width
    x2 = safe_lower + _INV_PHI * width
    carry0 = (safe_lower, safe_upper, x1, probe(x1), x2, probe(x2))

    def body(
        _i: _LoopIndex,
        carry: tuple[FloatND, FloatND, FloatND, FloatND, FloatND, FloatND],
    ) -> tuple[FloatND, FloatND, FloatND, FloatND, FloatND, FloatND]:
        a, b, xa, fa, xb, fb = carry
        # Keep the sub-bracket containing the better interior point; exact
        # interior ties keep the LEFT sub-bracket (smaller abscissae).
        take_left = fa >= fb
        a_next = jnp.where(take_left, a, xa)
        b_next = jnp.where(take_left, xb, b)
        # The surviving interior point and its (reused) evaluation ...
        x_kept = jnp.where(take_left, xa, xb)
        f_kept = jnp.where(take_left, fa, fb)
        # ... and its golden mirror in the new bracket — one new probe per
        # iteration, exactly the classic bookkeeping, elementwise.
        x_new = a_next + b_next - x_kept
        f_new = probe(x_new)
        left_is_kept = x_kept <= x_new
        xa_next = jnp.where(left_is_kept, x_kept, x_new)
        fa_next = jnp.where(left_is_kept, f_kept, f_new)
        xb_next = jnp.where(left_is_kept, x_new, x_kept)
        fb_next = jnp.where(left_is_kept, f_new, f_kept)
        return (a_next, b_next, xa_next, fa_next, xb_next, fb_next)

    a, b, xa, fa, xb, fb = jax.lax.fori_loop(0, iterations, body, carry0)

    # Explicit endpoint safeguard, then select in increasing-abscissa order
    # with a strict `>` so exact ties resolve to the smaller abscissa.
    best_x = safe_lower
    best_f = probe(safe_lower)
    for cand_x, cand_f in (
        (xa, fa),
        (xb, fb),
        (safe_upper, probe(safe_upper)),
    ):
        take = cand_f > best_f
        best_x = jnp.where(take, cand_x, best_x)
        best_f = jnp.where(take, cand_f, best_f)

    if width_tolerance is None:
        converged = is_valid
    else:
        converged = is_valid & ((b - a) <= width_tolerance)

    return GoldenSectionResult(
        x=jnp.where(is_valid, best_x, lower_arr),
        value=jnp.where(is_valid, best_f, -jnp.inf),
        lower=jnp.where(is_valid, a, lower_arr),
        upper=jnp.where(is_valid, b, lower_arr),
        iterations=iterations,
        converged=converged,
        valid=is_valid,
    )
