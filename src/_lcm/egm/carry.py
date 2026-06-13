"""Cross-period data channel of the DC-EGM solver.

Backward induction with DC-EGM threads more than the value-function array
between adjacent periods: the parent's Euler inversion needs the child's
optimal policy, value, and marginal utility on the child's endogenous
(resources-space) grid. `EGMCarry` bundles these rows; the solve loop rolls a
`next_regime_to_egm_carry` mapping alongside `next_regime_to_V_arr`, with one
entry per carry-producing regime (DC-EGM regimes and terminal regimes a
DC-EGM regime can target).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from _lcm.dtypes import canonical_float_dtype
from lcm.typing import FloatND, ScalarFloat


@dataclass(frozen=True, kw_only=True)
class EGMCarry:
    """Per-regime EGM solution rows threaded between adjacent periods.

    All rows share the trailing grid axis of static, per-regime length so the
    carry has a period-invariant pytree shape (periods sharing a compiled
    program never trigger retracing). Regimes with combo dimensions carry one
    row per combo: leading axes are the regime's discrete states (in V state
    order; process states are node-valued discrete dimensions), then its
    passive continuous states (in V continuous-state order, one node per
    combo), then its discrete actions. Every array is pinned to the canonical
    float dtype.
    """

    endog_grid: FloatND
    """Endogenous grid in resources space, NaN-padded in the tail.

    Weakly ascending per row: envelope kink abscissae appear twice, carrying
    the left- and right-extrapolated policy values.
    """

    policy: FloatND
    """Optimal continuous action at `endog_grid` (NaN on padding slots)."""

    value: FloatND
    """Choice-specific value at `endog_grid`; `-inf` marks infeasible rows."""

    marginal_utility: FloatND
    """Marginal value of resources $\\partial v / \\partial R$ at `endog_grid`.

    Exactly `0.0` (never NaN) wherever `value` is `-inf`: infeasible rows get
    zero choice probability, and `0 \\cdot \\mu` must stay finite in the
    parent's probability-weighted expectation.
    """

    taste_shock_scale: ScalarFloat
    """EV1 taste-shock scale of the regime as a 0-d array; `0.0` = hard max."""


# Pytree registration with an `__init__`-bypassing unflatten: JAX's transform
# and AOT-lowering machinery reconstructs pytrees with non-array leaves
# (`ArgInfo`, tracers, `None`), which the runtime-checked constructor would
# reject. Flatten order matches field declaration order.
_EGM_CARRY_FIELDS = (
    "endog_grid",
    "policy",
    "value",
    "marginal_utility",
    "taste_shock_scale",
)


def _flatten_egm_carry(carry: EGMCarry) -> tuple[tuple[Any, ...], None]:
    return tuple(getattr(carry, name) for name in _EGM_CARRY_FIELDS), None


def _unflatten_egm_carry(_aux: None, children: Sequence[Any]) -> EGMCarry:
    carry = object.__new__(EGMCarry)
    for name, child in zip(_EGM_CARRY_FIELDS, children, strict=True):
        object.__setattr__(carry, name, child)
    return carry


jax.tree_util.register_pytree_node(EGMCarry, _flatten_egm_carry, _unflatten_egm_carry)


def build_template_egm_carry(
    *,
    n_rows: int,
    leading_shape: tuple[int, ...] = (),
) -> EGMCarry:
    """Build a benign all-finite carry template with `n_rows` grid slots.

    Used to initialize the rolling `next_regime_to_egm_carry` mapping before
    a regime has been solved, and as the lowering argument when AOT-compiling
    EGM kernels. The endogenous grid is strictly ascending and every row is
    finite, so a parent kernel evaluated against the template produces finite
    (probability-zeroed) contributions rather than NaN.

    Args:
        n_rows: Static length of the carry rows.
        leading_shape: Sizes of the regime's combo dimensions (discrete
            states, then passive states, then discrete actions); empty for
            regimes without combo dimensions.

    Returns:
        Carry with an ascending unit-interval grid and all-zero policy,
        value, and marginal-utility rows, broadcast over `leading_shape`.

    """
    dtype = canonical_float_dtype()
    shape = (*leading_shape, n_rows)
    return EGMCarry(
        endog_grid=jnp.broadcast_to(jnp.linspace(0.0, 1.0, n_rows, dtype=dtype), shape),
        policy=jnp.zeros(shape, dtype=dtype),
        value=jnp.zeros(shape, dtype=dtype),
        marginal_utility=jnp.zeros(shape, dtype=dtype),
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
    )
