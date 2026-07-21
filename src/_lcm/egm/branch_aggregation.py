"""Outer branch aggregation: how keeper and adjuster values combine.

The nested solvers' historical behavior is a hard maximum,
`V = max(V_keeper, V_adjuster)`. When the model draws an i.i.d. fixed
adjustment cost `B * chi` with `chi ~ U[a, b]` that is *observed before the
branch choice*, enters utility only through the adjuster's fixed cost, and
moves neither the conditional optimal actions nor any state transition, the
shock can be integrated analytically instead of carried as a solve-state
grid: the branch comparison is `max(V_K, V_A - B*chi)`, monotone in `chi`,
so a cutoff `chi* = (V_A - V_K) / B` splits the support and both the
adjustment probability and the expected value have closed forms.

This module holds the aggregation *configurations* (re-exported through
`lcm.branch_aggregation`) and the closed-form kernel; validating that a
model's shock actually satisfies the observability/exclusion requirements is
the model-wiring layer's job.

Closed forms, with `Delta = V_A - V_K` and `t = Delta / B`:

    p_A = clip((t - a) / (b - a), 0, 1)

    E[max(V_K, V_A - B*chi)] =
        V_K                                 if t <= a
        V_K + B * (t - a)^2 / (2 (b - a))   if a < t < b
        V_A - B * (a + b) / 2               if t >= b

`B = 0` degenerates to the deterministic maximum with the keeper winning
exact ties (the finite fold's tie rule).
"""

from abc import ABC
from dataclasses import dataclass

import jax.numpy as jnp

from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND, FunctionName, StateName


class OuterBranchAggregator(ABC):  # noqa: B024
    """Configuration for combining keeper and adjuster branch values.

    Abstract marker base: a nested solver accepts any concrete aggregator
    and dispatches polymorphically. Pure configuration — the numerics live
    in the module-level kernels.
    """


@dataclass(frozen=True, kw_only=True)
class DeterministicOuterMaximum(OuterBranchAggregator):
    """The historical behavior: `V = max(V_keeper, V_adjuster)`, keeper
    winning exact ties."""


@dataclass(frozen=True, kw_only=True)
class UniformObservedFixedCost(OuterBranchAggregator):
    """Analytically integrated uniform fixed adjustment cost.

    The shock must be observed before the branch choice, enter only the
    adjuster's fixed cost (scaled by `scale_function`), leave the
    conditional optimal actions unchanged, and touch no state transition
    except through the branch choice. Those requirements are validated at
    model wiring time.
    """

    shock_name: StateName
    """The model's i.i.d. uniform shock this aggregation integrates out."""

    scale_function: FunctionName
    """Model function returning the cost scale `B >= 0` per state cell."""

    lower: float = 0.0
    """Lower support edge `a` of the uniform shock."""

    upper: float = 1.0
    """Upper support edge `b` of the uniform shock; must exceed `lower`."""

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            msg = (
                "UniformObservedFixedCost needs positive support width, got "
                f"[{self.lower}, {self.upper}]."
            )
            raise RegimeInitializationError(msg)


@dataclass(frozen=True, kw_only=True)
class BranchAggregateResult:
    """Pointwise outcome of aggregating the keeper/adjuster comparison."""

    expected_value: FloatND
    """`E[max(V_K, V_A - B*chi)]` per cell."""

    adjustment_probability: FloatND
    """`P(adjust)` per cell — an analytic moment for the inference layer."""

    no_adjustment_probability: FloatND
    """`1 - adjustment_probability` (exactly complementary)."""

    cutoff: FloatND
    """The indifference draw `chi* = (V_A - V_K) / B`; `-inf`/`+inf` where
    one branch dominates for every draw, NaN where both are infeasible."""


def aggregate_uniform_observed_fixed_cost(
    *,
    keeper_value: FloatND,
    adjuster_value: FloatND,
    scale: FloatND,
    lower: float,
    upper: float,
) -> BranchAggregateResult:
    """Closed-form aggregation of `max(V_K, V_A - B*chi)`, `chi ~ U[a, b]`.

    Args:
        keeper_value: `V_K` per cell; nonfinite marks an infeasible branch.
        adjuster_value: `V_A` per cell (gross of the fixed cost); nonfinite
            marks an infeasible branch.
        scale: Cost scale `B >= 0` per cell (broadcastable).
        lower: Support edge `a`.
        upper: Support edge `b > a`.

    Returns:
        Expected value, branch probabilities, and the cutoff draw. Where
        only one branch is feasible the result degenerates to it (an
        infeasible adjuster adjusts with probability 0 and vice versa);
        where neither is feasible the expected value is `-inf` with
        probability 0 of adjusting (nothing to adjust to). `B = 0`
        reproduces the deterministic maximum with keeper-wins-ties.

    """
    keeper_value = jnp.asarray(keeper_value)
    adjuster_value, scale = jnp.broadcast_arrays(
        jnp.asarray(adjuster_value), jnp.asarray(scale)
    )
    keeper_ok = jnp.isfinite(keeper_value)
    adjuster_ok = jnp.isfinite(adjuster_value)
    width = upper - lower

    delta = adjuster_value - keeper_value
    positive_scale = scale > 0.0
    safe_scale = jnp.where(positive_scale, scale, 1.0)
    cutoff = jnp.where(positive_scale, delta / safe_scale, jnp.inf * jnp.sign(delta))

    probability = jnp.clip((cutoff - lower) / width, 0.0, 1.0)
    interior = (cutoff > lower) & (cutoff < upper)
    expected = jnp.where(
        interior,
        keeper_value + safe_scale * (cutoff - lower) ** 2 / (2.0 * width),
        jnp.where(
            cutoff <= lower,
            keeper_value,
            adjuster_value - scale * (lower + upper) / 2.0,
        ),
    )

    # Feasibility degeneracies: a single feasible branch is taken outright;
    # no feasible branch keeps the fold's -inf vocabulary. `B = 0` with
    # Delta = 0 has cutoff = inf * 0 = NaN; the keeper wins that tie.
    tie_at_zero_scale = ~positive_scale & (delta == 0.0)
    expected = jnp.where(tie_at_zero_scale, keeper_value, expected)
    probability = jnp.where(tie_at_zero_scale, 0.0, probability)

    expected = jnp.where(
        keeper_ok & ~adjuster_ok,
        keeper_value,
        jnp.where(
            ~keeper_ok & adjuster_ok,
            adjuster_value - scale * (lower + upper) / 2.0,
            jnp.where(~keeper_ok & ~adjuster_ok, -jnp.inf, expected),
        ),
    )
    probability = jnp.where(
        keeper_ok & ~adjuster_ok,
        0.0,
        jnp.where(
            ~keeper_ok & adjuster_ok,
            1.0,
            jnp.where(~keeper_ok & ~adjuster_ok, 0.0, probability),
        ),
    )
    cutoff = jnp.where(
        keeper_ok & ~adjuster_ok,
        -jnp.inf,
        jnp.where(
            ~keeper_ok & adjuster_ok,
            jnp.inf,
            jnp.where(~keeper_ok & ~adjuster_ok, jnp.nan, cutoff),
        ),
    )

    return BranchAggregateResult(
        expected_value=expected,
        adjustment_probability=probability,
        no_adjustment_probability=1.0 - probability,
        cutoff=cutoff,
    )
