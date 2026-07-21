"""Continuous collapse of an outer candidate bank into the keeper.

The finite collapse (`_collapse_finite_candidate_bank` in the solver layer)
folds exact candidates pointwise; this module is its continuous counterpart:
per state cell it runs the globally safeguarded continuous argmax over the
outer axis of the bank's stacked surfaces (interpolant between exact nodes,
golden refinement inside mesh-identified brackets, exact nodes always
competing), then folds the continuous adjuster optimum into the keeper with
the *same* deterministic semantics as the finite fold:

- the keeper wins exact ties (strict `>`);
- a NaN-dead keeper cell is taken over by any non-NaN adjuster read;
- an all-NaN candidate column leaves the keeper untouched (aligned padding
  rides through), while an all-`-inf` column (feasibly-typed infeasibility)
  competes as `-inf` exactly as it does finitely.

Per the envelope theorem, the liquid marginal published at the selected
outer action is the *conditional* liquid marginal interpolated at that
action — for an interior unique optimum the derivative of the optimizing
outer action does not enter. At bounds and ties that reasoning degrades;
the published diagnostics expose how much mass sits there.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp

from _lcm.egm.branch_aggregation import aggregate_uniform_observed_fixed_cost
from _lcm.egm.carry import EGMCarry
from _lcm.egm.outer_candidates import OuterCandidateBank
from _lcm.egm.outer_interpolation import LocalCubicOuterInterpolant
from _lcm.egm.outer_refinement import (
    SafeguardedSearchResult,
    safeguarded_continuous_argmax,
)
from _lcm.egm.outer_search import AdaptiveOuterMesh
from lcm.typing import FloatND

_INTERPOLANT = LocalCubicOuterInterpolant()


@dataclass(frozen=True, kw_only=True)
class ContinuousCollapse:
    """Continuous collapse output — assembled into a `KernelResult` by the
    solver layer (the `egm` layer never imports `solution`)."""

    V_arr: FloatND
    """The collapsed value array, keeper folded with the continuous
    adjuster optimum."""

    carry: EGMCarry
    """The collapsed carry: off-grid outer value and conditional liquid
    marginal at the selected outer action, keeper-folded pointwise."""

    value_search: SafeguardedSearchResult
    """The value-surface search outcome (bounds, margins, bracket widths —
    the solver layer's diagnostic ingredients)."""

    keeper_adjuster_margin: FloatND
    """Smallest finite `|V_keeper - V_adjuster|` over cells (0-d)."""

    best_second_best_margin: FloatND
    """Smallest finite best/second-best margin over cells (0-d)."""

    adjustment_probability: FloatND | None = None
    """Analytic per-cell probability of the adjuster branch under a uniform
    observed fixed cost, or `None` under the deterministic maximum."""


def collapse_continuous_candidate_bank(
    *,
    keeper_v_arr: FloatND,
    keeper_carry: EGMCarry,
    bank: OuterCandidateBank,
    config: AdaptiveOuterMesh,
    fixed_cost_scale: FloatND | None = None,
    fixed_cost_support: tuple[float, float] | None = None,
) -> ContinuousCollapse:
    """Collapse the bank into the keeper via the continuous outer optimum.

    Args:
        keeper_v_arr: The keeper's exact value array.
        keeper_carry: The keeper's exact carry rows.
        bank: Exact adjuster solves stacked on the shared (refined) mesh.
        config: The `AdaptiveOuterMesh` strategy (refiner budgets).
        fixed_cost_scale: Scale `B` of a uniform observed fixed adjustment
            cost, integrated analytically into the keeper/adjuster fold
            (`None` keeps the deterministic maximum). Must broadcast against
            both the value array and the carry rows — in the supported scope
            it is a 0-d per-period scalar.
        fixed_cost_support: The cost draw's support `(lower, upper)`;
            required with `fixed_cost_scale`.

    Returns:
        The collapsed surfaces plus the search diagnostics' ingredients.

    """
    value_search = _continuous_outer_argmax(
        nodes=bank.outer_nodes, stacked=bank.V_arr, config=config
    )
    if fixed_cost_scale is None:
        V_arr = _fold_continuous(
            running=keeper_v_arr,
            stacked=bank.V_arr,
            search=value_search,
            candidate=value_search.value,
        )
        adjustment_probability = None
    else:
        if fixed_cost_support is None:
            msg = "fixed_cost_scale requires fixed_cost_support"
            raise ValueError(msg)
        V_arr, adjustment_probability = _fold_expected_fixed_cost(
            running=keeper_v_arr,
            stacked=bank.V_arr,
            search=value_search,
            scale=fixed_cost_scale,
            support=fixed_cost_support,
        )

    stacked_carry = bank.carry
    carry_search = _continuous_outer_argmax(
        nodes=bank.outer_nodes, stacked=stacked_carry.value, config=config
    )
    adjuster_marginal, _ = _INTERPOLANT.evaluate_with_derivative(
        nodes=bank.outer_nodes,
        values=stacked_carry.marginal_utility,
        query=carry_search.x,
    )
    if fixed_cost_scale is None:
        collapsed_value = _fold_continuous(
            running=keeper_carry.value,
            stacked=stacked_carry.value,
            search=carry_search,
            candidate=carry_search.value,
        )
        collapsed_marginal = _fold_continuous(
            running=keeper_carry.marginal_utility,
            stacked=stacked_carry.value,
            search=carry_search,
            candidate=adjuster_marginal,
            take_from=keeper_carry.value,
            take_candidate=carry_search.value,
            invalid_fill=0.0,
        )
    else:
        if fixed_cost_support is None:
            msg = "fixed_cost_scale requires fixed_cost_support"
            raise ValueError(msg)
        collapsed_value, row_probability = _fold_expected_fixed_cost(
            running=keeper_carry.value,
            stacked=stacked_carry.value,
            search=carry_search,
            scale=fixed_cost_scale,
            support=fixed_cost_support,
        )
        # Envelope: the cutoff's own derivative drops, so the expected
        # marginal is the probability-weighted blend of the two branches'
        # conditional marginals (each already `0.0` on its dead rows).
        collapsed_marginal = (
            row_probability
            * jnp.where(jnp.isfinite(carry_search.value), adjuster_marginal, 0.0)
            + (1.0 - row_probability) * keeper_carry.marginal_utility
        )
    carry = replace(
        keeper_carry,
        value=collapsed_value,
        marginal_utility=collapsed_marginal,
    )

    return ContinuousCollapse(
        V_arr=V_arr,
        carry=carry,
        value_search=value_search,
        keeper_adjuster_margin=_min_finite_margin(keeper_v_arr, value_search.value),
        best_second_best_margin=_min_finite_margin(
            value_search.value, value_search.second_best_value
        ),
        adjustment_probability=adjustment_probability,
    )


def _fold_expected_fixed_cost(
    *,
    running: FloatND,
    stacked: FloatND,
    search: SafeguardedSearchResult,
    scale: FloatND,
    support: tuple[float, float],
) -> tuple[FloatND, FloatND]:
    """Fold keeper vs continuous-adjuster through the analytic expectation.

    Keeps the deterministic fold's degeneracy vocabulary: a search-invalid
    candidate cell is restored to NaN when its whole stacked column is NaN
    (aligned padding rides through) and to `-inf` otherwise; a NaN-dead side
    then acts as `-inf` in the closed form (the live side's own expectation
    wins), and a cell dead on both sides keeps the keeper's original entry
    with adjustment probability `0.0`.
    """
    all_nan = jnp.all(jnp.isnan(stacked), axis=0)
    candidate = jnp.where(
        search.valid,
        search.value,
        jnp.where(all_nan, jnp.nan, -jnp.inf),
    )
    keeper_dead = jnp.isnan(running)
    candidate_dead = jnp.isnan(candidate)
    result = aggregate_uniform_observed_fixed_cost(
        keeper_value=jnp.where(keeper_dead, -jnp.inf, running),
        adjuster_value=jnp.where(candidate_dead, -jnp.inf, candidate),
        scale=jnp.broadcast_to(
            scale, jnp.broadcast_shapes(running.shape, candidate.shape)
        ),
        lower=support[0],
        upper=support[1],
    )
    both_dead = keeper_dead & candidate_dead
    value = jnp.where(both_dead, running, result.expected_value)
    probability = jnp.where(both_dead, 0.0, result.adjustment_probability)
    return value, probability


def _continuous_outer_argmax(
    *,
    nodes: FloatND,
    stacked: FloatND,
    config: AdaptiveOuterMesh,
) -> SafeguardedSearchResult:
    """Safeguarded continuous argmax over the outer axis of one surface."""

    def objective(query: FloatND) -> FloatND:
        return _INTERPOLANT.evaluate(nodes=nodes, values=stacked, query=query)

    return safeguarded_continuous_argmax(
        objective,
        nodes=nodes,
        node_values=stacked,
        golden_iterations=config.golden_iterations,
    )


def _fold_continuous(
    *,
    running: FloatND,
    stacked: FloatND,
    search: SafeguardedSearchResult,
    candidate: FloatND,
    take_from: FloatND | None = None,
    take_candidate: FloatND | None = None,
    invalid_fill: float = -jnp.inf,
) -> FloatND:
    """Fold the continuous adjuster read into the keeper, finite-fold style.

    The take decision compares *values* (`take_from` / `take_candidate`,
    defaulting to `running` / `candidate` themselves) so a companion array —
    the marginal — follows its own surface's winner: where the adjuster's
    value wins, the adjuster's marginal is published.

    Candidate cells the search marked invalid are restored to the finite
    fold's vocabulary before comparing: an all-NaN candidate column becomes
    NaN (never takes, NaN padding rides through), an all-`-inf` column
    becomes `-inf` (competes and loses, but does take over a NaN-dead
    keeper cell exactly as `fmax` does finitely). `invalid_fill` sets the
    *published* payload for that second case — `-inf` for value surfaces,
    `0.0` for marginals (the carry contract's infeasible-cell marginal).
    """
    compare_running = running if take_from is None else take_from
    compare_candidate = candidate if take_candidate is None else take_candidate
    all_nan = jnp.all(jnp.isnan(stacked), axis=0)
    restored = jnp.where(
        search.valid,
        compare_candidate,
        jnp.where(all_nan, jnp.nan, -jnp.inf),
    )
    take = (restored > compare_running) | (
        jnp.isnan(compare_running) & ~jnp.isnan(restored)
    )
    payload = jnp.where(
        search.valid,
        candidate,
        jnp.where(all_nan, jnp.nan, invalid_fill),
    )
    return jnp.where(take, payload, running)


def _min_finite_margin(first: FloatND, second: FloatND) -> FloatND:
    """Smallest `|first - second|` over cells where both are finite.

    `+inf` when no cell has both branches finite — an infinite margin is
    the correct \"no near-tie anywhere\" reading for the release gates.
    """
    both = jnp.isfinite(first) & jnp.isfinite(second)
    gap = jnp.where(both, jnp.abs(first - second), jnp.inf)
    return jnp.min(gap)
