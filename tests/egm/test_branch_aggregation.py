"""Closed-form uniform-fixed-cost aggregation against dense quadrature.

Every cutoff regime (`Delta < 0`, `Delta = 0`, interior
cutoff, cutoff at both support boundaries, `Delta > B*b`) is compared to a
1,000,001-node midpoint quadrature of `E[max(V_K, V_A - B*chi)]`; the
`B = 0` degenerate branch reproduces the deterministic keeper-wins-ties
maximum; boundary continuity holds across the regime switches; and the
feasibility degeneracies keep the fold's vocabulary.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.branch_aggregation import (
    OuterBranchAggregator,
    UniformObservedFixedCost,
    aggregate_uniform_observed_fixed_cost,
)
from lcm.consumption_savings_regime import OuterContinuousMargin, outer_unchanged
from lcm.exceptions import RegimeInitializationError

_N_QUADRATURE = 1_000_001


def _quadrature(
    *, keeper: float, adjuster: float, scale: float, lower: float, upper: float
) -> tuple[float, float]:
    """Midpoint-rule expectation and adjustment probability."""
    edges = np.linspace(lower, upper, _N_QUADRATURE + 1)
    chi = 0.5 * (edges[:-1] + edges[1:])
    net = adjuster - scale * chi
    values = np.maximum(keeper, net)
    return float(values.mean()), float((net > keeper).mean())


@pytest.mark.parametrize(
    ("keeper", "adjuster", "scale", "lower", "upper"),
    [
        (1.0, 0.5, 2.0, 0.0, 1.0),  # Delta < 0: never adjust
        (1.0, 1.0, 2.0, 0.0, 1.0),  # Delta = 0: cutoff at the lower edge
        (1.0, 2.0, 2.0, 0.0, 1.0),  # interior cutoff t = 0.5
        (1.0, 1.6, 2.0, 0.3, 1.0),  # cutoff exactly at lower support edge
        (1.0, 3.0, 2.0, 0.0, 1.0),  # cutoff exactly at upper support edge
        (1.0, 5.0, 2.0, 0.0, 1.0),  # Delta > B*b: always adjust
        (-2.0, -1.4, 0.7, 0.2, 0.9),  # negative values, shifted support
    ],
)
def test_closed_form_matches_dense_quadrature(
    *, keeper: float, adjuster: float, scale: float, lower: float, upper: float
) -> None:
    result = aggregate_uniform_observed_fixed_cost(
        keeper_value=jnp.asarray(keeper),
        adjuster_value=jnp.asarray(adjuster),
        scale=jnp.asarray(scale),
        lower=lower,
        upper=upper,
    )
    expected, probability = _quadrature(
        keeper=keeper, adjuster=adjuster, scale=scale, lower=lower, upper=upper
    )
    # The quadrature reference is always float64, but the closed form is evaluated in
    # whatever precision the suite runs at, so its own error is a few epsilon of that
    # format. `rtol=1e-9` is a float64-grade bound: at float32 it would be testing the
    # format rather than the formula. Flooring both tolerances at the working
    # resolution leaves float64 exactly as it was -- there the epsilon term is ~7e-15,
    # six decades under the constants.
    eps = float(np.finfo(jnp.zeros(()).dtype).eps)
    np.testing.assert_allclose(
        float(result.expected_value),
        expected,
        rtol=max(1e-9, 32.0 * eps),
        atol=max(1e-10, 32.0 * eps),
    )
    np.testing.assert_allclose(
        float(result.adjustment_probability),
        probability,
        rtol=max(1e-6, 32.0 * eps),
        atol=max(1e-6, 32.0 * eps),
    )
    np.testing.assert_allclose(
        float(result.no_adjustment_probability),
        1.0 - float(result.adjustment_probability),
        rtol=0.0,
        atol=0.0,
    )


def test_value_is_continuous_across_the_regime_switches() -> None:
    """No jump as the cutoff crosses either support edge."""
    scale, lower, upper = 2.0, 0.0, 1.0
    keeper = 1.0
    eps = 1e-9
    for edge in (lower, upper):
        below, above = (
            float(
                aggregate_uniform_observed_fixed_cost(
                    keeper_value=jnp.asarray(keeper),
                    adjuster_value=jnp.asarray(keeper + scale * (edge + sign * eps)),
                    scale=jnp.asarray(scale),
                    lower=lower,
                    upper=upper,
                ).expected_value
            )
            for sign in (-1.0, 1.0)
        )
        np.testing.assert_allclose(below, above, atol=1e-8)


def test_zero_scale_reproduces_the_deterministic_maximum() -> None:
    keeper = jnp.array([1.0, 1.0, 1.0])
    adjuster = jnp.array([0.5, 1.0, 2.0])
    result = aggregate_uniform_observed_fixed_cost(
        keeper_value=keeper,
        adjuster_value=adjuster,
        scale=jnp.zeros(3),
        lower=0.0,
        upper=1.0,
    )
    np.testing.assert_array_equal(np.asarray(result.expected_value), [1.0, 1.0, 2.0])
    # The keeper wins the exact tie: adjustment probability 0 there.
    np.testing.assert_array_equal(
        np.asarray(result.adjustment_probability), [0.0, 0.0, 1.0]
    )


def test_feasibility_degeneracies_keep_the_fold_vocabulary() -> None:
    keeper = jnp.array([1.0, -jnp.inf, -jnp.inf])
    adjuster = jnp.array([-jnp.inf, 2.0, -jnp.inf])
    result = aggregate_uniform_observed_fixed_cost(
        keeper_value=keeper,
        adjuster_value=adjuster,
        scale=jnp.full(3, 2.0),
        lower=0.0,
        upper=1.0,
    )
    np.testing.assert_array_equal(
        np.asarray(result.expected_value), [1.0, 2.0 - 1.0, -jnp.inf]
    )
    np.testing.assert_array_equal(
        np.asarray(result.adjustment_probability), [0.0, 1.0, 0.0]
    )


def test_probability_is_exact_in_the_interior() -> None:
    """Interior cutoff: p = (t - a) / (b - a) exactly."""
    result = aggregate_uniform_observed_fixed_cost(
        keeper_value=jnp.asarray(1.0),
        adjuster_value=jnp.asarray(2.0),
        scale=jnp.asarray(2.0),
        lower=0.0,
        upper=1.0,
    )
    np.testing.assert_allclose(float(result.adjustment_probability), 0.5, atol=1e-15)
    np.testing.assert_allclose(float(result.cutoff), 0.5, atol=1e-15)


def test_config_requires_positive_support_width() -> None:
    with pytest.raises(RegimeInitializationError, match="support width"):
        UniformObservedFixedCost(
            shock_name="adjustment_cost",
            scale_function="adjustment_cost_scale",
            lower=1.0,
            upper=1.0,
        )


def test_unsupported_aggregator_subclass_is_rejected_where_it_is_declared() -> None:
    """An adjustment cost with no kernel behind it fails loudly, not silently.

    `OuterBranchAggregator` is a closed set. A third concrete subclass names an
    aggregation the kernels do not implement, and running it as the
    deterministic maximum would publish a value function for a model nobody
    specified. The margin is where the cost is declared, so it is where the
    declaration is refused.
    """

    @dataclass(frozen=True, kw_only=True)
    class LogitAdjustment(OuterBranchAggregator):
        """A stand-in for an aggregation the kernels do not implement."""

        temperature: float = 1.0

    with pytest.raises(RegimeInitializationError, match="LogitAdjustment"):
        OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
            adjustment_cost=LogitAdjustment(),
        )
