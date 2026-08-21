"""A node of probability zero contributes nothing, whatever value stands there.

An entry law is evaluated at every node of a draw it reads, including nodes the
draw gives no probability. What the law names at an impossible node is not part of
the model, so it may name a value outside the target's support without consequence:
the node contributes exactly zero rather than a `NaN` to be multiplied away.

The loud failure is unaffected. A value outside the support at a node that *can*
occur is a misspecified model and still shows up as `NaN`.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.Q_and_F import _expectation_over_stochastic_nodes
from lcm import (
    AgeGrid,
    CertaintyEquivalent,
    DiscreteGrid,
    LinearExpectation,
    MarkovTransition,
    Model,
    PowerMean,
    Regime,
    UniformIIDProcess,
    categorical,
)
from lcm.typing import DiscreteState, FloatND, IntND, ScalarFloat, ScalarInt

_PARAMS = {"source": {"koopmans_aggregator": {"discount_factor": 1.0}}}
# The entry names 10.0 at the middle node, outside income's support.
_OFF_SUPPORT_NODE = 1
_ENTRY_VALUES = (1.0, 10.0, 2.0)


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class Health:
    good: ScalarInt
    off_support: ScalarInt
    poor: ScalarInt


def _to_target() -> ScalarFloat:
    return jnp.float32(1)


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _income_utility(income: ScalarFloat, health: DiscreteState) -> FloatND:
    return income + 0.0 * health


def _entry_income(next_health: DiscreteState) -> FloatND:
    """One physical value per health node, the middle one outside support."""
    return jnp.asarray(_ENTRY_VALUES)[next_health]


def _build(health_probabilities, certainty_equivalent=None) -> Model:
    def _health_probabilities() -> FloatND:
        return jnp.asarray(health_probabilities)

    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_to_target)},
                active=lambda age: age < 21,
                states={},
                state_transitions={
                    "income": {"target": _entry_income},
                    "health": {"target": MarkovTransition(_health_probabilities)},
                },
                functions={"utility": _zero_utility},
                certainty_equivalent=certainty_equivalent,
            ),
            "target": Regime(
                transition=None,
                states={
                    "income": UniformIIDProcess(start=0.0, stop=2.0, n_points=3),
                    "health": DiscreteGrid(Health),
                },
                functions={"utility": _income_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=21, step="Y"),
        regime_id_class=RegimeId,
    )


@pytest.mark.parametrize(
    ("health_probabilities", "expected"),
    [
        ((1.0, 0.0, 0.0), 1.0),
        ((0.0, 0.0, 1.0), 2.0),
        ((0.5, 0.0, 0.5), 1.5),
        ((0.25, 0.0, 0.75), 1.75),
    ],
    ids=["first_only", "last_only", "even_split", "uneven_split"],
)
def test_an_impossible_node_may_name_a_value_outside_the_targets_support(
    health_probabilities, expected
) -> None:
    """The continuation is the average over the nodes that can occur.

    The middle health node enters income at 10.0, outside the target's `[0, 2]`
    support, and carries no probability in any of these distributions. The value
    is the probability-weighted entry over the remaining nodes.
    """
    model = _build(health_probabilities)

    V = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_allclose(np.asarray(V[0]["source"]), np.asarray(expected))


def test_a_reachable_node_outside_the_targets_support_still_fails_loudly() -> None:
    """Giving the off-support node probability makes the model misspecified.

    The entry names 10.0 where income lives on `[0, 2]`, so there is no value to
    hand over and the continuation says so rather than quietly dropping the node.
    """
    model = _build((0.5, 0.5, 0.0))

    V = model.solve(params=_PARAMS, log_level="off")

    assert bool(jnp.isnan(jnp.asarray(V[0]["source"])))


def test_a_nonlinear_certainty_equivalent_omits_the_same_nodes() -> None:
    """`PowerMean` averages the live nodes only, and matches its own definition.

    With risk aversion two the certainty equivalent of the two possible entries
    is the harmonic mean `1 / (0.5/1 + 0.5/2)`, and the impossible node between
    them enters neither the transform nor the average.
    """
    model = _build((0.5, 0.0, 0.5), certainty_equivalent=PowerMean())
    params = {
        "source": {
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": 2.0},
        }
    }

    V = model.solve(params=params, log_level="off")

    np.testing.assert_allclose(
        np.asarray(V[0]["source"]), np.asarray(1.0 / (0.5 / 1.0 + 0.5 / 2.0)), rtol=1e-6
    )


def test_the_expectation_drops_a_zero_weight_node_rather_than_multiplying_it() -> None:
    """`E[V] = 1` for values `[1, NaN]` under weights `[1, 0]`."""
    got = _expectation_over_stochastic_nodes(
        values=jnp.asarray([1.0, jnp.nan]),
        weights=jnp.asarray([1.0, 0.0]),
        shifts=jnp.zeros(2, dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(1.0))


def test_a_negative_weight_is_not_laundered_into_a_zero_contribution() -> None:
    """A weight below zero is a malformed transition, not an impossible node.

    Values `[1, 2]` at weights `[1, -1]` give `1 - 2 = -1`. Dropping the
    negative node instead would give `1`, a value that looks like the answer to
    a well-posed question, so the malformed weight has to carry through.
    """
    got = _expectation_over_stochastic_nodes(
        values=jnp.asarray([1.0, 2.0]),
        weights=jnp.asarray([1.0, -1.0]),
        shifts=jnp.zeros(2, dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(-1.0))


def test_a_nan_weight_stays_poison() -> None:
    """A weight that is not a number is not a probability of zero either."""
    got = _expectation_over_stochastic_nodes(
        values=jnp.asarray([1.0, 2.0]),
        weights=jnp.asarray([1.0, jnp.nan]),
        shifts=jnp.zeros(2, dtype=jnp.int32),
    )

    assert bool(jnp.isnan(jnp.asarray(got)))


def _wide_spread() -> int:
    """A scale gap no ordinary weight can be named across.

    Wider than the exponent range, so the rare node's weight is zero once both
    are read against one base-two scale.
    """
    return 300 if jnp.zeros(()).dtype == jnp.float32 else 2800


def _mean_of_a_common_and_a_rare_node(
    *, rare_value: float, rare_weight: float, compile_it: bool
) -> FloatND:
    """The mean over a node of weight one and one a whole scale gap below it."""
    reduce = (
        jax.jit(_expectation_over_stochastic_nodes)
        if compile_it
        else _expectation_over_stochastic_nodes
    )
    return jnp.asarray(
        reduce(
            values=jnp.asarray([1.0, rare_value]),
            weights=jnp.asarray([1.0, rare_weight]),
            shifts=jnp.asarray([0, _wide_spread()], dtype=jnp.int32),
        )
    )


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_node_too_rare_to_weigh_against_a_nan_stays_poison(
    *, compile_it: bool
) -> None:
    """A live node keeps its `NaN` however far its weight is below the mass.

    Its weight cannot be named as a number on the scale the mean reads, but a
    weight the format cannot express is understated, not zero, and the node can
    still occur. Reporting an ordinary continuation for it would be an answer to
    a question the model does not pose.
    """
    got = _mean_of_a_common_and_a_rare_node(
        rare_value=float(jnp.nan), rare_weight=1.0, compile_it=compile_it
    )

    assert bool(jnp.isnan(got))


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_node_too_rare_to_weigh_against_an_infinity_keeps_it(
    *, compile_it: bool
) -> None:
    """Every strictly positive weight yields the same infinity, so it is kept."""
    got = _mean_of_a_common_and_a_rare_node(
        rare_value=float(jnp.inf), rare_weight=1.0, compile_it=compile_it
    )

    assert bool(jnp.isposinf(got))


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_node_too_rare_to_weigh_against_a_finite_value_is_omitted(
    *, compile_it: bool
) -> None:
    """A finite value at such a node cannot move the mean, so it is left out.

    This is the one omission the contract accepts, and it is one-sided: the
    node's contribution is understated rather than enlarged.
    """
    got = _mean_of_a_common_and_a_rare_node(
        rare_value=2.0, rare_weight=1.0, compile_it=compile_it
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(1.0))


@pytest.mark.parametrize("compile_it", [False, True], ids=["eager", "jit"])
def test_a_node_of_no_probability_stays_null_at_any_scale(*, compile_it: bool) -> None:
    """A represented zero is the null event whatever scale it is written against."""
    got = _mean_of_a_common_and_a_rare_node(
        rare_value=float(jnp.nan), rare_weight=0.0, compile_it=compile_it
    )

    np.testing.assert_allclose(np.asarray(got), np.asarray(1.0))


class _PlainWeightedMean(CertaintyEquivalent):
    """The ordinary weighted mean, written the way a user would write it.

    A user-written certainty equivalent states both reductions, because a
    lottery may arrive with its weights still carrying base-two scales. Neither
    does anything about impossible nodes, because nothing in either contract
    asks them to. The engine is what must guarantee that a node carrying no
    probability contributes nothing.
    """

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset()

    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],  # noqa: ARG002
    ) -> FloatND:
        return jnp.sum(weights * values, axis=-1) / jnp.sum(weights, axis=-1)

    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],
    ) -> FloatND:
        return LinearExpectation().aggregate_scaled(
            values=values, coefficients=coefficients, shifts=shifts, params=params
        )


def test_a_user_written_certainty_equivalent_is_handed_no_impossible_nodes() -> None:
    """An impossible node is neutralized before any `aggregate` implementation.

    The off-support entry makes the middle node's value NaN, and the middle
    node carries no probability. A user's weighted mean multiplies rather than
    masks, so `0 * NaN` would take the well-specified nodes down with it unless
    the engine has already replaced that value.
    """
    model = _build((0.5, 0.0, 0.5), certainty_equivalent=_PlainWeightedMean())

    V = model.solve(params=_PARAMS, log_level="off")

    np.testing.assert_allclose(np.asarray(V[0]["source"]), np.asarray(1.5), rtol=1e-6)
