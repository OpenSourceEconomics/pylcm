"""A fixed-zero joint edge must not take its source state's validity with it.

The source `source` moves to `low` with probability `1 - p_high` and to `high`
with probability `p_high`. `low` is terminal and worth 2. `high` is terminal and
worth its wealth, which a joint lottery draws as 1 with probability 1/4 and 3 with
probability 3/4. Two source layouts share this graph:

- `wealth_source`: `source` carries wealth in {1, 2, 3} and its utility reads it.
  The joint lottery is the only declared law for wealth. The value of `source` is
  `wealth + 2 + p_high / 2`, so at `p_high = 0` it is `[3, 4, 5]`.
- `driver_source`: `source` carries a binary `driver`, carried by identity, whose
  only reader is the joint lottery, which shifts drawn wealth by `driver`. The
  value of `source` is `2 + p_high / 2 + p_high * driver`.

Removing the `high` edge when `p_high` is fixed at exactly zero changes the
effective graph only; the authored model stays valid and keeps its source states.
"""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    JointTransition,
    LinSpacedGrid,
    Model,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime
from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarFloat, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    low: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Driver:
    low: ScalarInt
    high: ScalarInt


def _low_mass(*, p_high: ScalarFloat) -> FloatND:
    return 1.0 - p_high


def _high_mass(*, p_high: ScalarFloat) -> FloatND:
    return jnp.asarray(p_high)


def _lottery_probabilities() -> FloatND:
    return jnp.asarray([0.25, 0.75])


def _lottery_wealth(*, match: dict[str, FloatND]) -> ContinuousState:
    return match["wealth"]


def _lottery_wealth_with_driver(
    *, match: dict[str, FloatND], driver: DiscreteState
) -> ContinuousState:
    return match["wealth"] + driver


def _wealth_utility(*, wealth: ContinuousState) -> FloatND:
    return wealth


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _two() -> FloatND:
    return jnp.asarray(2.0)


def _model(
    *,
    source_kind: str,
    fixed: bool,
    probability: float,
    enable_jit: bool,
    omit_joint: bool = False,
    omit_driver_read: bool = False,
    high_wealth_law: bool = False,
    joint_output: str = "wealth",
) -> Model:
    if source_kind == "wealth_source":
        states = {"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)}
        state_transitions = (
            {"wealth": {"high": _wealth_utility}} if high_wealth_law else {}
        )
        utility = _wealth_utility
        output = _lottery_wealth
    else:
        states = {"driver": DiscreteGrid(category_class=_Driver)}
        state_transitions = {"driver": fixed_transition(state_name="driver")}
        utility = _zero
        output = _lottery_wealth if omit_driver_read else _lottery_wealth_with_driver
    joint_transitions = (
        {}
        if omit_joint
        else {
            "high": {
                "match": JointTransition(
                    support_size=2,
                    support={"wealth": jnp.asarray([1.0, 3.0])},
                    probabilities=_lottery_probabilities,
                    outputs={joint_output: output},
                )
            }
        }
    )
    regimes = {
        "source": Regime(
            regime_transitions={
                "low": StochasticTransition(func=_low_mass),
                "high": StochasticTransition(func=_high_mass),
            },
            states=states,
            state_transitions=state_transitions,
            joint_transitions=joint_transitions,
            functions={"utility": utility},
        ),
        "low": Regime(regime_transitions=None, functions={"utility": _two}),
        "high": Regime(
            regime_transitions=None,
            states={"wealth": LinSpacedGrid(start=1.0, stop=4.0, n_points=4)},
            functions={"utility": _wealth_utility},
        ),
    }
    return Model(
        regimes=regimes,
        regime_id_class=_RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": {"low": 0, "high": 0}},
        initial_nodes=((0, "source"),),
        fixed_params={"p_high": probability} if fixed else {},
        enable_jit=enable_jit,
    )


def _expected_source_value(*, source_kind: str, probability: float) -> np.ndarray:
    """Enumerate the two-period paths of `source` in exact rational arithmetic."""
    p_high = Fraction(probability)
    coordinates = (1, 2, 3) if source_kind == "wealth_source" else (0, 1)
    values = []
    for coordinate in coordinates:
        flow, shift = (
            (Fraction(coordinate), 0)
            if source_kind == "wealth_source"
            else (Fraction(0), coordinate)
        )
        paths = (
            (1 - p_high, Fraction(2)),
            (p_high / 4, Fraction(1 + shift)),
            (3 * p_high / 4, Fraction(3 + shift)),
        )
        values.append(flow + sum(mass * payoff for mass, payoff in paths))
    return np.asarray([float(value) for value in values])


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("source_kind", ["wealth_source", "driver_source"])
def test_source_value_matches_enumeration(
    *, source_kind: str, fixed: bool, probability: float, enable_jit: bool
) -> None:
    """The source value equals the exact path enumeration, zero edge or not."""
    model = _model(
        source_kind=source_kind,
        fixed=fixed,
        probability=probability,
        enable_jit=enable_jit,
    )
    params = {"discount_factor": 1.0} | ({} if fixed else {"p_high": probability})
    result = model.solve(params=params, log_level="off")
    np.testing.assert_allclose(
        np.asarray(result.values[0]["source"]).reshape(-1),
        _expected_source_value(source_kind=source_kind, probability=probability),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("source_kind", "state_name"),
    [("wealth_source", "wealth"), ("driver_source", "driver")],
)
def test_fixed_zero_edge_is_removed_and_source_state_kept(
    *, source_kind: str, state_name: str
) -> None:
    """A zero `high` edge leaves the effective graph; the source state stays."""
    model = _model(
        source_kind=source_kind, fixed=True, probability=0.0, enable_jit=False
    )
    assert (
        model.state_names(regime_name="source"),
        model.graph.solution.targets(period=0, source="source"),
        model.graph.simulation.targets(period=0, source="source"),
        set(model.graph.edges.solve["source"]),
        dict(model.graph.pruned_edges["solve"]),
    ) == (
        (state_name,),
        ("low",),
        ("low",),
        {"low", "high"},
        {(0, "source", "high"): "fixed_zero_probability"},
    )


def test_positive_edge_without_a_wealth_law_is_rejected() -> None:
    """Without the joint lottery, the positive `high` edge has no wealth producer."""
    with pytest.raises(ModelInitializationError, match=r"Missing: \{'wealth'\}"):
        _model(
            source_kind="wealth_source",
            fixed=True,
            probability=0.5,
            enable_jit=False,
            omit_joint=True,
        )


def test_driver_that_nothing_reads_is_rejected() -> None:
    """A source state read by no function of the authored model is rejected."""
    with pytest.raises(ModelInitializationError, match="never used"):
        _model(
            source_kind="driver_source",
            fixed=True,
            probability=0.5,
            enable_jit=False,
            omit_driver_read=True,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"high_wealth_law": True}, "multiple producers claim target-state cell"),
        ({"joint_output": "debt"}, "'debt' of kernel 'match' is not a target state"),
    ],
)
def test_malformed_joint_kernel_on_a_fixed_zero_edge_is_rejected(
    *, overrides: dict, message: str
) -> None:
    """A joint lottery removed with its zero edge must still be well formed."""
    with pytest.raises(ModelInitializationError, match=message):
        _model(
            source_kind="wealth_source",
            fixed=True,
            probability=0.0,
            enable_jit=False,
            **overrides,
        )
