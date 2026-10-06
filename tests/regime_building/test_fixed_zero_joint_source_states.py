"""A fixed-zero joint edge gives exactly the model declared without that edge.

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

Removing the `high` edge when `p_high` is fixed at exactly zero gives the model
the author would get by declaring no `high` edge at all. Pruning removes as much
as it can up front:

- wealth keeps the empty per-target law `{}` and the model stays valid;
- `driver` is read only across the removed edge, so it is unused and the model
  is rejected, exactly as the edge-free model is.
"""

from collections.abc import Callable
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
class _RegimeIdWithMid:
    source: ScalarInt
    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


@categorical(ordered=False)
class _Driver:
    low: ScalarInt
    high: ScalarInt


def _low_mass(*, p_high: ScalarFloat) -> FloatND:
    return 1.0 - p_high


def _low_mass_beside_mid(*, p_high: ScalarFloat) -> FloatND:
    return 0.5 - p_high


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _certain() -> FloatND:
    return jnp.asarray(1.0)


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
    with_mid: bool = False,
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
    mid_transition = {"mid": StochasticTransition(func=_half)} if with_mid else {}
    regimes = {
        "source": Regime(
            regime_transitions={
                "low": StochasticTransition(
                    func=_low_mass_beside_mid if with_mid else _low_mass
                ),
                "high": StochasticTransition(func=_high_mass),
            }
            | mid_transition,
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
    if with_mid:
        regimes["mid"] = Regime(
            regime_transitions=None,
            states={"wealth": LinSpacedGrid(start=1.0, stop=4.0, n_points=4)},
            functions={"utility": _wealth_utility},
        )
    return Model(
        regimes=regimes,
        regime_id_class=_RegimeIdWithMid if with_mid else _RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={"source": {"low": 0, "high": 0} | ({"mid": 0} if with_mid else {})},
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


_VALUED_CASES = [
    pytest.param(
        source_kind, fixed, probability, id=f"{source_kind}-{fixed}-{probability}"
    )
    for source_kind in ("wealth_source", "driver_source")
    for fixed in (False, True)
    for probability in (0.0, 0.5, 1.0)
    # The fixed-zero driver model is rejected; see the unused-driver test.
    if not (source_kind == "driver_source" and fixed and probability == 0.0)
]


@pytest.mark.parametrize("enable_jit", [False, True])
@pytest.mark.parametrize(("source_kind", "fixed", "probability"), _VALUED_CASES)
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


def test_fixed_zero_edge_is_removed_and_wealth_kept() -> None:
    """A zero `high` edge leaves the effective graph; the wealth state stays."""
    model = _model(
        source_kind="wealth_source", fixed=True, probability=0.0, enable_jit=False
    )
    assert (
        model.state_names(regime_name="source"),
        model.graph.solution.targets(period=0, source="source"),
        model.graph.simulation.targets(period=0, source="source"),
        set(model.graph.edges.solve["source"]),
        dict(model.graph.pruned_edges["solve"]),
    ) == (
        ("wealth",),
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


def test_kept_target_needing_a_state_only_a_removed_lottery_produced_is_rejected() -> (
    None
):
    """A target that keeps its edge and carries wealth still needs a wealth law.

    `mid` carries wealth and keeps a positive edge; the only declared wealth law
    is the joint lottery toward `high`, which leaves with the zero edge.
    """
    with pytest.raises(
        ModelInitializationError,
        match=r"does not cover reachable target\(s\) \['mid'\]",
    ):
        _model(
            source_kind="wealth_source",
            fixed=True,
            probability=0.0,
            enable_jit=False,
            with_mid=True,
        )


def _edge_free_wealth_model() -> Model:
    """The wealth source as authored without the `high` edge and its lottery."""
    regimes = {
        "source": Regime(
            regime_transitions={"low": StochasticTransition(func=_certain)},
            states={"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=3)},
            state_transitions={"wealth": {}},
            functions={"utility": _wealth_utility},
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
        edges={"source": {"low": 0}},
        initial_nodes=((0, "source"),),
        enable_jit=False,
    )


def _outcome(*, model: Model) -> tuple[object, ...]:
    """The source's states, pruned variables, effective targets and values."""
    values = model.solve(params={"discount_factor": 1.0}, log_level="off").values
    return (
        model.state_names(regime_name="source"),
        dict(model.pruned_variables),
        model.graph.solution.targets(period=0, source="source"),
        np.asarray(values[0]["source"]).tolist(),
    )


def test_fixed_zero_wealth_model_equals_the_edge_free_model() -> None:
    """A weight-0 edge gives exactly the model without that edge.

    The edge-free author writes wealth's law as the empty per-target mapping
    `{"wealth": {}}`; both models carry wealth, prune nothing, reach only `low`
    and value the source at `[3, 4, 5]`.
    """
    fixed_zero = _model(
        source_kind="wealth_source", fixed=True, probability=0.0, enable_jit=False
    )
    assert _outcome(model=fixed_zero) == _outcome(model=_edge_free_wealth_model())


def _edge_free_driver_model() -> Model:
    """The driver source as authored without the `high` edge and its lottery."""
    regimes = {
        "source": Regime(
            regime_transitions={"low": StochasticTransition(func=_certain)},
            states={"driver": DiscreteGrid(category_class=_Driver)},
            state_transitions={"driver": fixed_transition(state_name="driver")},
            functions={"utility": _zero},
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
        edges={"source": {"low": 0}},
        initial_nodes=((0, "source"),),
        enable_jit=False,
    )


def _initialization_error(build: Callable[[], object]) -> str:
    """Return the message of the `ModelInitializationError` building raises."""
    with pytest.raises(ModelInitializationError) as raised:
        build()
    return str(raised.value)


def test_driver_read_only_across_the_zero_edge_is_unused_like_without_the_edge() -> (
    None
):
    """Both driver models are rejected for the same unused `driver`.

    The fixed-zero model's message extends the edge-free one by naming the
    removed edge the only reader of `driver` sat on.
    """
    fixed_zero = _initialization_error(
        lambda: _model(
            source_kind="driver_source", fixed=True, probability=0.0, enable_jit=False
        )
    )
    edge_free = _initialization_error(_edge_free_driver_model)
    assert (
        fixed_zero.startswith(edge_free),
        "'driver' is read only across the edge(s) 'source' -> 'high'" in fixed_zero,
    ) == (True, True)
