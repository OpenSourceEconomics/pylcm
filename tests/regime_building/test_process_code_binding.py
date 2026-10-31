"""`<process>_code` reads the node position on the reading regime's own nodes.

A process state holds the value of a node, and an argument `z_code` receives that
node's int32 position. The position belongs to the regime whose state is read:
the declaring regime for its own functions, constraints and their projections,
and the landing target for a gated edge's gate and fallbacks. Each
oracle below is a literal node table or payoff, never a production coordinate.
"""

from collections.abc import Mapping
from itertools import pairwise
from math import erf, sqrt
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    ByAge,
    CollectiveUtility,
    DeterministicTransition,
    DiscreteGrid,
    Gate,
    Model,
    NormalIIDProcess,
    ParticipationCheck,
    ParticipationRead,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    UniformIIDProcess,
    ValueDependentConstraint,
    categorical,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserFunction,
)

NODES = (-2.0, -1.0, 0.0, 1.0, 2.0)
BRACKET = 0.25


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class RouteRegime:
    young: ScalarInt
    married: ScalarInt
    single: ScalarInt


@categorical(ordered=False)
class TwoRegimes:
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class CoupleRegime:
    couple: ScalarInt
    single: ScalarInt


@categorical(ordered=False)
class OnlyRegime:
    only: ScalarInt


@categorical(ordered=False)
class LandingRegime:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _process(*, mu: float, n_points: int = 5, sigma: float = 1.0, n_std: float = 2.0):
    return NormalIIDProcess(
        n_points=n_points,
        gauss_hermite=False,
        mu=mu,
        sigma=sigma,
        n_std=n_std,
        simulated_draws="nodes",
    )


def _cubic(x):
    return x * x * x + 2.0 * x + 5.0


def _source_utility(z: ContinuousState) -> FloatND:
    return 0.0 * z


def _single_utility(z: ContinuousState) -> FloatND:
    return _cubic(z)


def _wife(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return _cubic(z) + BRACKET * (2.0 * work - 1.0)


def _husband(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return 100.0 * (1.0 - work) + 0.0 * z


def _project_node(z_code: DiscreteState) -> ContinuousState:
    return jnp.asarray(NODES)[z_code]


def _participates(*, Q_f: FloatND, V_ref: FloatND) -> BoolND:
    return Q_f >= V_ref


def _gate(z_code: DiscreteState) -> BoolND:
    return z_code >= 3


def _one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _route_model(*, reverse: bool, enable_jit: bool) -> Model:
    """The source's nodes are [8, 9, 10, 11, 12]; both targets' are `NODES`."""
    reference = ProjectedRegimeValue(regime="single", projection={"z": _project_node})
    young = Regime(
        states={"z": _process(mu=10.0)}, functions={"utility": _source_utility}
    )
    married = Regime(
        states={"z": _process(mu=0.0)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": CollectiveUtility(utilities={"f": _wife, "m": _husband})},
        constraints={
            "participation": ValueDependentConstraint(
                predicate=_participates, references={"V_ref": reference}
            )
        },
    )
    single = Regime(
        states={"z": _process(mu=0.0)}, functions={"utility": _single_utility}
    )
    entries = [("young", young), ("married", married), ("single", single)]
    return Model(
        regimes=dict(reversed(entries) if reverse else entries),
        regime_id_class=RouteRegime,
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "young"},
        enable_jit=enable_jit,
        edges={
            "young": Transition(
                targets={"married": 0, "single": 0},
                law=ByAge(cases={0: {"married": StochasticTransition(func=_one)}}),
                gates={
                    "married": Gate(
                        predicate=_gate,
                        routes={
                            "only": StakeholderRoute(
                                target_stakeholder="f", fallback=reference
                            )
                        },
                    )
                },
            )
        },
    )


def _normal_node_weights() -> tuple[float, ...]:
    """Normal cell masses between the midpoints of `NODES`, from `math.erf`."""

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    cuts = (0.0, *(cdf(x) for x in (-1.5, -0.5, 0.5, 1.5)), 1.0)
    return tuple(right - left for left, right in pairwise(cuts))


def _route_values(*, reverse: bool, enable_jit: bool):
    params = {
        "young": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "married": {},
        "single": {},
    }
    model = _route_model(reverse=reverse, enable_jit=enable_jit)
    return model.solve(params=params, log_level="off").values


@pytest.mark.parametrize("reverse", [False, True], ids=["source-first", "source-last"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_constraint_projection_reads_the_declaring_regimes_nodes(
    *, reverse: bool, enable_jit: bool
):
    """At every married node, only work is feasible, worth `cubic(z) + 0.25`."""
    values = _route_values(reverse=reverse, enable_jit=enable_jit)
    expected = np.array([[_cubic(z) + BRACKET, 0.0] for z in NODES])
    np.testing.assert_array_equal(values[1]["married"], expected)


@pytest.mark.parametrize("reverse", [False, True], ids=["source-first", "source-last"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_gate_and_fallback_read_the_landing_targets_nodes(
    *, reverse: bool, enable_jit: bool
):
    """The gate opens at target positions 3 and 4, where it adds the surplus."""
    values = _route_values(reverse=reverse, enable_jit=enable_jit)
    expected = sum(
        weight * (_cubic(z) + (BRACKET if z > 0.0 else 0.0))
        for z, weight in zip(NODES, _normal_node_weights(), strict=True)
    )
    np.testing.assert_allclose(
        values[0]["young"], np.full(5, expected), rtol=0, atol=1e-5
    )


def test_process_code_is_not_a_parameter_of_a_projection_or_gate():
    model = _route_model(reverse=False, enable_jit=False)
    assert "z_code" not in str(model.get_params_template())


def _position_payoff(z_code: DiscreteState) -> FloatND:
    # Indexing with int64 would pass a value check, so the dtype is checked too.
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError(z_code.dtype)
    return jnp.asarray(z_code, dtype=float)


# (node count, mean in a, mean in b, sigma, n_std): coincident, overlapping,
# disjoint, shifted and differently scaled grids.
_GRID_CASES = [
    (3, 0.0, 0.0, 1.0, 1.0),
    (3, -4.0, 4.0, 1.0, 1.0),
    (5, 10.0, 0.0, 1.0, 2.0),
    (5, 0.0, 1.0, 1.0, 2.0),
    (6, 0.1, 0.4, 0.3, 1.3),
    (7, -0.7, 0.2, 0.17, 1.7),
    (9, 20.0, 21.0, 0.125, 2.0),
    (9, -20.0, -10.0, 3.0, 2.0),
]


@pytest.mark.parametrize("case", _GRID_CASES)
@pytest.mark.parametrize("reverse", [False, True], ids=["a-first", "b-first"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_one_callable_in_two_regimes_reads_each_regimes_positions(
    *, case: tuple[int, float, float, float, float], reverse: bool, enable_jit: bool
):
    count, mean_a, mean_b, sigma, n_std = case
    entries = [
        (
            name,
            Regime(
                states={
                    "z": _process(mu=mean, n_points=count, sigma=sigma, n_std=n_std)
                },
                functions={"utility": _position_payoff},
            ),
        )
        for name, mean in (("a", mean_a), ("b", mean_b))
    ]
    model = Model(
        regimes=dict(reversed(entries) if reverse else entries),
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: ("a", "b")},
        regime_id_class=TwoRegimes,
        enable_jit=enable_jit,
    )
    values = model.solve(params={"a": {}, "b": {}}, log_level="off")
    materialized = values.values[0]
    np.testing.assert_array_equal(
        np.stack([materialized["a"], materialized["b"]]),
        np.stack([np.arange(count, dtype=float)] * 2),
    )


def _payoff_utility(payoff: FloatND) -> FloatND:
    return payoff


def _to_b() -> ScalarInt:
    return TwoRegimes.b


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def test_an_age_specialized_reader_of_a_process_code_is_refused():
    """An age-specialized function cannot read a process state's node position."""
    with pytest.raises(RegimeInitializationError, match=r"age-specialized.*z_code"):
        Model(
            regimes={
                "a": Regime(
                    states={"z": _process(mu=0.0)},
                    functions={
                        "utility": _payoff_utility,
                        "payoff": AgeSpecializedFunction(
                            build=lambda _age: _position_payoff,
                            signature=lambda _age: 0,
                        ),
                    },
                ),
                "b": Regime(functions={"utility": _zero_utility}),
            },
            edges={
                "a": Transition(
                    targets={"b": (0,)}, law=DeterministicTransition(func=_to_b)
                )
            },
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            initial_nodes={0: "a"},
            regime_id_class=TwoRegimes,
        )


def test_a_foreign_continuous_process_does_not_refuse_a_local_code_read():
    """Regime `b` draws `z` continuously but never reads its position."""
    continuous = NormalIIDProcess(
        n_points=5, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0
    )
    model = Model(
        regimes={
            "a": Regime(
                states={"z": _process(mu=0.0)}, functions={"utility": _position_payoff}
            ),
            "b": Regime(
                states={"z": continuous}, functions={"utility": _single_utility}
            ),
        },
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: ("a", "b")},
        regime_id_class=TwoRegimes,
    )
    values = model.solve(params={"a": {}, "b": {}}, log_level="off")
    np.testing.assert_array_equal(values.values[0]["a"], np.arange(5, dtype=float))


def test_phased_process_on_different_nodes_is_refused():
    with pytest.raises(
        RegimeInitializationError, match="solve and simulate laws differ"
    ):
        Model(
            regimes={
                "only": Regime(
                    states={
                        "z": Phased(solve=_process(mu=0.0), simulate=_process(mu=1.0))
                    },
                    functions={"utility": _position_payoff},
                )
            },
            edges={},
            ages=AgeGrid(exact_values=(0,)),
            initial_nodes={0: "only"},
            regime_id_class=OnlyRegime,
        )


_COUPLE_NODES = np.array([-1.0, 0.0, 1.0])
_SURPLUS = 0.25


def _couple_wife(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return z + _SURPLUS * (2.0 * work - 1.0)


def _couple_husband(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return 10.0 * (1.0 - work) + 0.0 * z


def _couple_node_at_code(z_code: DiscreteState) -> ContinuousState:
    return jnp.asarray(_COUPLE_NODES)[z_code]


def _identity(z: ContinuousState) -> ContinuousState:
    return z


def _value_of(z: ContinuousState) -> FloatND:
    return z


def _participation(*, Q_f: FloatND, V_outside: FloatND) -> BoolND:
    return Q_f >= V_outside


def _couple_model(
    *,
    binding: Literal["constructor", "model_fixed", "runtime"],
    enable_jit: bool,
    read_code: bool = True,
) -> Model:
    shock = NormalIIDProcess(
        n_points=3,
        gauss_hermite=False,
        mu=0.0,
        sigma=1.0 if binding == "constructor" else None,
        n_std=1.0,
        simulated_draws="nodes",
    )
    couple = Regime(
        states={"z": shock},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _couple_wife, "m": _couple_husband}
            )
        },
        constraints={
            "participation_f": ValueDependentConstraint(
                predicate=_participation,
                references={
                    "V_outside": ProjectedRegimeValue(
                        regime="single",
                        projection={
                            "z": _couple_node_at_code if read_code else _identity
                        },
                    )
                },
            )
        },
    )
    single = Regime(states={"z": shock}, functions={"utility": _value_of})
    return Model(
        regimes={"couple": couple, "single": single},
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: "couple"},
        regime_id_class=CoupleRegime,
        fixed_params=(
            {name: {"z": {"sigma": 1.0}} for name in ("couple", "single")}
            if binding == "model_fixed"
            else {}
        ),
        enable_jit=enable_jit,
    )


def _terminal_couple_values(model: Model) -> np.ndarray:
    solution = model.solve(params={"couple": {}, "single": {}}, log_level="off")
    return np.asarray(solution.values[0]["couple"])


@pytest.mark.parametrize("binding", ["constructor", "model_fixed"])
@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_fixed_process_law_reads_the_same_positions_wherever_it_is_fixed(
    *, binding: Literal["constructor", "model_fixed"], enable_jit: bool
):
    """A law fixed by `fixed_params` means what the constructor value means."""
    model = _couple_model(binding=binding, enable_jit=enable_jit)
    expected = np.column_stack((_COUPLE_NODES + _SURPLUS, np.zeros(3)))
    np.testing.assert_array_equal(_terminal_couple_values(model), expected)


def test_model_fixed_process_law_value_projection():
    model = _couple_model(binding="model_fixed", enable_jit=False, read_code=False)
    expected = np.column_stack((_COUPLE_NODES + _SURPLUS, np.zeros(3)))
    np.testing.assert_array_equal(_terminal_couple_values(model), expected)


def test_runtime_process_law_refuses_a_code_projection():
    with pytest.raises(
        RegimeInitializationError,
        match=r"z_code.*fix the parameters of `z` at construction",
    ):
        _couple_model(binding="runtime", enable_jit=False)


def _authored_code_like_name(z: ContinuousState) -> FloatND:
    return 10.0 + z


def _utility_from_authored_function(z_code: FloatND) -> FloatND:
    return z_code


def _only_model(functions: Mapping[str, UserFunction]) -> Model:
    return Model(
        regimes={"only": Regime(states={"z": _process(mu=0.0)}, functions=functions)},
        regime_id_class=OnlyRegime,
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: "only"},
        edges={},
        enable_jit=False,
    )


def test_a_function_named_like_a_process_code_is_refused():
    with pytest.raises(
        RegimeInitializationError,
        match=r"'z_code' in regime 'only'.*collides with the code",
    ):
        _only_model(
            {
                "z_code": _authored_code_like_name,
                "utility": _utility_from_authored_function,
            }
        )


def test_utility_process_code_is_not_a_parameter():
    template = _only_model({"utility": _position_payoff}).get_params_template()
    assert "z_code" not in str(template)


def test_utility_process_code_reads_positions():
    model = _only_model({"utility": _position_payoff})
    values = model.solve(params={"only": {}}, log_level="off").values
    np.testing.assert_array_equal(values[0]["only"], np.arange(5, dtype=float))


def _zero(z: ContinuousState) -> FloatND:
    return 0.0 * z


def _five(z: ContinuousState) -> FloatND:
    return 5.0 + 0.0 * z


def _one_payoff() -> FloatND:
    return jnp.asarray(1.0)


def _gate_on_landing_value(z: ContinuousState) -> BoolND:
    return z > 0.0


def _gate_without_process_read() -> BoolND:
    return jnp.ones((), dtype=bool)


def _participation_on_landing_value(V_target: FloatND) -> BoolND:
    return V_target > 0.0


def _landed_z(z: ContinuousState) -> ContinuousState:
    return z


def _landing_model(
    *, runtime: bool, edge: Literal["gate-read", "gate-no-read", "participation"]
) -> Model:
    """Source and target carry one uniform process on nodes [-1, 0, 1].

    The edge is a state `Gate` whose predicate reads the landed process or reads
    nothing, or a `ParticipationCheck` reading the target's value at the landing.
    """
    shock = UniformIIDProcess(
        n_points=3, start=-1.0, stop=None if runtime else 1.0, simulated_draws="nodes"
    )
    route = StakeholderRoute(
        fallback=ProjectedRegimeValue(regime="fallback", projection={})
    )
    targets = {"target": 0, "fallback": 0}
    law = ByAge(cases={0: {"target": StochasticTransition(func=_one)}})
    transition = (
        Transition(
            targets=targets,
            law=law,
            gates={
                "target": Gate(
                    predicate=(
                        _gate_on_landing_value
                        if edge == "gate-read"
                        else _gate_without_process_read
                    ),
                    routes={"only": route},
                )
            },
        )
        if edge != "participation"
        else Transition(
            targets=targets,
            law=law,
            participation={
                "target": ParticipationCheck(
                    predicate=_participation_on_landing_value,
                    reads={
                        "V_target": ParticipationRead(
                            kind="Raw", regime="target", projection={"z": _landed_z}
                        )
                    },
                    routes={"only": route},
                )
            },
        )
    )
    return Model(
        regimes={
            "source": Regime(states={"z": shock}, functions={"utility": _zero}),
            "target": Regime(states={"z": shock}, functions={"utility": _five}),
            "fallback": Regime(functions={"utility": _one_payoff}),
        },
        edges={"source": transition},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=LandingRegime,
    )


def _landing_source_values(*, model: Model, runtime: bool) -> np.ndarray:
    params: dict[str, dict[str, dict[str, float]]] = {
        "source": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "target": {},
        "fallback": {},
    }
    if runtime:
        params["source"]["z"] = {"stop": 1.0}
        params["target"]["z"] = {"stop": 1.0}
    solution = model.solve(params=params, log_level="off")
    return np.asarray(solution.values[0]["source"])


def test_gate_reads_the_landing_value_of_a_fixed_process():
    """The gate opens only at node 1, so `E[V] = (1 + 1 + 5) / 3`."""
    model = _landing_model(runtime=False, edge="gate-read")
    np.testing.assert_allclose(
        _landing_source_values(model=model, runtime=False),
        np.full(3, 7.0 / 3.0),
        rtol=0.0,
        atol=2e-6,
    )


def test_gate_reading_a_runtime_process_value_at_its_landing_is_refused():
    with pytest.raises(ModelInitializationError, match="parameterized at runtime"):
        _landing_model(runtime=True, edge="gate-read")


def test_participation_landing_in_a_runtime_process_is_refused():
    """A participation check enumerates the landed process by node; runtime has none."""
    with pytest.raises(ModelInitializationError, match="parameterized at runtime"):
        _landing_model(runtime=True, edge="participation")


def test_gate_not_reading_a_runtime_process_keeps_it_supported():
    """The gate refusal is scoped to the process it reads; open, so E[V] = 5."""
    model = _landing_model(runtime=True, edge="gate-no-read")
    np.testing.assert_allclose(
        _landing_source_values(model=model, runtime=True),
        np.full(3, 5.0),
        rtol=0.0,
        atol=2e-6,
    )
