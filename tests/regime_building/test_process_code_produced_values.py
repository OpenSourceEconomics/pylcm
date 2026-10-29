"""A produced process value read as `<process>_code` must stand on a node.

The simulation, not the user, produces these values: a state-dependent entry
law, a gate reading the drawn target state, a dissolution route back into the
same regime, and a later period of a supplied shock path. A regime reading the
node position refuses an off-node value from any of them; a node value from the
same producer is held. Separately, a process read by node position in one regime
keeps its positions when another regime declares a discrete variable of the same
name.

The node table `(-2, -1, 0, 1, 2)` is the reference; the off-node value `1/4`
and every node are exactly representable at both precisions.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.simulation.shock_paths import fail_if_a_held_process_code_is_off_node
from lcm import (
    AgeGrid,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Gate,
    Model,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidSimulationInputError,
    InvalidStateTransitionError,
    ModelInitializationError,
)
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

_NODES = (-2.0, -1.0, 0.0, 1.0, 2.0)
_OFF_NODE = 0.25


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class EntryId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class GatedId:
    source: ScalarInt
    landing: ScalarInt
    fallback: ScalarInt


@categorical(ordered=False)
class DissolutionId:
    source: ScalarInt
    a: ScalarInt
    b: ScalarInt


@categorical(ordered=False)
class ForeignId:
    a: ScalarInt
    b: ScalarInt


def _process() -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=5,
        mu=0.0,
        sigma=1.0,
        n_std=2.0,
        gauss_hermite=False,
        simulate_draws="nodes",
    )


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _code_utility(*, z_code: DiscreteState, work: DiscreteAction) -> FloatND:
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError("z_code must be int32")
    return work * (jnp.asarray(_NODES)[z_code] - 0.125)


def _value_utility(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return work * (z - 0.125)


def _entry_off_node(age: FloatND) -> ContinuousState:
    return _OFF_NODE + 0.0 * age


def _entry_on_node(age: FloatND) -> ContinuousState:
    return 0.0 * age


def _entry_zero() -> ContinuousState:
    return jnp.asarray(0.0)


def _entry_constant_off_node() -> ContinuousState:
    return jnp.asarray(_OFF_NODE)


def _project_zero(z: ContinuousState) -> ContinuousState:
    return 0.0 * z


def _project_off_node(z: ContinuousState) -> ContinuousState:
    return _OFF_NODE + 0.0 * z


def _project_one(z: ContinuousState) -> ContinuousState:
    return 1.0 + 0.0 * z


def _closed_by_code(z_code: DiscreteState) -> BoolND:
    return z_code < 0


def _has_policy_support(target_support: BoolND) -> BoolND:
    return target_support


def _landed_z(z: ContinuousState) -> ContinuousState:
    return z


def _off_middle_node(z_code: DiscreteState) -> BoolND:
    return z_code != 2


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _entry_model(
    *,
    on_node: bool,
    code: bool,
    enable_jit: bool,
    law: UserFunction | None = None,
) -> Model:
    entry = law or (_entry_on_node if on_node else _entry_off_node)
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _zero},
                state_transitions={"z": {"target": entry}},
            ),
            "target": Regime(
                states={"z": _process()},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _code_utility if code else _value_utility},
            ),
        },
        edges={"source": {"target": 0}},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=EntryId,
        enable_jit=enable_jit,
    )


def _gate_model(*, on_node: bool, enable_jit: bool) -> Model:
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _zero},
                state_transitions={
                    "z": {"landing": _entry_on_node if on_node else _entry_off_node}
                },
            ),
            "landing": Regime(
                states={"z": _process()},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _value_utility},
            ),
            "fallback": Regime(
                states={"z": _process()},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _value_utility},
            ),
        },
        edges={
            "source": Transition(
                targets={"landing": 0, "fallback": 0},
                law=ByAge(cases={0: {"landing": StochasticTransition(func=_prob_one)}}),
                gates={
                    "landing": Gate(
                        predicate=_closed_by_code,
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={"z": _project_zero}
                                )
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=GatedId,
        enable_jit=enable_jit,
    )


def _dissolution_model(*, on_node: bool, enable_jit: bool) -> Model:
    """Collective source landing in `a`, which has no policy at the middle node.

    The arrival decides the landing by `a`'s policy support; the off-node model's
    `f` route projects back into `a`, which reads `z_code`.
    """
    work = DiscreteGrid(category_class=Work)
    routes = {
        "f": StakeholderRoute(
            target_stakeholder="f",
            fallback=ProjectedRegimeValue(
                regime="a",
                stakeholder="f",
                projection={"z": _project_one if on_node else _project_off_node},
            ),
        ),
        "m": StakeholderRoute(
            target_stakeholder="m",
            fallback=ProjectedRegimeValue(regime="b", projection={"z": _project_one}),
        ),
    }
    arrivals = {
        "a": ArrivalDecision(
            predicate=_has_policy_support,
            reads={
                "target_support": ArrivalRead(
                    regime="a",
                    stakeholder="f",
                    kind="Participating",
                    field="support",
                    projection={"z": _landed_z},
                )
            },
            routes=routes,
        )
    }
    return Model(
        regimes={
            "source": Regime(
                functions={
                    "utility": CollectiveUtility(utilities={"f": _zero, "m": _zero})
                },
                state_transitions={"z": {"a": _entry_zero}},
            ),
            "a": Regime(
                states={"z": _process()},
                actions={"work": work},
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _code_utility, "m": _code_utility}
                    )
                },
                constraints={"feasible": _off_middle_node},
            ),
            "b": Regime(
                states={"z": _process()},
                actions={"work": work},
                functions={"utility": _value_utility},
            ),
        },
        edges={
            "source": Transition(
                law=ByAge(cases={0: {"a": StochasticTransition(func=_prob_one)}}),
                arrivals=arrivals,
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=DissolutionId,
        enable_jit=enable_jit,
    )


def _two_period_model(*, enable_jit: bool) -> Model:
    def reading() -> Regime:
        return Regime(
            states={"z": _process()},
            actions={"work": DiscreteGrid(category_class=Work)},
            functions={"utility": _code_utility},
        )

    return Model(
        regimes={"a": reading(), "b": reading()},
        edges={"a": {"b": 0}},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "a"},
        regime_id_class=ForeignId,
        enable_jit=enable_jit,
    )


def _source_params(model: Model) -> dict[str, dict[str, dict[str, float]]]:
    return {
        name: (
            {"koopmans_aggregator": {"discount_factor": 1.0}}
            if name == "source"
            else {}
        )
        for name in model.regime_names_to_ids
    }


def _simulate_from_source(*, model: Model, roles: tuple[str, ...] = ()) -> pd.DataFrame:
    params = _source_params(model)
    n_subjects = max(len(roles), 1)
    initial = {
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(
            n_subjects, model.regime_names_to_ids["source"], dtype=jnp.int32
        ),
    }
    if roles:
        initial["own_stakeholder"] = jnp.asarray(
            [model.stakeholder_names_to_ids[role] for role in roles], dtype=jnp.int32
        )
    return (
        model.simulate(
            params=params,
            solution=model.solve(params=params, log_level="off"),
            initial_conditions=initial,
            seed=0,
            log_level="off",
        )
        .to_dataframe()
        .query("period == 1")
    )


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_state_dependent_entry_off_node_is_refused_where_code_is_read(
    *,
    enable_jit: bool,
):
    model = _entry_model(on_node=False, code=True, enable_jit=enable_jit)
    with pytest.raises(InvalidStateTransitionError, match=r"off the process's nodes"):
        _simulate_from_source(model=model)


def test_constant_entry_off_node_is_refused_at_build_where_code_is_read():
    with pytest.raises(ModelInitializationError, match=r"Enter at one of its nodes"):
        _entry_model(
            on_node=False, code=True, enable_jit=False, law=_entry_constant_off_node
        )


def test_constant_entry_off_node_is_built_where_only_values_are_read():
    model = _entry_model(
        on_node=False, code=False, enable_jit=False, law=_entry_constant_off_node
    )
    assert tuple(model.regime_names_to_ids) == ("source", "target")


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_state_dependent_entry_on_node_is_held_where_code_is_read(*, enable_jit: bool):
    panel = _simulate_from_source(
        model=_entry_model(on_node=True, code=True, enable_jit=enable_jit)
    )
    assert panel[["regime_name", "z", "work"]].to_dict("list") == {
        "regime_name": ["target"],
        "z": [0.0],
        "work": ["off"],
    }


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_state_dependent_entry_off_node_is_held_where_only_values_are_read(
    *,
    enable_jit: bool,
):
    panel = _simulate_from_source(
        model=_entry_model(on_node=False, code=False, enable_jit=enable_jit)
    )
    assert panel[["regime_name", "z", "work"]].to_dict("list") == {
        "regime_name": ["target"],
        "z": [_OFF_NODE],
        "work": ["on"],
    }


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_gate_reading_code_at_off_node_draw_is_refused_though_it_closes(
    *,
    enable_jit: bool,
):
    model = _gate_model(on_node=False, enable_jit=enable_jit)
    with pytest.raises(InvalidStateTransitionError, match=r"off the process's nodes"):
        _simulate_from_source(model=model)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_gate_reading_code_at_node_draw_routes_to_fallback(*, enable_jit: bool):
    panel = _simulate_from_source(
        model=_gate_model(on_node=True, enable_jit=enable_jit)
    )
    assert panel[["regime_name", "z"]].to_dict("list") == {
        "regime_name": ["fallback"],
        "z": [0.0],
    }


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_dissolution_route_off_node_back_into_code_reader_is_refused(
    *,
    enable_jit: bool,
):
    model = _dissolution_model(on_node=False, enable_jit=enable_jit)
    with pytest.raises(InvalidStateTransitionError, match=r"off the process's nodes"):
        _simulate_from_source(model=model, roles=("f",))


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_dissolution_route_on_node_back_into_code_reader_decides_again(
    *,
    enable_jit: bool,
):
    panel = _simulate_from_source(
        model=_dissolution_model(on_node=True, enable_jit=enable_jit), roles=("f",)
    )
    assert panel[["regime_name", "z", "work"]].to_dict("list") == {
        "regime_name": ["a"],
        "z": [1.0],
        "work": ["on"],
    }


def _simulate_shock_path(*, later: float, enable_jit: bool) -> pd.DataFrame:
    model = _two_period_model(enable_jit=enable_jit)
    params = {"a": {"koopmans_aggregator": {"discount_factor": 1.0}}, "b": {}}
    return model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "z": jnp.array([0.0]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([model.regime_names_to_ids["a"]], dtype=jnp.int32),
        },
        shock_paths={"z": np.array([[0.0, later]])},
        seed=0,
        log_level="off",
    ).to_dataframe()


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_shock_path_off_node_in_a_later_period_is_refused(*, enable_jit: bool):
    with pytest.raises(InvalidSimulationInputError, match=r"period 1"):
        _simulate_shock_path(later=_OFF_NODE, enable_jit=enable_jit)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_shock_path_on_node_in_a_later_period_is_held(*, enable_jit: bool):
    panel = _simulate_shock_path(later=1.0, enable_jit=enable_jit)
    np.testing.assert_array_equal(panel["z"].to_numpy(), np.array([0.0, 1.0]))


@categorical(ordered=False)
class Pair:
    low: ScalarInt
    high: ScalarInt


def _position(z_code: DiscreteState) -> FloatND:
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError("z_code must be int32")
    return jnp.asarray(z_code, dtype=float)


def _value(z: ContinuousState) -> FloatND:
    return z


def _foreign_action_value(z: DiscreteAction) -> FloatND:
    return jnp.asarray(z, dtype=float)


def _foreign_state_code(z_code: DiscreteState) -> FloatND:
    return jnp.asarray(z_code, dtype=float)


def _foreign_regime(kind: str) -> Regime:
    if kind == "category_state":
        return Regime(
            states={"z": DiscreteGrid(category_class=Pair)},
            functions={"utility": _foreign_state_code},
        )
    grid = (
        DiscreteGrid(values=jnp.array([0.5, 1.5]))
        if kind == "value_action"
        else DiscreteGrid(category_class=Pair)
    )
    return Regime(actions={"z": grid}, functions={"utility": _foreign_action_value})


_FOREIGN_KINDS = ("category_action", "category_state", "value_action")


@pytest.mark.parametrize("kind", _FOREIGN_KINDS)
@pytest.mark.parametrize(
    ("reader", "expected"),
    [(_position, (0.0, 1.0, 2.0, 3.0, 4.0)), (_value, _NODES)],
    ids=["code", "value"],
)
def test_foreign_discrete_declaration_leaves_process_reads_local(
    *, kind: str, reader: UserFunction, expected: tuple[float, ...]
):
    model = Model(
        regimes={
            "a": Regime(states={"z": _process()}, functions={"utility": reader}),
            "b": _foreign_regime(kind),
        },
        edges={},
        ages=AgeGrid(exact_values=(0,)),
        initial_nodes={0: ("a", "b")},
        regime_id_class=ForeignId,
    )
    values = model.solve(params={"a": {}, "b": {}}, log_level="off").values[0]
    np.testing.assert_array_equal(np.asarray(values["a"]), np.asarray(expected))


def _refuse_off_node_producer(*, producer: str) -> None:
    """Simulate a jitted model whose `producer` puts 'z' at `1/4` where read by code."""
    if producer == "entry":
        _simulate_from_source(
            model=_entry_model(on_node=False, code=True, enable_jit=True)
        )
    elif producer == "gate":
        _simulate_from_source(model=_gate_model(on_node=False, enable_jit=True))
    else:
        _simulate_from_source(
            model=_dissolution_model(on_node=False, enable_jit=True), roles=("f",)
        )


@pytest.mark.parametrize("producer", ["entry", "gate", "dissolution"])
def test_off_node_refusal_names_the_code_argument_the_regime_reads(*, producer: str):
    with pytest.raises(InvalidStateTransitionError, match=r"position `z_code`"):
        _refuse_off_node_producer(producer=producer)


@pytest.mark.parametrize(
    ("held", "refused"),
    [((-1.0, 2.0, _OFF_NODE), True), ((-1.0, 2.0, 0.0), False)],
    ids=["off_node_holder", "every_holder_on_a_node"],
)
def test_held_process_code_check_refuses_only_an_off_node_holder(
    *, held: tuple[float, ...], refused: bool
):
    """Only holders count: the non-holder's `1/4` never trips the check."""
    states = {"z": jnp.asarray((*held, _OFF_NODE))}
    holders = jnp.asarray((True, True, True, False))
    try:
        fail_if_a_held_process_code_is_off_node(
            states=states,
            processes={"z": _process()},
            holders=holders,
            where="in regime 'target' at period 0",
            memory=None,
        )
    except InvalidStateTransitionError:
        raised = True
    else:
        raised = False
    assert raised is refused
