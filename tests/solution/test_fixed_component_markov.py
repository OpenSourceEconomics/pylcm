"""A Markov state declaring a fixed component solves as the hand-split model does.

The toy state `kind_health` codes (kind, health) as `2 * kind + health`; the law keeps
`kind` and moves `health`. Declaring `fixed_component` must give the value function
of the model in which `kind` is its own identity-law state, and carry the state as a
group axis and a within-group axis instead of one axis over every code.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from _lcm.regime_building.fixed_components import _restricted_law
from _lcm.regime_building.next_state import _DiscreteStochasticNextState
from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.params import UserMappingLeaf
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
    ScalarFloat,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class _KindHealth:
    k0_h0: ScalarInt
    k0_h1: ScalarInt
    k1_h0: ScalarInt
    k1_h1: ScalarInt


@categorical(ordered=False)
class _Kind:
    k0: ScalarInt
    k1: ScalarInt


@categorical(ordered=False)
class _Health:
    h0: ScalarInt
    h1: ScalarInt


_HEALTH_LAW = jnp.array([[[0.9, 0.1], [0.3, 0.7]], [[0.6, 0.4], [0.2, 0.8]]])


def _utility(
    *,
    consumption: ContinuousAction,
    wealth: ContinuousState,
    kind_health: DiscreteState,
) -> FloatND:
    return jnp.log(consumption) + 0.1 * kind_health + 0.01 * wealth


def _next_kind_health(kind_health: DiscreteState) -> FloatND:
    kind, health = kind_health // 2, kind_health % 2
    within = _HEALTH_LAW[kind, health]
    return jnp.where(jnp.arange(4) // 2 == kind, within[jnp.arange(4) % 2], 0.0)


def _code_view(kind_health: DiscreteState) -> DiscreteState:
    return kind_health


def _nested_code_view(code_view: DiscreteState) -> DiscreteState:
    return code_view


def _next_kind_health_via_helper(code_view: DiscreteState) -> FloatND:
    return _next_kind_health(code_view)


def _next_kind_health_via_chain(nested_code_view: DiscreteState) -> FloatND:
    return _next_kind_health(nested_code_view)


def _constant_code_probabilities() -> FloatND:
    return jnp.full(4, 0.25)


def _next_health(*, kind: DiscreteState, health: DiscreteState) -> FloatND:
    return _HEALTH_LAW[kind, health]


def _kind_health(*, kind: DiscreteState, health: DiscreteState) -> DiscreteState:
    return 2 * kind + health


def _feasible(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 2


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 2, _RegimeId.alive, _RegimeId.dead)


def _model(
    *,
    factored: bool,
    fixed_component: tuple[int, ...] | None = (0, 0, 1, 1),
    sharded: bool = False,
    law_dependency: Literal["direct", "helper", "chain"] = "direct",
    enable_jit: bool = True,
) -> Model:
    model_states: dict[str, DiscreteGrid] = {}
    model_laws = {}
    if factored:
        states = {"kind_health": DiscreteGrid(_KindHealth)}
        laws = {
            "kind_health": MarkovTransition(
                {
                    "direct": _next_kind_health,
                    "helper": _next_kind_health_via_helper,
                    "chain": _next_kind_health_via_chain,
                }[law_dependency],
                fixed_component=fixed_component,
            )
        }
        functions = {}
        if law_dependency != "direct":
            functions["code_view"] = _code_view
        if law_dependency == "chain":
            functions["nested_code_view"] = _nested_code_view
    else:
        states = {"health": DiscreteGrid(_Health)}
        laws = {"health": MarkovTransition(_next_health)}
        model_states = {"kind": DiscreteGrid(_Kind)}
        model_laws = {"kind": fixed_transition("kind")}
        functions = {"kind_health": _kind_health}
    return Model(
        regimes={
            "alive": Regime(
                active=lambda age: age < 3,
                transition=_next_regime,
                states={
                    "wealth": LinSpacedGrid(start=1, stop=10, n_points=5),
                    **states,
                },
                actions={"consumption": LinSpacedGrid(start=1, stop=3, n_points=3)},
                functions={"utility": _utility, **functions},
                constraints={"feasible": _feasible},
                state_transitions={
                    "wealth": _next_wealth,
                    **laws,
                },
            ),
            "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
        },
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
        states=model_states,
        state_transitions=model_laws,
        execution_config=ExecutionConfig(
            sharded_states=(("kind_health_fixed" if factored else "kind"),)
            if sharded
            else ()
        ),
    )


def _values(model: Model) -> dict:
    solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    return {
        (period, regime): np.asarray(v)
        for period, by_regime in solution.values.items()
        for regime, v in by_regime.items()
    }


def test_fixed_component_carries_group_and_position_as_separate_axes():
    """The alive value function has a position axis and a group axis, not 4 codes."""
    values = _values(_model(factored=True))
    assert values[(0, "alive")].shape == (2, 2, 5)


def test_fixed_component_solve_equals_the_hand_split_model():
    """Declaring the fixed component reproduces the hand-split value functions."""
    factored, split = _values(_model(factored=True)), _values(_model(factored=False))
    assert all(np.array_equal(factored[key], split[key]) for key in split)


def test_fixed_component_rejects_unequal_groups():
    """Groups of different sizes cannot share one within-group axis."""
    with pytest.raises(RegimeInitializationError, match="equal size"):
        _model(factored=True, fixed_component=(0, 0, 0, 1))


def test_fixed_component_preserves_original_lottery_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear expectation retains every original code slot, including zero weights."""
    module = importlib.import_module("_lcm.regime_building.Q_and_F")
    original = module.zero_safe_average
    observed: set[tuple[int, ...]] = set()

    def record_support(
        *,
        a: FloatND,
        weights: FloatND,
        shifts: IntND | None,
        axis: int | None = None,
    ) -> FloatND:
        observed.add(weights.shape)
        return original(a=a, weights=weights, shifts=shifts, axis=axis)

    monkeypatch.setattr(module, "zero_safe_average", record_support)
    _model(factored=True, enable_jit=False).solve(
        params={"discount_factor": 0.95}, log_level="off"
    )
    assert observed == {(4,)}


def test_fixed_component_is_shardable_like_the_hand_split_model():
    """Naming the fixed component in `sharded_states` solves as the sharded split."""
    factored = _values(_model(factored=True, sharded=True))
    split = _values(_model(factored=False, sharded=True))
    assert all(np.array_equal(factored[key], split[key]) for key in split)


def _initial(*, factored: bool, as_frame: bool) -> dict | pd.DataFrame:
    code = np.arange(8) % 4
    common = {"wealth": np.linspace(1.0, 10.0, 8), "age": np.zeros(8)}
    if as_frame:
        kind_health = np.array(["k0_h0", "k0_h1", "k1_h0", "k1_h1"])[code]
        parts = (
            {"kind_health": kind_health}
            if factored
            else {
                "kind": np.array(["k0", "k1"])[code // 2],
                "health": np.array(["h0", "h1"])[code % 2],
            }
        )
        return pd.DataFrame(common | parts | {"regime_name": ["alive"] * 8})
    parts = (
        {"kind_health": code} if factored else {"kind": code // 2, "health": code % 2}
    )
    return {
        name: jnp.asarray(value)
        for name, value in (common | parts | {"regime_id": np.zeros(8, int)}).items()
    }


def _simulated_value(
    *,
    factored: bool,
    as_frame: bool,
    law_dependency: Literal["direct", "helper", "chain"] = "direct",
    enable_jit: bool = True,
) -> np.ndarray:
    model = _model(
        factored=factored, law_dependency=law_dependency, enable_jit=enable_jit
    )
    params = {"discount_factor": 0.95}
    result = model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions=_initial(factored=factored, as_frame=as_frame),
        seed=1,
        log_level="off",
    )
    return np.asarray(result.to_dataframe()["value"])


@pytest.mark.parametrize("as_frame", [False, True])
def test_fixed_component_simulates_from_the_declared_code(*, as_frame):
    """Initial conditions name the declared state and simulate as the hand split."""
    np.testing.assert_array_equal(
        _simulated_value(factored=True, as_frame=as_frame),
        _simulated_value(factored=False, as_frame=as_frame),
    )


def _code_utility(kind_health: DiscreteState) -> FloatND:
    return 1.0 * kind_health


def _two_code_utility(*, kind_health: DiscreteState, other: DiscreteState) -> FloatND:
    return 1.0 * kind_health + 10.0 * other


@pytest.mark.parametrize("form", ["local", "model", "per_target", "phased"])
@pytest.mark.parametrize("terminal_carries", [False, True])
def test_fixed_component_lowering_covers_declarations_and_terminal(
    *, form: str, terminal_carries: bool
) -> None:
    """Each declaration lowers every carrier and preserves public initial values."""
    law = MarkovTransition(_next_kind_health, fixed_component=(0, 0, 1, 1))
    grid = DiscreteGrid(_KindHealth)
    local_states = {"kind_health": grid}
    local_laws = {"kind_health": law}
    model_states, model_laws = {}, {}
    if form == "model":
        model_states, model_laws = local_states, local_laws
        local_states, local_laws = {}, {}
    elif form == "per_target":
        local_laws = {"kind_health": {"dead": law}}
    elif form == "phased":
        local_laws = {"kind_health": Phased(solve=law, simulate=law)}
    if not terminal_carries and form == "per_target":
        local_laws = {"kind_health": {"alive": law}}
    alive = Regime(
        active=(lambda age: age == 0) if terminal_carries else (lambda age: age < 2),
        transition={"dead": MarkovTransition(lambda: jnp.asarray(1.0))}
        if terminal_carries
        else _next_regime,
        states=local_states,
        state_transitions=local_laws,
        functions={"utility": _code_utility},
    )
    dead = Regime(
        transition=None,
        states={"kind_health": grid} if terminal_carries and form != "model" else {},
        functions={"utility": _code_utility}
        if terminal_carries
        else {"utility": lambda: 0.0},
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        states=model_states,
        state_transitions=model_laws,
    )
    assert {"kind_health_rest", "kind_health_fixed"} <= model.user_regimes[
        "alive"
    ].states.keys()
    if terminal_carries:
        assert {"kind_health_rest", "kind_health_fixed"} <= model.user_regimes[
            "dead"
        ].states.keys()


@pytest.mark.parametrize("inactive", [None, np.nan, "irrelevant"])
@pytest.mark.parametrize("reverse", [False, True])
def test_fixed_component_initial_labels_are_scoped_to_the_initial_regime(
    *, inactive: object, reverse: bool
) -> None:
    """Rows in a state-absent regime do not require that state's label."""
    model = _model(factored=True)
    frame = pd.DataFrame(
        {
            "regime_name": ["alive", "dead"],
            "age": [0.0, 0.0],
            "wealth": [2.0, np.nan],
            "kind_health": ["k0_h1", inactive],
        }
    )
    if reverse:
        frame = frame.iloc[::-1].reset_index(drop=True)
    original = frame.copy(deep=True)
    result = model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions=frame,
        seed=1,
        log_level="off",
    ).to_dataframe()
    pd.testing.assert_frame_equal(frame, original)
    reference_frame = frame.drop(columns="kind_health").assign(
        kind=frame["regime_name"].map({"alive": "k0"}),
        health=frame["regime_name"].map({"alive": "h1"}),
    )
    reference = (
        _model(factored=False)
        .simulate(
            params={"discount_factor": 0.95},
            initial_conditions=reference_frame,
            seed=1,
            log_level="off",
        )
        .to_dataframe()
    )
    np.testing.assert_array_equal(result["value"], reference["value"])


def test_fixed_component_rejects_an_occupied_model_state_name():
    """A user state cannot alias the generated fixed coordinate."""
    with pytest.raises(RegimeInitializationError, match="kind_health_fixed"):
        Model(
            regimes={
                "alive": Regime(
                    active=lambda age: age < 1,
                    transition=_next_regime,
                    states={"kind_health": DiscreteGrid(_KindHealth)},
                    state_transitions={
                        "kind_health": MarkovTransition(
                            _next_kind_health, fixed_component=(0, 0, 1, 1)
                        )
                    },
                    functions={
                        "utility": lambda kind_health, kind_health_fixed: (
                            1.0 * kind_health + kind_health_fixed
                        )
                    },
                ),
                "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
            },
            states={"kind_health_fixed": DiscreteGrid(_Kind)},
            state_transitions={
                "kind_health_fixed": fixed_transition("kind_health_fixed")
            },
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=_RegimeId,
        )


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("state_absent", [False, True])
def test_fixed_component_public_validation_accepts_original_observations(
    *,
    as_frame: bool,
    state_absent: bool,
) -> None:
    """Validation and feasibility share the original-state input contract."""
    model = _model(factored=True)
    if as_frame:
        initial = pd.DataFrame(
            {
                "regime_name": ["dead"] if state_absent else ["alive", "dead"],
                "age": [0.0] if state_absent else [0.0, 0.0],
            }
        )
        if not state_absent:
            initial["wealth"] = [2.0, np.nan]
            initial["kind_health"] = ["k0_h1", "irrelevant"]
    else:
        initial = {
            "regime_id": jnp.array([1] if state_absent else [0, 1]),
            "age": jnp.zeros(1 if state_absent else 2),
        }
        if not state_absent:
            initial.update(
                wealth=jnp.array([2.0, np.nan]),
                kind_health=jnp.array([1, np.iinfo(np.int32).min]),
            )
    model.validate_initial_conditions(
        initial_conditions=initial, params={"discount_factor": 0.95}
    )
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(
            initial_conditions=initial, params={"discount_factor": 0.95}
        ),
        [True] if state_absent else [True, True],
    )


def _next_kind_health_with_persistence(
    *,
    kind_health: DiscreteState,
    persistence: float,
) -> FloatND:
    return persistence * (jnp.arange(4) == kind_health) + (
        1 - persistence
    ) * _next_kind_health(kind_health)


def _next_kind_health_with_payload(
    *,
    kind_health: DiscreteState,
    payload: UserMappingLeaf,
) -> FloatND:
    inner = cast("UserMappingLeaf", payload.data["next_kind_health"])
    persistence = cast("ScalarFloat", inner.data["next_kind_health"])
    return persistence * (jnp.arange(4) == kind_health) + (
        1 - persistence
    ) * _next_kind_health(kind_health)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("per_target", [False, True])
@pytest.mark.parametrize("payload", [False, True])
def test_fixed_component_transition_parameters_keep_their_public_names(
    *,
    fixed: bool,
    per_target: bool,
    payload: bool,
) -> None:
    """Original transition names bind fixed and runtime parameters at either scope."""
    law = MarkovTransition(
        _next_kind_health_with_payload
        if payload
        else _next_kind_health_with_persistence,
        fixed_component=(0, 0, 1, 1),
    )
    transition_params = {
        "next_kind_health": {
            "payload": UserMappingLeaf(
                {"next_kind_health": UserMappingLeaf({"next_kind_health": 0.5})}
            )
        }
        if payload
        else {"persistence": 0.5}
    }
    named_params = {
        "alive": {"dead": transition_params} if per_target else transition_params
    }
    model = Model(
        regimes={
            "alive": Regime(
                active=lambda age: age == 0,
                transition={"dead": MarkovTransition(lambda: jnp.asarray(1.0))},
                states={"kind_health": DiscreteGrid(_KindHealth)},
                functions={"utility": _code_utility},
                state_transitions={"kind_health": {"dead": law} if per_target else law},
            ),
            "dead": Regime(
                transition=None,
                states={"kind_health": DiscreteGrid(_KindHealth)},
                functions={"utility": _code_utility},
            ),
        },
        fixed_params=named_params if fixed else {},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )
    params = {"discount_factor": 0.5} | ({} if fixed else named_params)
    panel = (
        model.simulate(
            params=params,
            initial_conditions={
                "regime_id": jnp.zeros(4, dtype=jnp.int32),
                "age": jnp.zeros(4),
                "kind_health": jnp.arange(4),
            },
            seed=1,
            log_level="off",
        )
        .to_dataframe()
        .reset_index()
    )
    np.testing.assert_allclose(
        panel.loc[panel["period"] == 0, "value"],
        [0.025, 1.425, 3.1, 4.45],
        rtol=10**-DECIMAL_PRECISION,
    )


@categorical(ordered=False)
class _CarrierRegimeId:
    first: ScalarInt
    middle: ScalarInt
    last: ScalarInt


def _interleaved_kind_health(kind_health: DiscreteState) -> FloatND:
    return 0.75 * (jnp.arange(4) == kind_health) + 0.25 * (
        jnp.arange(4) == (kind_health ^ 2)
    )


def _interleaved_other(other: DiscreteState) -> FloatND:
    return 0.75 * (jnp.arange(4) == other) + 0.25 * (jnp.arange(4) == (other ^ 2))


@pytest.mark.parametrize(
    ("enable_jit", "sharded"), [(False, False), (True, False), (True, True)]
)
def test_two_fixed_components_cross_adjacent_carriers_and_terminal(
    *,
    enable_jit: bool,
    sharded: bool,
) -> None:
    """Independent interleaved chains preserve values and original initial codes."""
    states = {
        "kind_health": DiscreteGrid(_KindHealth),
        "other": DiscreteGrid(_KindHealth),
    }
    laws = {
        "kind_health": MarkovTransition(
            _interleaved_kind_health, fixed_component=(0, 1, 0, 1)
        ),
        "other": MarkovTransition(_interleaved_other, fixed_component=(0, 1, 0, 1)),
    }
    functions = {"utility": _two_code_utility}
    model = Model(
        regimes={
            "first": Regime(
                active=lambda age: age == 0,
                transition={"middle": MarkovTransition(lambda: jnp.asarray(1.0))},
                states=states,
                state_transitions=laws,
                functions=functions,
            ),
            "middle": Regime(
                active=lambda age: age == 1,
                transition={"last": MarkovTransition(lambda: jnp.asarray(1.0))},
                states=states,
                state_transitions=laws,
                functions=functions,
            ),
            "last": Regime(transition=None, states=states, functions=functions),
        },
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_CarrierRegimeId,
        enable_jit=enable_jit,
        execution_config=ExecutionConfig(
            sharded_states=("kind_health_fixed",) if sharded else ()
        ),
    )
    first = np.repeat(np.arange(4), 4)
    other = np.tile(np.arange(4), 4)
    panel = (
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions={
                "regime_id": jnp.zeros(16, dtype=jnp.int32),
                "age": jnp.zeros(16),
                "kind_health": first,
                "other": other,
            },
            seed=42,
            log_level="off",
        )
        .to_dataframe(additional_targets=["kind_health", "other"])
        .reset_index()
    )
    initial = panel.loc[panel["period"] == 0].sort_values("subject_id")
    # Two literal Bernoulli steps give P²(stay)=5/8, P²(switch)=3/8.
    expected_first = 1.53125 * first + 0.21875 * (first ^ 2)
    expected_other = 1.53125 * other + 0.21875 * (other ^ 2)
    np.testing.assert_allclose(
        initial["value"],
        expected_first + 10 * expected_other,
        rtol=10**-DECIMAL_PRECISION,
    )
    np.testing.assert_array_equal(initial["kind_health"], first)
    np.testing.assert_array_equal(initial["other"], other)


@pytest.mark.parametrize("reset", [False, True])
def test_fixed_component_rejects_incoherent_handoff_laws(*, reset: bool) -> None:
    """Every carrier must preserve the same declared group identity."""
    first_law = MarkovTransition(_interleaved_kind_health, fixed_component=(0, 1, 0, 1))
    incompatible = (
        (lambda: jnp.asarray(0, dtype=jnp.int32))
        if reset
        else MarkovTransition(_next_kind_health, fixed_component=(0, 0, 1, 1))
    )
    states = {"kind_health": DiscreteGrid(_KindHealth)}
    functions = {"utility": _code_utility}
    with pytest.raises(RegimeInitializationError, match=r"group|fixed_component"):
        Model(
            regimes={
                "first": Regime(
                    active=lambda age: age == 0,
                    transition={"middle": MarkovTransition(lambda: jnp.asarray(1.0))},
                    states=states,
                    state_transitions={"kind_health": first_law},
                    functions=functions,
                ),
                "middle": Regime(
                    active=lambda age: age == 1,
                    transition={"last": MarkovTransition(lambda: jnp.asarray(1.0))},
                    states=states,
                    state_transitions={"kind_health": incompatible},
                    functions=functions,
                ),
                "last": Regime(transition=None, states=states, functions=functions),
            },
            ages=AgeGrid(start=0, stop=3, step="Y"),
            regime_id_class=_CarrierRegimeId,
        )


@pytest.mark.parametrize("slot", ["constraints", "derived_categoricals"])
def test_fixed_component_generated_names_do_not_shadow_regime_slots(
    *, slot: str
) -> None:
    """Generated state names cannot occupy user constraint or categorical slots."""
    with pytest.raises(RegimeInitializationError, match="kind_health_fixed"):
        Model(
            regimes={
                "alive": Regime(
                    active=lambda age: age < 1,
                    transition=_next_regime,
                    states={"kind_health": DiscreteGrid(_KindHealth)},
                    state_transitions={
                        "kind_health": MarkovTransition(
                            _next_kind_health, fixed_component=(0, 0, 1, 1)
                        )
                    },
                    functions={"utility": _code_utility},
                    constraints={"kind_health_fixed": lambda: jnp.ones((), dtype=bool)}
                    if slot == "constraints"
                    else {},
                    derived_categoricals={"kind_health_fixed": DiscreteGrid(_Kind)}
                    if slot == "derived_categoricals"
                    else {},
                ),
                "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
            },
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=_RegimeId,
        )


def test_fixed_component_ignores_object_codes_for_a_wholly_absent_cohort() -> None:
    """An irrelevant mapping column is never converted to categorical codes."""
    model = _model(factored=True)
    initial = {
        "regime_id": np.array([1]),
        "age": np.array([0.0]),
        "kind_health": np.array(["irrelevant"], dtype=object),
    }
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(
            initial_conditions=initial, params={"discount_factor": 0.95}
        ),
        [True],
    )


def test_fixed_component_rejects_cross_state_identity_at_model_level() -> None:
    """A law named for another state cannot establish this state's group identity."""
    with pytest.raises(RegimeInitializationError, match="names must match"):
        Model(
            regimes={
                "alive": Regime(
                    active=lambda age: age < 2,
                    transition=_next_regime,
                    states={"kind_health": DiscreteGrid(_KindHealth)},
                    functions={"utility": _code_utility},
                ),
                "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
            },
            state_transitions={
                "kind_health": Phased(
                    solve=MarkovTransition(
                        _next_kind_health, fixed_component=(0, 0, 1, 1)
                    ),
                    simulate=fixed_transition("other"),
                )
            },
            ages=AgeGrid(start=0, stop=3, step="Y"),
            regime_id_class=_RegimeId,
        )


def test_fixed_component_eager_sharding_with_a_state_absent_terminal() -> None:
    """Eager sharded reconstruction preserves values before dropping the state."""
    model = Model(
        regimes={
            "alive": Regime(
                active=lambda age: age == 0,
                transition={"dead": MarkovTransition(lambda: jnp.asarray(1.0))},
                states={"kind_health": DiscreteGrid(_KindHealth)},
                state_transitions={
                    "kind_health": MarkovTransition(
                        _next_kind_health, fixed_component=(0, 0, 1, 1)
                    )
                },
                functions={"utility": _code_utility},
            ),
            "dead": Regime(transition=None, functions={"utility": lambda: 0.0}),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        enable_jit=False,
        execution_config=ExecutionConfig(sharded_states=("kind_health_fixed",)),
    )
    panel = (
        model.simulate(
            params={"discount_factor": 0.5},
            initial_conditions={
                "regime_id": jnp.zeros(4, dtype=jnp.int32),
                "age": jnp.zeros(4),
                "kind_health": jnp.arange(4),
            },
            seed=1,
            log_level="off",
        )
        .to_dataframe()
        .reset_index()
    )
    initial = panel.loc[panel["period"] == 0].sort_values("subject_id")
    np.testing.assert_array_equal(initial["value"], [0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize("law_dependency", ["helper", "chain"])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_fixed_component_law_preserves_indirect_state_dependencies(
    *, law_dependency: Literal["helper", "chain"], enable_jit: bool
) -> None:
    """A law reading the state through helpers matches the hand-split values."""
    observed = _simulated_value(
        factored=True,
        as_frame=False,
        law_dependency=law_dependency,
        enable_jit=enable_jit,
    )
    expected = _simulated_value(factored=False, as_frame=False, enable_jit=enable_jit)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_fixed_component_constant_law_preserves_single_group_probabilities(
    *, enable_jit: bool
) -> None:
    """A constant law accepts the adapter's state without forwarding it."""
    law = _restricted_law(
        func=_constant_code_probabilities,
        state_name="kind_health",
        fixed_of_code=np.zeros(4, dtype=np.int32),
        code_by_parts=np.arange(4, dtype=np.int32).reshape(4, 1),
    )
    if enable_jit:
        law = jax.jit(law)
    for code in range(4):
        np.testing.assert_array_equal(
            law(kind_health=jnp.asarray(code)), np.full(4, 0.25)
        )


@pytest.mark.parametrize("enable_jit", [False, True])
def test_fixed_component_simulation_preserves_original_sampling_support(
    *, monkeypatch: pytest.MonkeyPatch, enable_jit: bool
) -> None:
    """Simulation samples original code slots and returns their within-group code."""
    observed: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    original = _DiscreteStochasticNextState.__call__

    def record_support(
        self: _DiscreteStochasticNextState, **kwargs: FloatND
    ) -> DiscreteState:
        if self.qname == "alive__next_kind_health_rest":
            observed.add((self.labels.shape, kwargs[f"weight_{self.qname}"].shape))
        return original(self, **kwargs)

    monkeypatch.setattr(_DiscreteStochasticNextState, "__call__", record_support)
    panels = []
    for annotation in (None, (0, 0, 1, 1)):
        model = _model(factored=True, fixed_component=annotation, enable_jit=enable_jit)
        params = {"discount_factor": 0.95}
        result = model.simulate(
            params=params,
            solution=model.solve(params=params, log_level="off"),
            initial_conditions=_initial(factored=True, as_frame=False),
            seed=1,
            log_level="off",
        )
        frame = result.to_dataframe()
        frame = frame.loc[frame["regime_name"] == "alive"].sort_values(
            ["period", "subject_id"]
        )
        if annotation is None:
            codes = frame["kind_health"].cat.codes.to_numpy()
        else:
            codes = (
                2 * frame["kind_health_fixed"].cat.codes.to_numpy()
                + frame["kind_health_rest"].cat.codes.to_numpy()
            )
        panels.append(codes)
    np.testing.assert_array_equal(panels[0], panels[1])
    assert observed == {((4,), (4,))}


@categorical(ordered=False)
class _NextOutputRegimeId:
    source: ScalarInt
    end: ScalarInt


@categorical(ordered=False)
class _NextOutputCode:
    c0: ScalarInt
    c1: ScalarInt
    c2: ScalarInt
    c3: ScalarInt


def _next_output_utility(*, s: DiscreteState, y: DiscreteState) -> FloatND:
    return 1.0 * (s + y)


def _next_output_law(*, s: DiscreteState) -> FloatND:
    groups = jnp.array([0, 0, 1, 1])
    return jnp.where(groups == groups[s], 0.5, 0.0)


def _next_output_copy(*, next_s: DiscreteState) -> DiscreteState:
    return next_s


def _next_output_from_landing(*, landing: DiscreteState) -> DiscreteState:
    return landing


def _next_output_to_end() -> ScalarFloat:
    return jnp.asarray(1.0)


def _next_output_parameter_leaves(
    *, value: object, prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return []
    found: list[tuple[str, ...]] = []
    for name, child in value.items():
        path = (*prefix, str(name))
        if str(name) == "next_s" and not isinstance(child, Mapping):
            found.append(path)
        found.extend(_next_output_parameter_leaves(value=child, prefix=path))
    return found


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("through_helper", [False, True])
def test_fixed_component_preserves_a_transition_reading_its_next_code(
    *, annotated: bool, through_helper: bool
) -> None:
    """A deterministic transition shares the original code of the same draw."""
    grid = DiscreteGrid(_NextOutputCode)
    functions: dict[str, Callable[..., object]] = {"utility": _next_output_utility}
    if through_helper:
        functions["landing"] = _next_output_copy
    model = Model(
        regimes={
            "source": Regime(
                active=lambda age: age == 0,
                transition={"end": MarkovTransition(_next_output_to_end)},
                states={"s": grid, "y": grid},
                state_transitions={
                    "s": MarkovTransition(
                        _next_output_law,
                        fixed_component=(0, 0, 1, 1) if annotated else None,
                    ),
                    "y": _next_output_from_landing
                    if through_helper
                    else _next_output_copy,
                },
                functions=functions,
            ),
            "end": Regime(
                transition=None,
                states={"s": grid, "y": grid},
                functions={"utility": _next_output_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_NextOutputRegimeId,
    )
    assert _next_output_parameter_leaves(value=model.get_params_template()) == []
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    s = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 4)
    y = jnp.tile(jnp.arange(4, dtype=jnp.int32), 4)
    panel = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "regime_id": jnp.zeros(16, dtype=jnp.int32),
            "age": jnp.zeros(16),
            "s": s,
            "y": y,
        },
        seed=123,
        log_level="off",
    ).to_dataframe(use_labels=False)
    if "period" not in panel.columns:
        panel = panel.reset_index()
    first = panel.loc[panel["period"] == 0].sort_values("subject_id")
    expected = np.repeat([0.5, 1.5, 4.5, 5.5], 4) + np.tile(np.arange(4), 4)
    np.testing.assert_array_equal(first["value"].to_numpy(), expected)
    last = panel.loc[panel["period"] == 1].sort_values("subject_id")
    next_s = last["s_rest"] + 2 * last["s_fixed"] if annotated else last["s"]
    np.testing.assert_array_equal(last["y"].to_numpy(), next_s.to_numpy())
