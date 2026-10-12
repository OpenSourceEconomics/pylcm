"""A state a regime hands to a reachable target that reads it is used there.

`work` declares `skill` and reads nothing of it; it only hands the value on with
`fixed_transition`. The hand-over is a use when a target `work` can transition
into keeps `skill`, because that target's value function has a `skill` axis that
the source has to fill. Hand-overs chain: a regime that keeps the state only to
hand it on is itself a target that keeps it. A hand-over toward targets that do
not keep the state, or toward no reachable target at all, is no use.
"""

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.model_graph import bind_edge_laws
from _lcm.regime_building.broadcast import compute_kept_variables
from lcm import (
    AgeGrid,
    AgeSpecializedFunction,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime, StateTransitionEntry
from lcm.transition import ModelEdges
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=1.0, n_points=3)


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt
    gone: ScalarInt


@categorical(ordered=True)
class _Skill:
    low: ScalarInt
    high: ScalarInt


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_with_skill(*, consumption: ContinuousAction, skill: ScalarInt) -> FloatND:
    return jnp.log(consumption) + 0.1 * skill


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _bequest_with_skill(*, wealth: ContinuousState, skill: ScalarInt) -> FloatND:
    return jnp.log(wealth) + 0.1 * skill


def _carry_skill(skill: ScalarInt) -> ScalarInt:
    return skill


def _reset_skill() -> ScalarInt:
    return jnp.asarray(_Skill.low)


def _utility_with_skill_action(
    *, consumption: ContinuousAction, skill: ScalarInt
) -> FloatND:
    return jnp.log(consumption) + 0.1 * skill


def _utility_built_from_age_one(age: float) -> Callable[..., FloatND]:
    if age < 1:
        msg = f"no utility before age 1, asked for age {age}"
        raise ValueError(msg)
    return _utility


def _alive(
    *,
    utility: Callable[..., FloatND]
    | Phased[Callable[..., FloatND], AgeSpecializedFunction] = _utility,
    with_skill: bool = True,
    skill_law: StateTransitionEntry | None = None,
) -> Regime:
    return Regime(
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID}
        | ({"skill": DiscreteGrid(category_class=_Skill)} if with_skill else {}),
        state_transitions={"wealth": _next_wealth}
        | ({"skill": skill_law or fixed_transition("skill")} if with_skill else {}),
        functions={"utility": utility},
    )


def _terminal(*, reads_skill: bool) -> Regime:
    return Regime(
        states={"wealth": WEALTH_GRID}
        | ({"skill": DiscreteGrid(category_class=_Skill)} if reads_skill else {}),
        functions={"utility": _bequest_with_skill if reads_skill else _bequest},
    )


def _model(
    *,
    regimes: Mapping[str, Regime],
    edges: ModelEdges,
    states: Mapping[str, DiscreteGrid] | None = None,
    state_transitions: Mapping[str, StateTransitionEntry] | None = None,
    initial_nodes: Mapping[int, str | tuple[str, ...]] | None = None,
) -> Model:
    return Model(
        regimes=dict(regimes),
        edges=edges,
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes=dict(initial_nodes or {0: "work"}),
        states=dict(states or {}),
        state_transitions=dict(state_transitions or {}),
    )


def _work_retired_dead() -> dict[str, Regime]:
    return {
        "work": _alive(),
        "retired": _alive(utility=_utility_with_skill),
        "dead": _terminal(reads_skill=False),
        "gone": _terminal(reads_skill=False),
    }


CHAIN_EDGES = {"work": {"retired": 0}, "retired": {"dead": 1}}
SIMULATION_ONLY_EDGES = Phased(
    solve={"work": {"gone": 0}, "retired": {"dead": 1}},
    simulate={"work": {"retired": 0}, "retired": {"dead": 1}},
)


def _model_level_skill() -> Model:
    return _model(
        regimes={
            "work": _alive(with_skill=False),
            "retired": _alive(utility=_utility_with_skill, with_skill=False),
            "dead": _terminal(reads_skill=False),
            "gone": _terminal(reads_skill=False),
        },
        edges=CHAIN_EDGES,
        states={"skill": DiscreteGrid(category_class=_Skill)},
        state_transitions={"skill": fixed_transition("skill")},
    )


def _work_values(model: Model) -> np.ndarray:
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    V = np.asarray(solution.values[0]["work"])
    order = model.state_names(regime_name="work")
    return np.transpose(V, [order.index(name) for name in ("wealth", "skill")])


def test_regime_level_state_carried_to_a_reading_target_is_used() -> None:
    model = _model(regimes=_work_retired_dead(), edges=CHAIN_EDGES)

    assert "skill" in model.user_regimes["work"].states


def test_model_level_state_carried_to_a_reading_target_is_kept() -> None:
    model = _model_level_skill()

    assert "skill" in model.user_regimes["work"].states


def test_regime_level_hand_over_values_match_the_model_level_declaration() -> None:
    regime_level = _work_values(_model(regimes=_work_retired_dead(), edges=CHAIN_EDGES))
    model_level = _work_values(_model_level_skill())

    np.testing.assert_array_almost_equal(
        regime_level, model_level, decimal=DECIMAL_PRECISION
    )


def test_the_carrier_value_varies_with_the_handed_over_state() -> None:
    V = _work_values(_model(regimes=_work_retired_dead(), edges=CHAIN_EDGES))

    assert np.all(V[:, 1] > V[:, 0])


def test_a_chain_of_hand_overs_to_a_reading_terminal_regime_is_used() -> None:
    model = _model(
        regimes={
            "work": _alive(),
            "retired": _alive(),
            "dead": _terminal(reads_skill=True),
            "gone": _terminal(reads_skill=False),
        },
        edges=CHAIN_EDGES,
    )

    assert "skill" in model.user_regimes["work"].states


def test_hand_over_chain_cut_before_the_reader_is_rejected_in_every_carrier() -> None:
    with pytest.raises(ModelInitializationError) as error:
        _model(
            regimes={
                "work": _alive(),
                "retired": _alive(),
                "dead": _terminal(reads_skill=True),
                "gone": _terminal(reads_skill=False),
            },
            edges={"work": {"retired": 0}, "retired": {"gone": 1}},
        )

    message = str(error.value)
    assert "never used in regime 'work'" in message
    assert "never used in regime 'retired'" in message


def test_hand_over_toward_a_target_that_does_not_keep_the_state_is_rejected() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"never used in regime 'work': state \['skill'\]",
    ):
        _model(
            regimes=_work_retired_dead() | {"retired": _alive(with_skill=False)},
            edges=CHAIN_EDGES,
        )


def test_hand_over_without_an_edge_to_the_reader_is_rejected() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"never used in regime 'work': state \['skill'\]",
    ):
        _model(
            regimes=_work_retired_dead(),
            edges={"work": {"gone": 0}, "retired": {"dead": 1}},
        )


def test_hand_over_to_a_target_with_a_same_named_action_is_rejected() -> None:
    retired_with_skill_action = Regime(
        actions={
            "consumption": CONSUMPTION_GRID,
            "skill": DiscreteGrid(category_class=_Skill),
        },
        states={"wealth": WEALTH_GRID},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility_with_skill_action},
    )
    with pytest.raises(
        ModelInitializationError,
        match=r"never used in regime 'work': state \['skill'\]",
    ):
        _model(
            regimes=_work_retired_dead() | {"retired": retired_with_skill_action},
            edges=CHAIN_EDGES,
        )


def test_hand_over_in_the_phase_that_misses_the_reader_is_rejected() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"never used in regime 'work': state \['skill'\]",
    ):
        _model(
            regimes=_work_retired_dead()
            | {
                "work": _alive(
                    skill_law=Phased(
                        solve=_reset_skill, simulate=fixed_transition("skill")
                    )
                )
            },
            edges=Phased(
                solve={"work": {"retired": 0}, "retired": {"dead": 1}},
                simulate={"work": {"gone": 0}, "retired": {"dead": 1}},
            ),
        )


def test_hand_over_in_the_phase_that_reaches_the_reader_is_used() -> None:
    model = _model(
        regimes=_work_retired_dead()
        | {
            "work": _alive(
                skill_law=Phased(solve=_reset_skill, simulate=fixed_transition("skill"))
            )
        },
        edges=SIMULATION_ONLY_EDGES,
    )

    assert "skill" in model.user_regimes["work"].states


def test_hand_over_along_an_edge_the_carrier_never_takes_is_rejected() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"never used in regime 'work': state \['skill'\]",
    ):
        _model(
            regimes=_work_retired_dead(),
            edges={"work": {"gone": 0, "retired": 1}, "retired": {"dead": 1}},
        )


def test_rejection_names_the_hand_over_rule_and_the_targets_inspected() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"carries .* to a reachable target that uses it.*\['gone'\]",
    ):
        _model(
            regimes=_work_retired_dead(),
            edges={"work": {"gone": 0}, "retired": {"dead": 1}},
        )


def test_hand_over_reaching_the_reader_only_in_simulation_is_used() -> None:
    model = _model(
        regimes=_work_retired_dead(),
        edges=SIMULATION_ONLY_EDGES,
    )

    assert "skill" in model.user_regimes["work"].states


@pytest.mark.parametrize("retired_reads", [True, False])
def test_hand_written_identity_law_is_a_use_whatever_the_target(
    *, retired_reads: bool
) -> None:
    model = _model(
        regimes=_work_retired_dead()
        | {"work": _alive(skill_law=_carry_skill)}
        | ({} if retired_reads else {"retired": _alive(with_skill=False)}),
        edges=CHAIN_EDGES,
    )

    assert "skill" in model.user_regimes["work"].states


def test_a_hand_over_only_to_itself_is_rejected() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"never used in regime 'work': state \['skill'\]",
    ):
        _model(
            regimes=_work_retired_dead() | {"retired": _alive(with_skill=False)},
            edges={"work": {"work": 0, "gone": 1}, "retired": {"dead": 1}},
        )


def test_a_hand_over_around_a_cycle_without_a_reader_is_rejected_in_both() -> None:
    with pytest.raises(ModelInitializationError) as error:
        _model(
            regimes=_work_retired_dead() | {"retired": _alive()},
            edges={
                "work": {"retired": 0, "gone": 1},
                "retired": {"work": 0, "gone": 1},
            },
            initial_nodes={0: ("work", "retired")},
        )

    message = str(error.value)
    assert "never used in regime 'work'" in message
    assert "never used in regime 'retired'" in message


def test_hand_over_reaching_the_reader_only_in_the_solution_is_used() -> None:
    model = _model(
        regimes=_work_retired_dead(),
        edges=Phased(
            solve={"work": {"retired": 0}, "retired": {"dead": 1}},
            simulate={"work": {"gone": 0}, "retired": {"dead": 1}},
        ),
    )

    assert "skill" in model.user_regimes["work"].states


def test_rejection_in_a_terminal_regime_says_it_reaches_no_regime() -> None:
    with pytest.raises(ModelInitializationError, match="'dead' reaches no regime"):
        _model(
            regimes=_work_retired_dead()
            | {
                "dead": Regime(
                    states={
                        "wealth": WEALTH_GRID,
                        "skill": DiscreteGrid(category_class=_Skill),
                    },
                    functions={"utility": _bequest},
                )
            },
            edges=CHAIN_EDGES,
        )


def test_simulation_side_markers_resolve_at_an_age_the_regime_is_visited() -> None:
    ages = AgeGrid(start=0, inclusive_stop=2, step="Y")
    regimes = _work_retired_dead() | {
        "work": _alive(
            utility=Phased(
                solve=_utility,
                simulate=AgeSpecializedFunction(
                    build=_utility_built_from_age_one, signature=lambda age: age
                ),
            )
        )
    }
    laws, _ = bind_edge_laws(edges=CHAIN_EDGES, regimes=regimes, ages=ages)

    # `unread` stays a candidate after the solution slice, so the simulation
    # slice, whose utility is only built from age 1, is walked too.
    kept = compute_kept_variables(
        user_regimes=regimes,
        laws=laws,
        candidates={"work": frozenset({"skill", "unread"})},
        koopmans_aggregator=None,
        ages=ages,
        active_periods_by_regime={"work": (0, 1), "retired": (1,), "dead": (2,)},
        visited_periods_by_regime={"work": (1,), "retired": (1,), "dead": (2,)},
    )

    assert kept["work"] == {"wealth", "consumption", "skill"}
