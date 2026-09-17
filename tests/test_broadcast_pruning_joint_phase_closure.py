"""Pruning closes the kept-sets jointly over both phase slices.

Which broadcast variables a regime keeps is decided by two operators, one per
phase slice, and they feed each other: a target that keeps a state only because
its simulation slice reads it exposes the *solution*-side entry law toward that
target, and whatever that law reads has to survive in the source. Applying each
operator once, in a fixed order, stops before that hand-over is seen and leaves
a retained entry law reading a variable that was pruned.

The behavior pinned here is that the kept-sets are the least common fixed point
of both operators: every retained law's own inputs survive, promoting a state
from regime level to model level changes neither the retained states nor the
laws' inputs, and the result does not depend on which phase slice is closed
first.
"""

import inspect
from collections.abc import Callable, Mapping
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.broadcast import (
    _joint_phase_closure,
    merge_model_slots,
)
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.regime_building.processing import compute_active_periods_by_regime
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinearAggregator,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=False)
class _CoupleRegimeId:
    couple: ScalarInt
    widow: ScalarInt
    widower: ScalarInt


@categorical(ordered=False)
class _ChainRegimeId:
    early: ScalarInt
    middle: ScalarInt
    late: ScalarInt


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=True)
class _Pension:
    small: ScalarInt
    large: ScalarInt


@categorical(ordered=True)
class _Flag:
    low: ScalarInt
    high: ScalarInt


_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=10)
_CONSUMPTION_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
_ENDOWMENT_GRID = LinSpacedGrid(start=0.0, stop=1.0, n_points=3)
_HEALTH_GRID = DiscreteGrid(category_class=_Health)
_PENSION_GRID = DiscreteGrid(category_class=_Pension)

_PARAMS = {"working": {"discount_factor": 0.95}}
_INITIAL_CONDITIONS = {
    "age": jnp.zeros(4),
    "wealth": jnp.asarray([20.0, 40.0, 60.0, 80.0]),
    "endowment": jnp.asarray([0.0, 0.5, 0.5, 1.0]),
    "regime_id": jnp.full(4, _RegimeId.working),
}


def _utility_from_consumption(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _bequest(wealth: float) -> FloatND:
    return jnp.log(wealth)


def _bequest_with_health(*, wealth: float, health: int) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.1 * health)


def _bequest_with_pension(*, wealth: float, pension: int) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.2 * pension)


def _next_wealth(*, wealth: float, consumption: float) -> float:
    return wealth - consumption


def _certain() -> ScalarFloat:
    return jnp.asarray(1.0)


def _even_split() -> ScalarFloat:
    return jnp.asarray(0.5)


def _entry_from_endowment(endowment: float) -> FloatND:
    """Probabilities over a two-category grid, richer endowments arriving high."""
    high = jnp.clip(endowment, 0.0, 1.0)
    return jnp.stack([1.0 - high, high], axis=-1)


def _entry_from_wealth(wealth: float) -> FloatND:
    """Probabilities over a two-category grid, richer agents arriving high."""
    high = jnp.clip(wealth / 100.0, 0.0, 1.0)
    return jnp.stack([1.0 - high, high], axis=-1)


def _entry_uniform() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _law_reads(*, law: object) -> frozenset[str]:
    """Collect every argument name a `state_transitions` entry reads.

    A law is a plain callable, a `MarkovTransition`, a per-target mapping of
    either, a `Phased` pair of any of those, or `None`.
    """
    if isinstance(law, Phased):
        return _law_reads(law=law.solve) | _law_reads(law=law.simulate)
    if isinstance(law, Mapping):
        reads: frozenset[str] = frozenset()
        for cell in law.values():
            reads |= _law_reads(law=cell)
        return reads
    if law is None:
        return frozenset()
    func = cast(
        "Callable[..., object]", law.func if isinstance(law, MarkovTransition) else law
    )
    return frozenset(inspect.signature(func).parameters)


def _dangling_reads(
    *, regime: Regime, state_name: str, model_variables: frozenset[str]
) -> frozenset[str]:
    """Return the model variables a regime's law reads but can no longer supply."""
    reads = _law_reads(law=regime.state_transitions[state_name])
    supplied = set(regime.states) | set(regime.actions) | set(regime.functions)
    return frozenset(reads & model_variables) - supplied


def _retained_variables(*, model: Model) -> dict[str, frozenset[str]]:
    """Return, per regime, the states and actions that survived pruning."""
    return {
        regime_name: frozenset(regime.states) | frozenset(regime.actions)
        for regime_name, regime in model.user_regimes.items()
    }


def _entry_targets(*, regime: Regime, state_name: str, phase: str) -> set[str]:
    """Return the targets one phase side of a keyed entry law names."""
    law = regime.state_transitions[state_name]
    cell = (
        (law.solve if phase == "solve" else law.simulate)
        if isinstance(law, Phased)
        else law
    )
    assert isinstance(cell, Mapping)
    return set(cell)


_PHASED_HEALTH_LAW_FORMS = {
    "phased-keyed": Phased(
        solve={"retired": MarkovTransition(_entry_from_endowment)},
        simulate={"retired": MarkovTransition(_entry_from_wealth)},
    ),
    "phased-raw": Phased(
        solve=MarkovTransition(_entry_from_endowment),
        simulate=MarkovTransition(_entry_from_wealth),
    ),
    "keyed": {"retired": MarkovTransition(_entry_from_endowment)},
    "raw": MarkovTransition(_entry_from_endowment),
}


def _working_regime(*, health_law: object, **overrides: Any) -> Regime:
    spec: dict[str, Any] = {
        "transition": {"retired": MarkovTransition(_certain)},
        "active": lambda age: age < 1,
        "states": {"wealth": _WEALTH_GRID},
        "actions": {"consumption": _CONSUMPTION_GRID},
        "functions": {"utility": _utility_from_consumption},
        "state_transitions": {
            "wealth": _next_wealth,
            "endowment": fixed_transition("endowment"),
            "health": health_law,
        },
    }
    spec.update(overrides)
    return Regime(**spec)


def _retired_regime(**overrides: Any) -> Regime:
    """A regime whose payoff reads `health` on the simulation side only."""
    spec: dict[str, Any] = {
        "transition": None,
        "active": lambda age: age >= 1,
        "states": {"wealth": _WEALTH_GRID},
        "functions": {"utility": Phased(solve=_bequest, simulate=_bequest_with_health)},
    }
    spec.update(overrides)
    return Regime(**spec)


def _phased_model(
    *, health_law: object = _PHASED_HEALTH_LAW_FORMS["phased-keyed"]
) -> Model:
    """`health` and `endowment` promoted to model-level states."""
    return Model(
        regimes={
            "working": _working_regime(health_law=health_law),
            "retired": _retired_regime(),
        },
        states={"health": _HEALTH_GRID, "endowment": _ENDOWMENT_GRID},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
    )


def _regime_level_model(
    *, health_law: object = _PHASED_HEALTH_LAW_FORMS["phased-keyed"]
) -> Model:
    """The same model with both states declared on the regimes that use them.

    A regime's state axes follow its declaration order, and merging puts the
    model-level slot first, so the control declares the promoted state ahead of
    `wealth` to compare value arrays without transposing them.
    """
    return Model(
        regimes={
            "working": _working_regime(
                health_law=health_law,
                states={"endowment": _ENDOWMENT_GRID, "wealth": _WEALTH_GRID},
            ),
            "retired": _retired_regime(
                states={"health": _HEALTH_GRID, "wealth": _WEALTH_GRID}
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
    )


@pytest.mark.parametrize("law_form", sorted(_PHASED_HEALTH_LAW_FORMS))
def test_the_input_of_a_retained_entry_law_survives_pruning(*, law_form: str) -> None:
    """A law kept because its target keeps the state keeps its own inputs too."""
    model = _phased_model(health_law=_PHASED_HEALTH_LAW_FORMS[law_form])

    assert "endowment" in model.user_regimes["working"].states


@pytest.mark.parametrize("law_form", sorted(_PHASED_HEALTH_LAW_FORMS))
def test_a_retained_entry_law_has_no_dangling_read(*, law_form: str) -> None:
    """Nothing a retained entry law reads is missing from the source regime."""
    model = _phased_model(health_law=_PHASED_HEALTH_LAW_FORMS[law_form])

    assert (
        _dangling_reads(
            regime=model.user_regimes["working"],
            state_name="health",
            model_variables=frozenset({"health", "endowment"}),
        )
        == frozenset()
    )


def test_the_source_keeps_the_entry_law_toward_the_retaining_target() -> None:
    """The solution side of the entry law still points at the target that keeps it."""
    model = _phased_model()

    assert _entry_targets(
        regime=model.user_regimes["working"], state_name="health", phase="solve"
    ) == {"retired"}


def test_declaration_placement_leaves_the_retained_variables_unchanged() -> None:
    """Promoting states to model level keeps every regime's retained variables."""
    assert _retained_variables(model=_phased_model()) == _retained_variables(
        model=_regime_level_model()
    )


def test_declaration_placement_leaves_the_entry_law_inputs_unchanged() -> None:
    """Promoting states to model level keeps what the entry law reads."""
    assert _law_reads(
        law=_phased_model().user_regimes["working"].state_transitions["health"]
    ) == _law_reads(
        law=_regime_level_model().user_regimes["working"].state_transitions["health"]
    )


def _couple_model() -> Model:
    """One source, two terminal targets, each keeping a different state."""
    return Model(
        regimes={
            "couple": Regime(
                transition={
                    "widow": MarkovTransition(_even_split),
                    "widower": MarkovTransition(_even_split),
                },
                active=lambda age: age < 1,
                states={"wealth": _WEALTH_GRID},
                actions={"consumption": _CONSUMPTION_GRID},
                functions={"utility": _utility_from_consumption},
                state_transitions={
                    "wealth": _next_wealth,
                    "endowment": fixed_transition("endowment"),
                    "health": Phased(
                        solve={"widow": MarkovTransition(_entry_from_endowment)},
                        simulate={"widow": MarkovTransition(_entry_from_wealth)},
                    ),
                    "pension": Phased(
                        solve={"widower": MarkovTransition(_entry_from_endowment)},
                        simulate={"widower": MarkovTransition(_entry_from_wealth)},
                    ),
                },
            ),
            "widow": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wealth": _WEALTH_GRID},
                functions={
                    "utility": Phased(solve=_bequest, simulate=_bequest_with_health)
                },
            ),
            "widower": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wealth": _WEALTH_GRID},
                functions={
                    "utility": Phased(solve=_bequest, simulate=_bequest_with_pension)
                },
            ),
        },
        states={
            "health": _HEALTH_GRID,
            "pension": _PENSION_GRID,
            "endowment": _ENDOWMENT_GRID,
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_CoupleRegimeId,
    )


def test_two_targets_keeping_different_states_both_rescue_their_law_inputs() -> None:
    """A source feeding two targets keeps what either target's entry law reads."""
    assert "endowment" in _couple_model().user_regimes["couple"].states


@pytest.mark.parametrize(
    ("state_name", "target"), [("health", "widow"), ("pension", "widower")]
)
def test_each_entry_law_survives_toward_the_target_that_keeps_its_state(
    *, state_name: str, target: str
) -> None:
    """Each keyed law is restricted to the one target that keeps its state."""
    assert _entry_targets(
        regime=_couple_model().user_regimes["couple"],
        state_name=state_name,
        phase="solve",
    ) == {target}


@pytest.mark.parametrize(
    ("regime_name", "state_name"), [("widow", "health"), ("widower", "pension")]
)
def test_a_target_keeps_only_the_state_its_own_payoff_reads(
    *, regime_name: str, state_name: str
) -> None:
    """Neither widowed regime carries the other's state."""
    assert _retained_variables(model=_couple_model())[regime_name] == frozenset(
        {"wealth", state_name}
    )


def _entry_flag_from_flag(flag: int) -> FloatND:
    return jnp.where(
        flag == _Flag.high, jnp.asarray([0.2, 0.8]), jnp.asarray([0.8, 0.2])
    )


def _bequest_with_flag(*, wealth: float, late_flag: int) -> FloatND:
    return jnp.log(wealth) * (1.0 + 0.3 * late_flag)


def _chain_regimes() -> dict[str, Regime]:
    """Three regimes whose retention alternates between the two phase slices.

    `late` keeps `late_flag` only on its simulation side; the solution side of
    `middle`'s entry law for it reads `flag`; the simulation side of `early`'s
    entry law for `flag` reads `endowment`. Each hop therefore becomes visible
    only after the other phase's operator has run.
    """
    return {
        "early": Regime(
            transition={"middle": MarkovTransition(_certain)},
            active=lambda age: age < 1,
            states={"wealth": _WEALTH_GRID},
            actions={"consumption": _CONSUMPTION_GRID},
            functions={"utility": _utility_from_consumption},
            state_transitions={
                "wealth": _next_wealth,
                "endowment": fixed_transition("endowment"),
                "flag": Phased(
                    solve={"middle": MarkovTransition(_entry_uniform)},
                    simulate={"middle": MarkovTransition(_entry_from_endowment)},
                ),
            },
        ),
        "middle": Regime(
            transition={"late": MarkovTransition(_certain)},
            active=lambda age: age == 1,
            states={"wealth": _WEALTH_GRID},
            actions={"consumption": _CONSUMPTION_GRID},
            functions={"utility": _utility_from_consumption},
            state_transitions={
                "wealth": _next_wealth,
                "flag": fixed_transition("flag"),
                "late_flag": Phased(
                    solve={"late": MarkovTransition(_entry_flag_from_flag)},
                    simulate={"late": MarkovTransition(_entry_uniform)},
                ),
            },
        ),
        "late": Regime(
            transition=None,
            active=lambda age: age >= 2,
            states={"wealth": _WEALTH_GRID},
            functions={"utility": Phased(solve=_bequest, simulate=_bequest_with_flag)},
        ),
    }


_CHAIN_MODEL_STATES = {
    "flag": DiscreteGrid(category_class=_Flag),
    "late_flag": DiscreteGrid(category_class=_Flag),
    "endowment": _ENDOWMENT_GRID,
}
_CHAIN_AGES = AgeGrid(start=0, stop=2, step="Y")


def _chain_model() -> Model:
    return Model(
        regimes=_chain_regimes(),
        states=_CHAIN_MODEL_STATES,
        ages=_CHAIN_AGES,
        regime_id_class=_ChainRegimeId,
    )


@pytest.mark.parametrize(
    ("regime_name", "state_name"), [("middle", "flag"), ("early", "endowment")]
)
def test_an_alternating_dependency_chain_is_closed_to_its_end(
    *, regime_name: str, state_name: str
) -> None:
    """A chain whose hops alternate phases is followed past its second hop."""
    assert state_name in _chain_model().user_regimes[regime_name].states


def _closure_arguments() -> dict[str, Any]:
    """Assemble the arguments the joint closure takes for the chain model."""
    regimes = _chain_regimes()
    model_slots: dict[str, Mapping[str, Any]] = {
        "functions": {},
        "constraints": {},
        "states": _CHAIN_MODEL_STATES,
        "state_transitions": {},
        "actions": {},
    }
    merged_regimes, broadcast_variables = merge_model_slots(
        user_regimes=regimes, model_slots=model_slots
    )
    seed = {
        regime_name: frozenset(
            (set(regime.states) | set(regime.actions))
            - broadcast_variables[regime_name]
        )
        for regime_name, regime in merged_regimes.items()
    }
    return {
        "specs": {
            regime_name: normalize_regime_phases(regime)
            for regime_name, regime in merged_regimes.items()
        },
        "user_regimes": merged_regimes,
        "broadcast_variables": broadcast_variables,
        "koopmans_aggregator": LinearAggregator(),
        "kept": seed,
        "all_regime_names": frozenset(merged_regimes),
        "ages": _CHAIN_AGES,
        "active_periods_by_regime": compute_active_periods_by_regime(
            ages=_CHAIN_AGES, user_regimes=regimes
        ),
    }


def test_the_joint_closure_does_not_depend_on_which_phase_runs_first() -> None:
    """Closing solution-first and simulation-first reach the same kept-sets."""
    assert _joint_phase_closure(
        **_closure_arguments(), phase_order=("solution", "simulation")
    ) == _joint_phase_closure(
        **_closure_arguments(), phase_order=("simulation", "solution")
    )


def test_the_joint_closure_is_idempotent() -> None:
    """Re-closing an already closed kept-set adds nothing."""
    arguments = _closure_arguments()
    closed = _joint_phase_closure(**arguments)
    arguments["kept"] = closed

    assert _joint_phase_closure(**arguments) == closed


def test_the_joint_closure_reaches_the_models_retained_variables() -> None:
    """The closure's kept-sets are what the built model carries."""
    assert _joint_phase_closure(**_closure_arguments()) == _retained_variables(
        model=_chain_model()
    )


def test_the_phased_model_solves_like_the_regime_level_control() -> None:
    """Model-level and regime-level declarations solve to the same value arrays."""
    model_level = _phased_model().solve(params=_PARAMS, log_level="off")
    regime_level = _regime_level_model().solve(params=_PARAMS, log_level="off")

    for period in (0, 1):
        expected_period = regime_level.values[period]
        assert set(model_level.values[period]) == set(expected_period)
        for regime_name, expected in expected_period.items():
            np.testing.assert_array_almost_equal(
                np.asarray(model_level.values[period][regime_name]),
                np.asarray(expected),
                decimal=DECIMAL_PRECISION,
                err_msg=f"{regime_name}, period {period}",
            )


def test_the_phased_model_simulates_like_the_regime_level_control() -> None:
    """Model-level and regime-level declarations simulate identically."""
    model_level = _phased_model().simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        log_level="off",
        seed=7,
    )
    regime_level = _regime_level_model().simulate(
        params=_PARAMS,
        initial_conditions=_INITIAL_CONDITIONS,
        log_level="off",
        seed=7,
    )

    expected = regime_level.to_dataframe(use_labels=False).sort_values(
        ["subject_id", "period"]
    )
    got = model_level.to_dataframe(use_labels=False).sort_values(
        ["subject_id", "period"]
    )
    assert list(got.columns) == list(expected.columns)
    np.testing.assert_array_almost_equal(
        got["wealth"].to_numpy(dtype=float),
        expected["wealth"].to_numpy(dtype=float),
        decimal=DECIMAL_PRECISION,
    )
