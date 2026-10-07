"""Regime-selection integrity, admission, result compatibility and random sites.

Every regime law is checked on the rows it is evaluated on: mass outside the
source law's declared support, non-finite or out-of-range mass, invalid
deterministic codes and non-Boolean gates fail at every log level, even when
another start makes the stray destination a solved pair. A simulated start
outside `model.initial_nodes` fails before any user law runs, and simulating
never re-derives demand. A result solved for different visits is refused.
Random draws are keyed by `(period, regime)` site, so a deterministic exit
leaves later shock draws where a stochastic one would put them.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.model_graph
from _lcm.simulation.random import site_simulation_key
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    StochasticTransition,
    Transition,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidRegimeTransitionProbabilitiesError,
    InvalidSimulationInputError,
    RegimeInitializationError,
)
from lcm.phased import Phased
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.regime_building.test_same_period_ref_period_axes import (
    EXPECTED_V_COUPLE,
)
from tests.regime_building.test_same_period_ref_period_axes import (
    _make_model as _make_outside_option_model,
)
from tests.test_admission_and_random_sites import _model as _admission_model
from tests.test_demand_worklists import _phased_model

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}
_AGES = AgeGrid(start=25, inclusive_stop=75, step="10Y")
_LOG_LEVELS = pytest.mark.parametrize("log_level", ["off", "warning"])


@categorical(ordered=False)
class _LifeId:
    working: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _nonterminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _terminal() -> Regime:
    return Regime(
        states={"wealth": _WEALTH},
        functions={"utility": _utility},
    )


def _stay() -> FloatND:
    return jnp.asarray(0.9)


def _die() -> FloatND:
    return jnp.asarray(0.1)


def _half() -> FloatND:
    return jnp.asarray(0.5)


_EARLY = {
    "working": StochasticTransition(func=_stay),
    "dead": StochasticTransition(func=_die),
}


def _life_model(
    *,
    law_at_55: Any,
    initial_nodes: Any,
    early: Any = None,
    retires_at_55: bool = True,
) -> Model:
    return Model(
        edges={
            "working": Transition(
                targets={"working": (25, 35, 45), "dead": (25, 35, 45, 55)}
                | ({"retirement": 55} if retires_at_55 else {}),
                law=ByAge(
                    cases={
                        AgeRange(start=25, exclusive_stop=55): early or _EARLY,
                        55: law_at_55,
                    }
                ),
            ),
            "retirement": {"dead": 65},
        },
        regimes={
            "working": _nonterminal(),
            "retirement": _nonterminal(),
            "dead": _terminal(),
        },
        ages=_AGES,
        regime_id_class=_LifeId,
        initial_nodes=initial_nodes,
    )


def _stay_vector() -> FloatND:
    return jnp.asarray([0.9, 0.0, 0.1])


def _all_to_retirement() -> FloatND:
    return jnp.asarray([0.0, 1.0, 0.0])


@_LOG_LEVELS
def test_stray_mass_into_a_pair_another_start_solves_fails(
    log_level: LogLevel,
) -> None:
    """A law declaring only `dead` may not send mass to the solved retirement."""
    model = _life_model(
        law_at_55=StochasticTransition(func=_all_to_retirement),
        early=StochasticTransition(func=_stay_vector),
        retires_at_55=False,
        initial_nodes={25: "working", 65: "retirement"},
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"'retirement' is outside the declared targets of 'working' at age 55",
    ):
        model.solve(params=_PARAMS, log_level=log_level)


def _nan() -> FloatND:
    return jnp.asarray(jnp.nan)


def _negative() -> FloatND:
    return jnp.asarray(-0.5)


def _excess() -> FloatND:
    return jnp.asarray(1.5)


@pytest.mark.parametrize(
    ("retire", "match"),
    [
        (_nan, r"Non-finite values in regime transition probabilities from 'working'"),
        (_negative, r"from 'working' between ages 55 and 65 contain values outside"),
        (_excess, r"from 'working' between ages 55 and 65 contain values outside"),
    ],
    ids=["nan", "negative", "excess"],
)
@_LOG_LEVELS
def test_invalid_selection_mass_fails_at_every_log_level(
    *, retire: Any, match: str, log_level: LogLevel
) -> None:
    """NaN, negative and above-one cells are refused whatever the verbosity."""
    model = _life_model(
        law_at_55={
            "retirement": StochasticTransition(func=retire),
            "dead": StochasticTransition(func=_half),
        },
        initial_nodes={25: "working"},
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match=match):
        model.solve(params=_PARAMS, log_level=log_level)


def _code_seven() -> ScalarInt:
    return jnp.asarray(7)


def test_a_deterministic_code_outside_the_targets_fails_with_logging_off() -> None:
    """A selector returning the unregistered code 7 leaves a zero-mass row."""
    model = _life_model(
        law_at_55=DeterministicTransition(func=_code_seven),
        initial_nodes={25: "working"},
    )
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError,
        match=r"from 'working' between ages 55 and 65 do not sum to 1\.0",
    ):
        model.solve(params=_PARAMS, log_level="off")


def _even_split() -> FloatND:
    return jnp.asarray([0.5, 0.0, 0.5])


def test_public_probability_kernel_rejects_embedded_topology() -> None:
    """Destination mappings own topology; kernels reject the former targets field."""
    with pytest.raises(TypeError, match="targets"):
        StochasticTransition(
            func=_even_split,
            targets=("working", "working", "dead"),  # ty: ignore[unknown-argument]
        )


@categorical(ordered=False)
class _GatedId:
    source: ScalarInt
    target: ScalarInt
    reference: ScalarInt
    fallback: ScalarInt


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _numeric_gate(*, V_target: FloatND, V_reference: FloatND) -> FloatND:
    return V_target - V_reference + 0.5


def _identity(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _numeric_gate_model() -> Model:
    return Model(
        edges={
            "source": Transition(
                targets={"target": 40, "fallback": 40},
                law=ByAge(
                    cases={
                        40: {
                            "target": ValueDependentTransition(
                                probability=StochasticTransition(func=_prob_one),
                                gate=_numeric_gate,
                                routes={
                                    "only": StakeholderRoute(
                                        fallback=ProjectedRegimeValue(
                                            regime="fallback",
                                            projection={"wealth": _identity},
                                        )
                                    )
                                },
                                gate_references={
                                    "V_reference": ProjectedRegimeValue(
                                        regime="reference",
                                        projection={"wealth": _identity},
                                    )
                                },
                            )
                        }
                    }
                ),
            )
        },
        regimes={
            "source": _nonterminal(),
            "target": _terminal(),
            "reference": _terminal(),
            "fallback": _terminal(),
        },
        ages=AgeGrid(start=40, inclusive_stop=50, step="5Y"),
        regime_id_class=_GatedId,
        initial_nodes={40: "source"},
    )


def test_a_numeric_gate_output_fails_with_logging_off() -> None:
    """A gate returning floats instead of booleans is refused."""
    with pytest.raises(
        RegimeInitializationError, match=r"returned dtype 'float(32|64)'"
    ):
        _numeric_gate_model().solve(params=_PARAMS, log_level="off")


_N_SUBJECTS = 1000


def _panel(*, model: Model, age: float, regime: str, seed: int = 0) -> Any:
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(_N_SUBJECTS),
            "age": jnp.full(_N_SUBJECTS, age),
            "regime_id": jnp.full(_N_SUBJECTS, model.regime_names_to_ids[regime]),
        },
        solution=model.solve(params=_PARAMS, log_level="off"),
        log_level="off",
        seed=seed,
    ).to_dataframe()


def test_two_half_probabilities_split_the_panel() -> None:
    """Seed 0 splits 1000 subjects at 55 about evenly into retirees and deaths.

    The uniform draws depend on the float precision, so the exact split is
    pinned per precision: 495 / 505 under x64, 513 / 487 under float32.
    """
    model = _life_model(
        law_at_55={
            "retirement": StochasticTransition(func=_half),
            "dead": StochasticTransition(func=_half),
        },
        initial_nodes={55: "working"},
    )
    panel = _panel(model=model, age=55.0, regime="working")
    retirees = 495 if jax.config.jax_enable_x64 else 513
    assert panel.query("age == 65")["regime_name"].value_counts().to_dict() == {
        "working": 0,
        "retirement": retirees,
        "dead": 1000 - retirees,
    }


@categorical(ordered=False)
class _JobId:
    unemployed_before_switch: ScalarInt
    unemployed_after_switch: ScalarInt
    employed: ScalarInt
    dead: ScalarInt


_JOB_FINDING_RATE = 0.2


def _remain() -> FloatND:
    return jnp.asarray(1 - _JOB_FINDING_RATE)


def _find_job() -> FloatND:
    return jnp.asarray(_JOB_FINDING_RATE)


def _remain_split() -> FloatND:
    return jnp.asarray((1 - _JOB_FINDING_RATE) / 2)


_DOUBLE_REMAIN = {
    "unemployed_before_switch": StochasticTransition(func=_remain),
    "unemployed_after_switch": StochasticTransition(func=_remain),
    "employed": StochasticTransition(func=_find_job),
}
_SPLIT_REMAIN = {
    "unemployed_before_switch": StochasticTransition(func=_remain_split),
    "unemployed_after_switch": StochasticTransition(func=_remain_split),
    "employed": StochasticTransition(func=_find_job),
}
_SHARED_HALF = {
    "unemployed_after_switch": StochasticTransition(func=_half),
    "employed": StochasticTransition(func=_half),
}
_REMAIN_TARGETS = (
    "unemployed_before_switch",
    "unemployed_after_switch",
    "employed",
)


# keyword-only-exempt: primary-argument=first_law
def _job_model(
    first_law: Any, *, targets_at_25: tuple[str, ...] = _REMAIN_TARGETS
) -> Model:
    """Every destination is a known regime with a law at 35 and a wealth handoff."""
    return Model(
        edges={
            "unemployed_before_switch": Transition(
                targets=dict.fromkeys(targets_at_25, 25) | {"dead": 35},
                law=ByAge(cases={25: first_law, 35: "dead"}),
            ),
            "unemployed_after_switch": {"dead": 35},
            "employed": {"dead": 35},
        },
        regimes={
            "unemployed_before_switch": _nonterminal(),
            "unemployed_after_switch": _nonterminal(),
            "employed": _nonterminal(),
            "dead": _terminal(),
        },
        ages=AgeGrid(start=25, inclusive_stop=45, step="10Y"),
        regime_id_class=_JobId,
        initial_nodes={25: "unemployed_before_switch"},
    )


def _job_simulate(*, model: Model, log_level: LogLevel, n_subjects: int) -> Any:
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(n_subjects),
            "age": jnp.full(n_subjects, 25.0),
            "regime_id": jnp.full(
                n_subjects, model.regime_names_to_ids["unemployed_before_switch"]
            ),
        },
        solution=model.solve(params=_PARAMS, log_level="off"),
        log_level=log_level,
        seed=0,
    )


_DOUBLE_COUNTED = (
    r"from 'unemployed_before_switch' between ages 25 and 35 do not sum to 1\.0"
    r"(.|\n)*1\.8"
)


@_LOG_LEVELS
def test_remain_mass_counted_for_two_distinct_targets_fails_in_solve(
    log_level: LogLevel,
) -> None:
    """Rows (0.8, 0.8, 0.2) to three distinct valid regimes sum to 1.8 in solve."""
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=_DOUBLE_COUNTED
    ):
        _job_model(_DOUBLE_REMAIN).solve(params=_PARAMS, log_level=log_level)


@_LOG_LEVELS
def test_remain_mass_counted_for_two_distinct_targets_fails_in_simulate(
    log_level: LogLevel,
) -> None:
    """The realized law (0.8, 0.8, 0.2) is refused although the solve law is valid."""
    model = _job_model(Phased(solve=_SPLIT_REMAIN, simulate=_DOUBLE_REMAIN))
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=_DOUBLE_COUNTED
    ):
        _job_simulate(model=model, log_level=log_level, n_subjects=4)


def test_two_targets_sharing_a_half_probability_callable_simulate() -> None:
    """Seed 0 splits 1000 subjects at 25 into the two targets of `_half`.

    The uniform draws depend on the float precision, so the split is pinned per
    precision: 501 / 499 under x64, 487 / 513 under float32.
    """
    panel = _job_simulate(
        model=_job_model(
            _SHARED_HALF, targets_at_25=("unemployed_after_switch", "employed")
        ),
        log_level="warning",
        n_subjects=_N_SUBJECTS,
    ).to_dataframe()
    after_switch = 501 if jax.config.jax_enable_x64 else 487
    assert panel.query("age == 35")["regime_name"].value_counts().to_dict() == {
        "unemployed_before_switch": 0,
        "unemployed_after_switch": after_switch,
        "employed": _N_SUBJECTS - after_switch,
        "dead": 0,
    }


def _nonfinite_between_nodes(wealth: ContinuousState) -> FloatND:
    return jnp.where((wealth > 0.25) & (wealth < 0.75), jnp.nan, 0.5)


def _complement_between_nodes(wealth: ContinuousState) -> FloatND:
    return 1 - _nonfinite_between_nodes(wealth)


def _phased_by_age(*, valid: dict, invalid: dict) -> ByAge:
    return ByAge(
        cases={
            AgeRange(start=25, exclusive_stop=45): Phased(solve=valid, simulate=valid),
            AgeRange(start=45, exclusive_stop=65): Phased(
                solve=valid, simulate=invalid
            ),
        },
        default="dead",
    )


def _later_age_invalid_model() -> Model:
    valid = {
        "working": StochasticTransition(func=_half),
        "dead": StochasticTransition(func=_half),
    }
    invalid = {
        "working": StochasticTransition(func=_excess),
        "dead": StochasticTransition(func=_half),
    }
    return Model(
        edges={
            "working": Transition(
                targets={
                    "working": AgeRange(exclusive_stop=65),
                    "dead": AgeRange(exclusive_stop=75),
                },
                law=_phased_by_age(valid=valid, invalid=invalid),
            )
        },
        regimes={
            "working": _nonterminal(),
            "retirement": _terminal(),
            "dead": _terminal(),
        },
        ages=_AGES,
        regime_id_class=_LifeId,
        initial_nodes={25: "working"},
    )


def _simulate_off(*, model: Model, wealth: float) -> None:
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.full(4, wealth),
            "age": jnp.full(4, 25.0),
            "regime_id": jnp.full(4, model.regime_names_to_ids["working"]),
        },
        log_level="off",
        seed=0,
    )


def test_a_realized_law_invalid_only_at_a_later_age_fails() -> None:
    """A simulate law valid at 25 and 35 but not from 45 on is refused."""
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=r"'working'.*outside"
    ):
        _simulate_off(model=_later_age_invalid_model(), wealth=0.0)


def _off_grid_model() -> Model:
    law = {
        "working": StochasticTransition(func=_nonfinite_between_nodes),
        "dead": StochasticTransition(func=_complement_between_nodes),
    }
    return Model(
        edges={
            "working": Transition(
                targets={"working": 25, "dead": (25, 35)},
                law=ByAge.until(
                    stop_age_exclusive=45,
                    law=Phased(solve=_EARLY, simulate=law),
                    then="dead",
                ),
            )
        },
        regimes={
            "working": _nonterminal(),
            "retirement": _terminal(),
            "dead": _terminal(),
        },
        ages=AgeGrid(start=25, inclusive_stop=45, step="10Y"),
        regime_id_class=_LifeId,
        initial_nodes={25: "working"},
    )


def test_a_realized_law_nonfinite_only_between_grid_nodes_fails() -> None:
    """Wealth 0.5 lies between the nodes 0 and 1; its NaN row is still refused."""
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=r"Non-finite.*'working'"
    ):
        _simulate_off(model=_off_grid_model(), wealth=0.5)


def test_a_realized_law_on_grid_rows_simulates() -> None:
    """The control: starting on the node 0 the same law is valid throughout."""
    _simulate_off(model=_off_grid_model(), wealth=0.0)


_RECORDED_CALLS: list[str] = []


def _recording_stay() -> FloatND:
    _RECORDED_CALLS.append("stay")
    return jnp.asarray(0.5)


def _recording_die() -> FloatND:
    _RECORDED_CALLS.append("die")
    return jnp.asarray(0.5)


def _recording_model() -> Model:
    return _life_model(
        law_at_55={
            "retirement": StochasticTransition(func=_recording_stay),
            "dead": StochasticTransition(func=_recording_die),
        },
        early={
            "working": StochasticTransition(func=_recording_stay),
            "dead": StochasticTransition(func=_recording_die),
        },
        initial_nodes={25: "working"},
    )


def _simulate_after_solve(*, model: Model, age: float, log_level: LogLevel) -> None:
    solution = model.solve(params=_PARAMS, log_level="off")
    _RECORDED_CALLS.clear()
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(2),
            "age": jnp.full(2, age),
            "regime_id": jnp.full(2, model.regime_names_to_ids["working"]),
        },
        solution=solution,
        log_level=log_level,
        seed=0,
    )


@_LOG_LEVELS
def test_a_refused_start_is_refused_by_name(log_level: LogLevel) -> None:
    """A solved but undeclared start fails naming its pair at every log level."""
    with pytest.raises(InvalidInitialConditionsError, match=r"\(35, 'working'\)"):
        _simulate_after_solve(model=_recording_model(), age=35.0, log_level=log_level)


@_LOG_LEVELS
def test_a_refused_start_evaluates_no_user_law(log_level: LogLevel) -> None:
    """Admission fails before any regime law is traced or evaluated."""
    with pytest.raises(InvalidInitialConditionsError):
        _simulate_after_solve(model=_recording_model(), age=35.0, log_level=log_level)
    assert _RECORDED_CALLS == []


def test_an_admitted_start_evaluates_the_user_law() -> None:
    """The control: simulating a declared start does evaluate the laws."""
    _simulate_after_solve(model=_recording_model(), age=25.0, log_level="off")
    assert set(_RECORDED_CALLS) == {"stay", "die"}


@pytest.mark.parametrize(
    ("n_subjects", "seed"), [(2, 0), (5, 0), (5, 1)], ids=["two", "five", "reseeded"]
)
def test_simulate_does_not_resolve_demand_again(
    *, n_subjects: int, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch size and seed do not re-derive the demanded pairs."""
    model = _admission_model({25: "working"})
    solution = model.solve(params=_PARAMS, log_level="off")
    calls: list[int] = []

    def _counting(**kwargs: Any) -> Any:
        calls.append(1)
        return original(**kwargs)

    original = _lcm.model_graph.resolve_demand
    monkeypatch.setattr(_lcm.model_graph, "resolve_demand", _counting)
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(n_subjects),
            "age": jnp.full(n_subjects, 25.0),
            "regime_id": jnp.full(n_subjects, model.regime_names_to_ids["working"]),
        },
        solution=solution,
        log_level="off",
        seed=seed,
    )
    assert calls == []


def test_mutating_the_declared_roots_after_build_changes_nothing() -> None:
    """The model keeps its own snapshot of the starts and their closure."""
    roots: dict[Any, Any] = {25: "working"}
    model = _admission_model(roots)
    before = (model.initial_nodes, model.reachability.nodes)
    roots[25] = "island"
    roots[35] = "working"
    assert (model.initial_nodes, model.reachability.nodes) == before


@pytest.mark.parametrize("age", [25.0, 35.0, 45.0, 55.0, 65.0, 75.0])
def test_a_dead_entrant_fails_at_every_age(age: float) -> None:
    """Starting in the terminal regime is refused, also where it is solved."""
    model = _admission_model({25: "working"})
    with pytest.raises(InvalidInitialConditionsError, match=rf"\({age:g}, 'dead'\)"):
        model.simulate(
            params=_PARAMS,
            initial_conditions={
                "wealth": jnp.zeros(2),
                "age": jnp.full(2, age),
                "regime_id": jnp.full(2, model.regime_names_to_ids["dead"]),
            },
            solution=model.solve(params=_PARAMS, log_level="off"),
            log_level="off",
            seed=0,
        )


def test_a_dissolved_cell_keeps_its_nonfinite_value_with_logging_off() -> None:
    """The couple's dissolved cell stays `-inf`; nothing masks it to a number."""
    values = _make_outside_option_model(
        later_ceiling=10.0, initial_nodes={0: "couple"}
    ).solve(
        params={
            "single_f": {"koopmans_aggregator": {"discount_factor": 0.0}},
            "single_f_terminal": {},
            "couple": {
                "koopmans_aggregator": {"discount_factor": 0.0},
                "participation_f": {"delta_f": 0.0},
            },
            "couple_terminal": {},
        },
        log_level="off",
    )
    np.testing.assert_array_equal(
        np.asarray(values.values[0]["couple"]), np.asarray(EXPECTED_V_COUPLE)
    )


def test_a_solution_for_other_visits_is_refused() -> None:
    """Same solved pairs, different visited pairs: the result does not fit."""
    visits_perceived = _phased_model({0: "source", 1: "perceived"})
    values_perceived = _phased_model({0: "source", 2: "realized_end"})
    solution = visits_perceived.solve(params=_PARAMS, log_level="off")
    with pytest.raises(
        InvalidSimulationInputError, match="model_fingerprint does not match"
    ):
        values_perceived.simulate(
            params=_PARAMS,
            initial_conditions={
                "wealth": jnp.zeros(2),
                "age": jnp.zeros(2),
                "regime_id": jnp.full(
                    2, values_perceived.regime_names_to_ids["source"]
                ),
            },
            solution=solution,
            log_level="off",
            seed=0,
        )


def _program_keys(model: Model) -> dict[str, tuple]:
    program_fingerprint = model._program_fingerprint(
        flat_params=model._process_params(_PARAMS)
    )
    return {
        name: (
            program_fingerprint,
            tuple(sorted(regime.solution.period_signatures.items())),
            tuple(sorted(regime.solution.solver_period_group_keys.items())),
        )
        for name, regime in model._regimes.items()
    }


def test_a_redundant_start_keeps_every_regime_program_key() -> None:
    """A start that is already visited leaves each regime's program keys as is."""
    assert _program_keys(_phased_model({0: "source"})) == _program_keys(
        _phased_model({0: "source", 1: "realized"})
    )


def test_a_start_changing_only_visits_changes_the_program_keys() -> None:
    """Same solved pairs, different replay requirements: the keys differ."""
    assert _program_keys(_phased_model({0: "source", 1: "perceived"})) != (
        _program_keys(_phased_model({0: "source", 2: "realized_end"}))
    )


@pytest.mark.parametrize(
    ("first", "second"),
    [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (0, 1))],
    ids=["other-regime", "other-period", "swapped"],
)
def test_site_keys_differ_between_sites(
    *, first: tuple[int, int], second: tuple[int, int]
) -> None:
    """Each `(period, regime)` site derives its own key from the root key."""
    root = jax.random.key(0)
    keys = [
        jax.random.key_data(site_simulation_key(key=root, period=p, regime_id=r))
        for p, r in (first, second)
    ]
    assert not np.array_equal(np.asarray(keys[0]), np.asarray(keys[1]))


def test_site_key_is_stable_for_a_seed() -> None:
    """The same seed and site derive the same key on every call."""
    keys = [
        jax.random.key_data(
            site_simulation_key(key=jax.random.key(3), period=2, regime_id=1)
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(np.asarray(keys[0]), np.asarray(keys[1]))


def test_site_key_changes_with_the_seed() -> None:
    """Different root seeds derive different keys at the same site."""
    keys = [
        jax.random.key_data(
            site_simulation_key(key=jax.random.key(seed), period=2, regime_id=1)
        )
        for seed in (3, 4)
    ]
    assert not np.array_equal(np.asarray(keys[0]), np.asarray(keys[1]))


def _choose_retirement() -> ScalarInt:
    return jnp.asarray(1)


def _certain() -> FloatND:
    return jnp.asarray(1.0)


def _shock_utility(*, wealth: ContinuousState, income: ContinuousState) -> FloatND:
    return wealth + income


def _shock_model(exit_law: Any) -> Model:
    return Model(
        edges={
            "working": Transition(
                targets={"retirement": 25, "dead": 25},
                law=ByAge(cases={25: exit_law}),
            ),
            "retirement": {"retirement": (35, 45, 55), "dead": 65},
        },
        regimes={
            "working": _nonterminal(),
            "retirement": Regime(
                states={
                    "wealth": _WEALTH,
                    "income": NormalIIDProcess(
                        n_points=3, gauss_hermite=False, mu=0.0, sigma=1.0, n_std=2.0
                    ),
                },
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _shock_utility},
            ),
            "dead": _terminal(),
        },
        ages=_AGES,
        regime_id_class=_LifeId,
        initial_nodes={25: "working"},
    )


def _shock_panel(exit_law: Any) -> Any:
    model = _shock_model(exit_law)
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(8),
            "age": jnp.full(8, 25.0),
            "regime_id": jnp.full(8, model.regime_names_to_ids["working"]),
        },
        solution=model.solve(params=_PARAMS, log_level="off"),
        log_level="off",
        seed=0,
    ).to_dataframe()


_EXIT_FORMS = pytest.mark.parametrize(
    "exit_law",
    [
        DeterministicTransition(func=_choose_retirement),
        {"retirement": StochasticTransition(func=_certain)},
    ],
    ids=["choose", "singleton-markov"],
)


@_EXIT_FORMS
def test_every_subject_takes_the_single_route(exit_law: Any) -> None:
    """A deterministic or singleton exit sends all eight subjects to retirement."""
    panel = _shock_panel(exit_law)
    assert panel.query("age == 35")["regime_name"].tolist() == ["retirement"] * 8


@_EXIT_FORMS
def test_downstream_shock_draws_do_not_depend_on_the_exit_form(exit_law: Any) -> None:
    """Retirement income draws match those after a plain named exit."""
    columns = ["subject_id", "age", "income"]
    np.testing.assert_array_equal(
        _shock_panel(exit_law)[columns].to_numpy(),
        _shock_panel("retirement")[columns].to_numpy(),
    )
