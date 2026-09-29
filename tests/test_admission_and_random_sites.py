"""Empirical admission against the declared starts, and stable random sites.

A simulated subject may start only at a pair in `model.initial_nodes`; every other
known pair is refused by name, whether or not the engine solves it. A subject's
draws at a given age and regime depend on the seed, the age, the regime and the
subject alone, not on which other pairs the declared starts make visitable.
"""

from typing import Any

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidRegimeTransitionProbabilitiesError,
)
from lcm.phased import Phased
from lcm.typing import ContinuousState, FloatND, ScalarInt
from tests.test_demand_worklists import _phased_model

_WEALTH = LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
_PARAMS = {"discount_factor": 0.9}
_AGES = AgeGrid(start=25, stop=75, step="10Y")
_N_SUBJECTS = 64


@categorical(ordered=False)
class _Life:
    island: ScalarInt
    working: ScalarInt
    dead: ScalarInt


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _stay() -> FloatND:
    return jnp.asarray(0.5)


def _die() -> FloatND:
    return jnp.asarray(0.5)


def _mortal() -> Regime:
    return Regime(
        regime_transitions=ByAge.until(
            stop_age_exclusive=75,
            law={
                "working": MarkovTransition(func=_stay),
                "dead": MarkovTransition(func=_die),
            },
            then="dead",
        ),
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _island() -> Regime:
    return Regime(
        regime_transitions=ByAge.until(
            stop_age_exclusive=75,
            law={
                "island": MarkovTransition(func=_stay),
                "dead": MarkovTransition(func=_die),
            },
            then="dead",
        ),
        states={"wealth": _WEALTH},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
    )


def _model(initial_regimes: Any) -> Model:
    return Model(
        regimes={
            "island": _island(),
            "working": _mortal(),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        ages=_AGES,
        regime_id_class=_Life,
        initial_regimes=initial_regimes,
    )


def _panel(*, model: Model, age: float, regime: str) -> Any:
    solution = model.solve(params=_PARAMS, log_level="off")
    result = model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(_N_SUBJECTS),
            "age": jnp.full(_N_SUBJECTS, age),
            "regime_id": jnp.full(_N_SUBJECTS, model.regime_names_to_ids[regime]),
        },
        solution=solution,
        log_level="off",
        seed=0,
    )
    return result.to_dataframe()


@pytest.mark.parametrize(
    ("age", "regime"),
    [(35.0, "working"), (75.0, "dead"), (25.0, "island"), (25.0, "dead")],
    ids=["solved-living", "solved-terminal", "unsolved-regime", "unsolved-age"],
)
def test_simulate_refuses_starts_outside_initial_nodes(
    *, age: float, regime: str
) -> None:
    """A start outside the declared set is refused by name, solved or not."""
    model = _model({25: "working"})
    with pytest.raises(
        InvalidInitialConditionsError, match=rf"\({age:g}, '{regime}'\)"
    ):
        _panel(model=model, age=age, regime=regime)


def test_simulate_admits_starts_inside_initial_nodes() -> None:
    """Subjects at a declared start are simulated from there."""
    panel = _panel(model=_model({25: "working"}), age=25.0, regime="working")
    assert panel.query("age == 25")["regime_name"].eq("working").all()


def test_initial_nodes_survive_pickling() -> None:
    """The declared starts are part of the persisted model."""
    model = _model({25: "working"})
    assert (
        cloudpickle.loads(cloudpickle.dumps(model)).initial_nodes == model.initial_nodes
    )


def test_draws_do_not_depend_on_other_visitable_pairs() -> None:
    """Adding an unrelated start leaves every working subject's trajectory as is."""
    alone = _panel(model=_model({25: "working"}), age=25.0, regime="working")
    with_island = _panel(
        model=_model({25: ("working", "island")}), age=25.0, regime="working"
    )
    np.testing.assert_array_equal(
        alone[["subject_id", "age", "regime_name"]].to_numpy(),
        with_island[["subject_id", "age", "regime_name"]].to_numpy(),
    )


def test_the_extra_start_adds_visited_pairs() -> None:
    """The trajectory comparison differs in its visited pairs, not only its roots."""
    assert (25, "island") in _model(
        {25: ("working", "island")}
    ).reachability.visited_nodes


def _identity(model: Model) -> tuple[frozenset, frozenset, str]:
    reachability = model.reachability
    return (
        reachability.nodes,
        reachability.visited_nodes,
        model._model_structure_fingerprint,
    )


def test_a_redundant_start_keeps_the_structure_identity() -> None:
    """A start that is already visited changes neither S, H nor the identity."""
    alone = _phased_model({0: "source"})
    redundant = _phased_model({0: "source", 1: "realized"})
    assert alone.initial_nodes != redundant.initial_nodes
    assert _identity(alone) == _identity(redundant)


def test_a_start_changing_only_visits_changes_the_structure_identity() -> None:
    """Same solved pairs, different visited pairs: the identity differs."""
    visits_perceived = _identity(_phased_model({0: "source", 1: "perceived"}))
    values_perceived = _identity(_phased_model({0: "source", 2: "realized_end"}))
    assert visits_perceived[0] == values_perceived[0]
    assert visits_perceived[1] != values_perceived[1]
    assert visits_perceived[2] != values_perceived[2]


def _overweight_death() -> FloatND:
    return jnp.asarray(0.9)


def _laws(*, die: Any) -> dict:
    return {
        "working": MarkovTransition(func=_stay),
        "dead": MarkovTransition(func=die),
    }


def _law_model(*, law: Any, n_wealth: int = 2) -> Model:
    wealth = LinSpacedGrid(start=0.0, stop=1.0, n_points=n_wealth)
    return Model(
        regimes={
            "island": _island(),
            "working": Regime(
                regime_transitions=ByAge.until(
                    stop_age_exclusive=75, law=law, then="dead"
                ),
                states={"wealth": wealth},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": wealth},
                functions={"utility": _utility},
            ),
        },
        ages=_AGES,
        regime_id_class=_Life,
        initial_regimes={25: "working"},
    )


def _simulate_off(model: Model) -> None:
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.zeros(2),
            "age": jnp.full(2, 25.0),
            "regime_id": jnp.full(2, model.regime_names_to_ids["working"]),
        },
        log_level="off",
        seed=0,
    )


def test_solve_refuses_an_invalid_regime_law_with_logging_off() -> None:
    """A selection row with mass 1.4 raises even at `log_level="off"`."""
    model = _law_model(law=_laws(die=_overweight_death))
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params=_PARAMS, log_level="off")


def test_simulate_refuses_an_invalid_realized_law_with_logging_off() -> None:
    """An invalid simulate-side `Phased` law raises before simulation succeeds."""
    model = _law_model(
        law=Phased(solve=_laws(die=_die), simulate=_laws(die=_overweight_death))
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        _simulate_off(model)


def test_valid_realized_law_simulates_with_logging_off() -> None:
    """The control: the same model with a valid realized law simulates."""
    _simulate_off(
        _law_model(law=Phased(solve=_laws(die=_die), simulate=_laws(die=_die)))
    )


def _one_third() -> FloatND:
    return jnp.asarray(1 / 3)


def _two_thirds() -> FloatND:
    return jnp.asarray(2 / 3)


def test_a_changed_law_changes_the_structure_identity() -> None:
    """Different selection probabilities give a different structure digest."""
    base = _law_model(law=_laws(die=_die))
    changed = _law_model(
        law={
            "working": MarkovTransition(func=_two_thirds),
            "dead": MarkovTransition(func=_one_third),
        }
    )
    assert base._model_structure_fingerprint != changed._model_structure_fingerprint


def _durable_digest(*, model: Model, params: dict) -> str:
    return model._model_fingerprint(flat_params=model._process_params(params))


def test_a_changed_grid_changes_the_durable_identity() -> None:
    """A finer wealth grid gives a different durable result digest."""
    coarse = _law_model(law=_laws(die=_die))
    fine = _law_model(law=_laws(die=_die), n_wealth=3)
    assert _durable_digest(model=coarse, params=_PARAMS) != _durable_digest(
        model=fine, params=_PARAMS
    )


def test_changed_params_change_the_durable_identity() -> None:
    """A different discount factor gives a different durable result digest."""
    model = _law_model(law=_laws(die=_die))
    assert _durable_digest(model=model, params=_PARAMS) != _durable_digest(
        model=model, params={"discount_factor": 0.8}
    )


def test_identical_models_share_the_durable_identity() -> None:
    """The control: two builds of the same model agree on the digest."""
    assert _durable_digest(model=_law_model(law=_laws(die=_die)), params=_PARAMS) == (
        _durable_digest(model=_law_model(law=_laws(die=_die)), params=_PARAMS)
    )


@pytest.mark.parametrize("initial_frame", [False, True], ids=["mapping", "frame"])
@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_refused_start_raises_before_any_regime_law_is_evaluated(
    *, log_level: str, initial_frame: bool
) -> None:
    """Admission runs first: a refused start never reaches a user regime law."""
    calls: list[None] = []

    def counting_stay() -> FloatND:
        calls.append(None)
        return jnp.asarray(0.5)

    model = Model(
        regimes={
            "island": _island(),
            "working": Regime(
                regime_transitions=ByAge.until(
                    stop_age_exclusive=75,
                    law={
                        "working": MarkovTransition(func=counting_stay),
                        "dead": MarkovTransition(func=_die),
                    },
                    then="dead",
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": fixed_transition("wealth")},
                functions={"utility": _utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        ages=_AGES,
        regime_id_class=_Life,
        initial_regimes={25: "working"},
    )
    initial_conditions: Any = {
        "wealth": np.zeros(2),
        "age": np.full(2, 35.0),
        "regime_id": np.full(2, model.regime_names_to_ids["working"]),
    }
    if initial_frame:
        initial_conditions = pd.DataFrame(
            {"wealth": [0.0, 0.0], "age": [35.0, 35.0], "regime_name": ["working"] * 2}
        )
    with pytest.raises(InvalidInitialConditionsError, match=r"\(35, 'working'\)"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=initial_conditions,
            log_level=log_level,  # ty: ignore[invalid-argument-type]
            seed=0,
        )
    assert calls == []


def _stay_with_wealth(wealth: ContinuousState) -> FloatND:
    return wealth


def _die_with_wealth(wealth: ContinuousState) -> FloatND:
    return 1 - wealth


def _drift(wealth: ContinuousState) -> ContinuousState:
    return wealth + 0.75


def _drifting_model() -> Model:
    """Valid regime-law rows on the wealth grid [0, 1]; simulated wealth leaves it."""
    return Model(
        regimes={
            "island": _island(),
            "working": Regime(
                regime_transitions=ByAge.until(
                    stop_age_exclusive=75,
                    law={
                        "working": MarkovTransition(func=_stay_with_wealth),
                        "dead": MarkovTransition(func=_die_with_wealth),
                    },
                    then="dead",
                ),
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _drift},
                functions={"utility": _utility},
            ),
            "dead": Regime(
                regime_transitions=None,
                states={"wealth": _WEALTH},
                functions={"utility": _utility},
            ),
        },
        ages=_AGES,
        regime_id_class=_Life,
        initial_regimes={25: "working"},
    )


@pytest.mark.parametrize("log_level", ["off", "warning"])
def test_simulate_refuses_an_invalid_law_row_off_the_grid(*, log_level: str) -> None:
    """A realized row outside every grid row with mass outside [0, 1] raises."""
    model = _drifting_model()
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match="outside \\[0, 1\\]"
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions={
                "wealth": jnp.full(_N_SUBJECTS, 0.5),
                "age": jnp.full(_N_SUBJECTS, 25.0),
                "regime_id": jnp.full(
                    _N_SUBJECTS, model.regime_names_to_ids["working"]
                ),
            },
            log_level=log_level,  # ty: ignore[invalid-argument-type]
            seed=0,
        )


def test_simulate_accepts_valid_law_rows_off_the_grid() -> None:
    """The control: the same model starting where every row stays valid."""
    model = _drifting_model()
    model.simulate(
        params=_PARAMS,
        initial_conditions={
            "wealth": jnp.full(2, 0.0),
            "age": jnp.full(2, 25.0),
            "regime_id": jnp.full(2, model.regime_names_to_ids["working"]),
        },
        log_level="off",
        seed=0,
    )
