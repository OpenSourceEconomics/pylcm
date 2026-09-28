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
from lcm.exceptions import InvalidInitialConditionsError
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
