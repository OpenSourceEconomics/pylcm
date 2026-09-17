"""Per-subject action feasibility of initial conditions."""

import jax.numpy as jnp
import pytest

from _lcm.params.processing import process_params
from _lcm.simulation import initial_conditions as preflight
from _lcm.simulation.initial_conditions import (
    initial_conditions_feasibility_mask,
    validate_initial_conditions,
)
from lcm import IrregSpacedGrid, LinSpacedGrid
from lcm.exceptions import InvalidInitialConditionsError
from tests.simulation.initial_conditions._models import (
    make_constrained_asymmetric_model,
    make_constraint_model,
    make_state_only_constraint_model,
)


def test_infeasible_initial_states_detected():
    """Wealth below the constraint threshold makes all actions infeasible.

    wealth=0.25 < min consumption (0.5), so consumption <= wealth is always False.
    """
    model = make_constraint_model(
        wealth_grid=LinSpacedGrid(start=2.0, stop=10, n_points=15)
    )
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    _working_life = model.regime_names_to_ids["working_life"]
    with pytest.raises(InvalidInitialConditionsError):
        model.simulate(
            log_level="debug",
            params=params,
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([0.25]),
                "regime_id": jnp.array([_working_life]),
            },
        )


def test_on_grid_state_but_combination_infeasible():
    """State ON the grid but constraint fails for ALL action combinations.

    wealth=0.3 is the grid minimum, but min consumption (0.5) > 0.3,
    so consumption <= wealth is always False.
    """
    model = make_constraint_model(
        wealth_grid=LinSpacedGrid(start=0.3, stop=10, n_points=15)
    )
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    _working_life = model.regime_names_to_ids["working_life"]
    with pytest.raises(InvalidInitialConditionsError):
        model.simulate(
            log_level="debug",
            params=params,
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([0.3]),
                "regime_id": jnp.array([_working_life]),
            },
        )


def test_extrapolated_initial_states_accepted():
    """wealth=1.0 is above constraint threshold but below grid min — feasible."""
    model = make_constraint_model(
        wealth_grid=LinSpacedGrid(start=2.0, stop=10, n_points=15)
    )
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    _working_life = model.regime_names_to_ids["working_life"]
    model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([1.0]),
            "regime_id": jnp.array([_working_life]),
        },
    )


def test_on_grid_initial_states_accepted():
    """wealth=5.0 is above grid min — fully on grid, feasible."""
    model = make_constraint_model(
        wealth_grid=LinSpacedGrid(start=2.0, stop=10, n_points=15)
    )
    params = {
        "discount_factor": 0.95,
        "working_life": {"next_regime": {"final_age_alive": 1}},
    }
    _working_life = model.regime_names_to_ids["working_life"]
    model.simulate(
        log_level="debug",
        params=params,
        initial_conditions={
            "age": jnp.array([0.0]),
            "wealth": jnp.array([5.0]),
            "regime_id": jnp.array([_working_life]),
        },
    )


def test_irreg_spaced_grid_with_runtime_points():
    """Feasibility check works when grid points are supplied at runtime via params."""
    model = make_constraint_model(wealth_grid=IrregSpacedGrid(n_points=15))
    params = {
        "discount_factor": 0.95,
        "working_life": {
            "wealth": {"points": jnp.linspace(0.3, 10, 15)},
            "next_regime": {"final_age_alive": 1},
        },
    }
    _working_life = model.regime_names_to_ids["working_life"]
    with pytest.raises(InvalidInitialConditionsError):
        model.simulate(
            log_level="debug",
            params=params,
            initial_conditions={
                "wealth": jnp.array([0.3]),
                "regime_id": jnp.array([_working_life]),
            },
        )


def test_constraint_not_checked_for_unused_regime() -> None:
    """Subject in dead (no constraint); wealth=40 fine even though alive needs > 50."""
    model = make_constrained_asymmetric_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _dead = model.regime_names_to_ids["dead"]

    validate_initial_conditions(
        initial_conditions={
            "age": jnp.array([2.0]),
            "wealth": jnp.array([40.0]),
            "regime_id": jnp.array([_dead]),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )


def test_constraint_checked_for_starting_regime() -> None:
    """Subject in alive (has constraint); wealth=40 is infeasible."""
    model = make_constrained_asymmetric_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _alive = model.regime_names_to_ids["alive"]

    with pytest.raises(InvalidInitialConditionsError, match="infeasible"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0]),
                "wealth": jnp.array([40.0]),
                "regime_id": jnp.array([_alive]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_mixed_regimes_constraint_only_checked_for_starting_regime() -> None:
    """One subject in alive (infeasible), one in dead (no constraint).

    wealth=40 violates alive's constraint; dead declares no constraint to check.
    """
    model = make_constrained_asymmetric_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _alive = model.regime_names_to_ids["alive"]
    _dead = model.regime_names_to_ids["dead"]

    with pytest.raises(InvalidInitialConditionsError, match="infeasible"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([0.0, 2.0]),
                "wealth": jnp.array([40.0, 40.0]),
                "regime_id": jnp.array([_alive, _dead]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_action_free_regime_state_only_constraint_rejects_violating_subject() -> None:
    """A subject of an action-free regime violating a state constraint is infeasible."""
    model = make_state_only_constraint_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _dead = model.regime_names_to_ids["dead"]

    with pytest.raises(InvalidInitialConditionsError, match="infeasible"):
        validate_initial_conditions(
            initial_conditions={
                "age": jnp.array([2.0, 2.0]),
                "wealth": jnp.array([-1.0, 5.0]),
                "regime_id": jnp.array([_dead, _dead]),
            },
            regimes=model._regimes,
            regime_names_to_ids=model.regime_names_to_ids,
            flat_params=flat_params,
            ages=model.ages,
        )


def test_action_free_regime_state_only_constraint_accepts_satisfying_subjects() -> None:
    """Subjects in an action-free regime whose states satisfy the constraint pass."""
    model = make_state_only_constraint_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _dead = model.regime_names_to_ids["dead"]

    validate_initial_conditions(
        initial_conditions={
            "age": jnp.array([2.0, 2.0]),
            "wealth": jnp.array([1.0, 5.0]),
            "regime_id": jnp.array([_dead, _dead]),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )


@pytest.mark.parametrize("device_memory_bytes", [None, 2**24])
def test_simulate_rejects_state_only_violation_in_action_free_regime(
    device_memory_bytes: int | None,
) -> None:
    """`simulate` refuses a subject whose action-free regime constraint fails."""
    model = make_state_only_constraint_model(device_memory_bytes=device_memory_bytes)
    _dead = model.regime_names_to_ids["dead"]

    with pytest.raises(InvalidInitialConditionsError, match="infeasible"):
        model.simulate(
            log_level="debug",
            params={"discount_factor": 0.95},
            initial_conditions={
                "age": jnp.array([2.0]),
                "wealth": jnp.array([-1.0]),
                "regime_id": jnp.array([_dead]),
            },
        )


def test_budgeted_simulate_accepts_action_free_regime_without_serial_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under a memory budget, a valid action-free population is admitted in one pass.

    The reduced host summary answers for the action-free regime; the serial
    diagnostic validator is never consulted.
    """

    def forbidden(**kwargs: object) -> None:
        del kwargs
        pytest.fail("Serial validation ran for a valid action-free population.")

    monkeypatch.setattr(preflight, "validate_initial_conditions", forbidden)
    model = make_state_only_constraint_model(device_memory_bytes=2**24)
    _dead = model.regime_names_to_ids["dead"]

    result = model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.array([2.0, 2.0]),
            "wealth": jnp.array([1.0, 5.0]),
            "regime_id": jnp.array([_dead, _dead]),
        },
    )

    assert result.to_dataframe()["wealth"].tolist() == [1.0, 5.0]


def test_feasibility_mask_marks_each_subject_in_caller_order() -> None:
    """The mask is `True` exactly for subjects admitting an action, in input order."""
    model = make_constraint_model(
        wealth_grid=LinSpacedGrid(start=2.0, stop=10, n_points=15)
    )
    flat_params = process_params(
        params={
            "discount_factor": 0.95,
            "working_life": {"next_regime": {"final_age_alive": 1}},
        },
        params_template=model._params_template,
    )
    _working_life = model.regime_names_to_ids["working_life"]

    mask = initial_conditions_feasibility_mask(
        initial_conditions={
            "age": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "wealth": jnp.array([0.25, 5.0, 0.3, 7.0]),
            "regime_id": jnp.array([_working_life] * 4),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )

    assert mask.tolist() == [False, True, False, True]


def test_feasibility_mask_scatters_regime_verdicts_into_interleaved_rows() -> None:
    """Subjects of different regimes keep their own verdicts at their own rows."""
    model = make_state_only_constraint_model()
    flat_params = process_params(
        params={"discount_factor": 0.95}, params_template=model._params_template
    )
    _alive = model.regime_names_to_ids["alive"]
    _dead = model.regime_names_to_ids["dead"]

    mask = initial_conditions_feasibility_mask(
        initial_conditions={
            "age": jnp.array([0.0, 2.0, 1.0, 2.0]),
            "wealth": jnp.array([0.5, -1.0, 60.0, 5.0]),
            "regime_id": jnp.array([_alive, _dead, _alive, _dead]),
        },
        regimes=model._regimes,
        regime_names_to_ids=model.regime_names_to_ids,
        flat_params=flat_params,
        ages=model.ages,
    )

    assert mask.tolist() == [False, False, True, True]
