"""Tests for lazy NaN diagnostic enrichment in validate_V."""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from lcm import Model, Regime, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import InvalidValueFunctionError
from lcm.grids import LinSpacedGrid
from lcm.interfaces import StateActionSpace
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from lcm.utils.error_handling import validate_V


def _make_state_action_space(
    *,
    n_wealth: int = 3,
    n_consumption: int = 2,
) -> StateActionSpace:
    """Build a minimal `StateActionSpace` used across diagnostic tests."""
    return StateActionSpace(
        states=MappingProxyType(
            {"wealth": jnp.linspace(1.0, 5.0, n_wealth)},
        ),
        discrete_actions=MappingProxyType({}),
        continuous_actions=MappingProxyType(
            {"consumption": jnp.linspace(0.1, 2.0, n_consumption)},
        ),
        state_and_discrete_action_names=("wealth",),
    )


def _make_nan_V(n_wealth: int = 3) -> jnp.ndarray:
    """Create a V array with NaN to trigger diagnostics."""
    return jnp.full(n_wealth, jnp.nan)


def test_diagnostic_arrays_have_state_action_grid_shape():
    """Diagnostic by_dim breakdown has entries for each state and action."""
    n_wealth, n_consumption = 3, 2
    sas = _make_state_action_space(n_wealth=n_wealth, n_consumption=n_consumption)

    def mock_compute_intermediates(**kwargs: jnp.ndarray) -> dict:  # noqa: ARG001
        # Return the fused-reduction dict the real closure produces after
        # productmap-wrapping + on-device reduction.
        return {
            "U_nan_overall": jnp.array(0.0),
            "U_nan_by_wealth": jnp.zeros(n_wealth),
            "U_nan_by_consumption": jnp.zeros(n_consumption),
            "E_nan_overall": jnp.array(0.0),
            "E_nan_by_wealth": jnp.zeros(n_wealth),
            "E_nan_by_consumption": jnp.zeros(n_consumption),
            "Q_nan_overall": jnp.array(0.0),
            "Q_nan_by_wealth": jnp.zeros(n_wealth),
            "Q_nan_by_consumption": jnp.zeros(n_consumption),
            "F_feasible_overall": jnp.array(1.0),
            "F_feasible_by_wealth": jnp.ones(n_wealth),
            "F_feasible_by_consumption": jnp.ones(n_consumption),
            "regime_probs": MappingProxyType({"alive": jnp.array(1.0)}),
        }

    with pytest.raises(InvalidValueFunctionError) as exc_info:
        validate_V(
            V_arr=_make_nan_V(3),
            age=0.0,
            regime_name="alive",
            partial_solution=MappingProxyType({}),
            compute_intermediates=mock_compute_intermediates,
            state_action_space=sas,
            next_regime_to_V_arr=MappingProxyType(
                {"alive": jnp.zeros(3)},
            ),
            internal_params=MappingProxyType({}),
        )

    exc = exc_info.value
    assert exc.diagnostics is not None
    diagnostics: dict = exc.diagnostics  # ty: ignore[invalid-assignment]
    u_by_dim = diagnostics["U_nan_fraction"]["by_dim"]
    assert "wealth" in u_by_dim, f"Expected 'wealth' in by_dim, got: {u_by_dim}"
    assert "consumption" in u_by_dim, (
        f"Expected 'consumption' in by_dim, got: {u_by_dim}"
    )


def _build_nan_model() -> tuple[Model, dict]:
    """Build a minimal model that produces NaN in V during backward induction."""

    @categorical(ordered=False)
    class _Rid:
        non_terminal: int
        terminal: int

    def utility(
        consumption: ContinuousAction,
        wealth: ContinuousState,
    ) -> FloatND:
        nan_term = jnp.where(wealth < 1.1, jnp.nan, 0.0)
        return jnp.log(consumption) + nan_term

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    def next_regime(period: int, n_periods: int) -> ScalarInt:
        return jnp.where(period == (n_periods - 2), 1, 0)

    def borrowing_constraint(
        consumption: ContinuousAction, wealth: ContinuousState
    ) -> BoolND:
        return consumption <= wealth

    non_terminal = Regime(
        actions={"consumption": LinSpacedGrid(start=1, stop=2, n_points=3)},
        states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=3)},
        state_transitions={"wealth": next_wealth},
        functions={"utility": utility},
        constraints={"borrowing_constraint": borrowing_constraint},
        transition=next_regime,
        active=lambda age: age < 1,
    )
    terminal = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age >= 1,
    )
    model = Model(
        regimes={"non_terminal": non_terminal, "terminal": terminal},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_Rid,
    )
    params = {
        "discount_factor": 0.95,
        "non_terminal": {"next_regime": {"n_periods": 2}},
        "terminal": {},
    }
    return model, params


def test_nan_diagnostics_end_to_end() -> None:
    """Real model: `model.solve()` attaches a diagnostics dict when V has NaN.

    This exercises the full build → productmap → reduction → summarize
    chain. If `_build_compute_intermediates_per_period` does not produce
    a dict-returning closure, `_summarize_diagnostics` silently fails
    (broad try/except) and `exc.diagnostics` is missing.
    """
    model, params = _build_nan_model()

    with pytest.raises(InvalidValueFunctionError) as exc_info:
        model.solve(params=params)

    exc = exc_info.value
    assert exc.diagnostics is not None, (
        "Diagnostic enrichment failed: exception has no diagnostics attribute. "
        "Likely cause: compute_intermediates closure returns a tuple but "
        "_summarize_diagnostics expects a dict — see _wrap_with_reduction."
    )
    diagnostics: dict = exc.diagnostics  # ty: ignore[invalid-assignment]
    assert "U_nan_fraction" in diagnostics
    by_dim = diagnostics["U_nan_fraction"]["by_dim"]
    assert "wealth" in by_dim
    assert "consumption" in by_dim


def test_diagnostic_failure_preserves_original_error():
    """If diagnostics crash, the original InvalidValueFunctionError survives."""
    sas = _make_state_action_space()

    def broken_compute_intermediates(**kwargs: jnp.ndarray) -> None:  # noqa: ARG001
        msg = "intentional diagnostic failure"
        raise RuntimeError(msg)

    with pytest.raises(InvalidValueFunctionError, match="NaN"):
        validate_V(
            V_arr=_make_nan_V(),
            age=0.0,
            regime_name="test",
            partial_solution=MappingProxyType({}),
            compute_intermediates=broken_compute_intermediates,
            state_action_space=sas,
            next_regime_to_V_arr=MappingProxyType(
                {"test": jnp.zeros(3)},
            ),
            internal_params=MappingProxyType({}),
        )
