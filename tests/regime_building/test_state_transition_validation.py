"""Automatic state transition probability validation.

Exercises the pre-solve sweep `validate_state_transitions_all_periods`
(`regime_building/runtime_checks.py`) and the process-time AST + n_outcomes
derivation (`regime_building/static_checks.py`).
"""

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.exceptions import InvalidStateTransitionProbabilitiesError
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from lcm.user_regime import Regime as UserRegime


@categorical(ordered=False)
class _Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    terminal: ScalarInt


WEALTH_GRID = LinSpacedGrid(start=1, stop=10, n_points=3)
CONSUMPTION_GRID = LinSpacedGrid(start=1, stop=5, n_points=3)


def _next_wealth(wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _budget(wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def _next_regime(age: float) -> ScalarInt:  # noqa: ARG001
    # Alive is active only at age 0, so the next-period regime is always
    # the terminal one — keeping this transition simple lets the tests
    # focus on the state-transition validator rather than regime
    # bookkeeping.
    return jnp.asarray(_RegimeId.terminal)


def _utility_alive(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_terminal(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _terminal_regime() -> UserRegime:
    return UserRegime(
        transition=None,
        functions={"utility": _utility_terminal},
        states={"wealth": WEALTH_GRID},
        active=lambda age: age >= 1,
    )


def _model_with_state_probs(next_health_func) -> Model:
    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": MarkovTransition(next_health_func),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
        transition=_next_regime,
        active=lambda age: age < 1,
    )
    return Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_valid_state_probs_pass() -> None:
    """A correct MarkovTransition function passes pre-solve validation."""

    def good_health_probs(health: DiscreteState) -> FloatND:
        return jnp.where(
            health == _Health.good,
            jnp.array([0.2, 0.8]),
            jnp.array([0.6, 0.4]),
        )

    model = _model_with_state_probs(good_health_probs)
    model.solve(params={"discount_factor": 0.95})


def test_runtime_check_raises_on_wrong_outcome_axis_size() -> None:
    """Wrong outcome-axis size (length 3 instead of 2) surfaces at solve."""

    def too_many_outcomes(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.4, 0.4, 0.2])

    model = _model_with_state_probs(too_many_outcomes)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="outcome axis"):
        model.solve(params={"discount_factor": 0.95})


def test_runtime_check_raises_on_values_out_of_range() -> None:
    """Negative or >1 probability values surface at solve."""

    def negative_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([1.2, -0.2])

    model = _model_with_state_probs(negative_probs)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match=r"\[0, 1\]"):
        model.solve(params={"discount_factor": 0.95})


def test_runtime_check_raises_on_rows_not_summing_to_one() -> None:
    """A row that sums to 0.7 surfaces at solve."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"):
        model.solve(params={"discount_factor": 0.95})


def test_log_level_off_skips_runtime_check() -> None:
    """A model whose state probs violate sum-to-1 still solves at log_level='off'."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    # With log_level='off' the runtime numerical check is skipped — solve
    # returns a (numerically dubious) V_arr rather than raising.
    model.solve(params={"discount_factor": 0.95}, log_level="off")


@pytest.mark.parametrize("log_level", ["warning", "progress", "debug"])
def test_runtime_check_runs_at_all_non_off_log_levels(log_level: str, tmp_path) -> None:
    """Validation fires at every log level except 'off'."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    kwargs: dict = {"params": {"discount_factor": 0.95}, "log_level": log_level}
    if log_level == "debug":
        kwargs["log_path"] = tmp_path
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"):
        model.solve(**kwargs)


def test_subscript_order_swap_raises_at_process_time() -> None:
    """Mismatched subscript order vs signature order raises at process time."""

    @categorical(ordered=False)
    class _Local:
        bad: ScalarInt
        good: ScalarInt

    @categorical(ordered=False)
    class _LocalRegimeId:
        alive: ScalarInt
        terminal: ScalarInt

    def swapped_probs(
        period: ScalarInt,
        health: DiscreteState,
        probs_array: FloatND,
    ) -> FloatND:
        # Subscripts in wrong order: signature is (period, health) but
        # body indexes as [health, period].
        return probs_array[health, period]

    def _local_next_regime(age: float) -> ScalarInt:  # noqa: ARG001
        return jnp.asarray(_LocalRegimeId.terminal)

    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(_Local)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": MarkovTransition(swapped_probs),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
        transition=_local_next_regime,
        active=lambda age: age < 1,
    )
    terminal = UserRegime(
        transition=None,
        functions={"utility": _utility_terminal},
        states={"wealth": WEALTH_GRID},
        active=lambda age: age >= 1,
    )

    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="subscript"):
        Model(
            regimes={"alive": alive, "terminal": terminal},
            ages=AgeGrid(start=0, stop=2, step="Y"),
            regime_id_class=_LocalRegimeId,
        )


def test_ast_check_is_permissive_when_no_probs_array_subscript() -> None:
    """A function without `probs_array[...]` survives the static AST check."""

    def no_subscript_probs(health: DiscreteState) -> FloatND:
        # No `probs_array[...]` pattern — the AST check silently skips.
        return jnp.where(
            health == _Health.good,
            jnp.array([0.1, 0.9]),
            jnp.array([0.7, 0.3]),
        )

    # Model construction must not raise just because the function lacks
    # the subscript pattern; runtime numerical checks still apply.
    model = _model_with_state_probs(no_subscript_probs)
    model.solve(params={"discount_factor": 0.95})


def test_per_target_dict_validates_each_entry() -> None:
    """Each MarkovTransition inside a per-target dict is validated independently."""

    @categorical(ordered=False)
    class _Heir:
        no: ScalarInt
        yes: ScalarInt

    @categorical(ordered=False)
    class _RegId:
        alive: ScalarInt
        dead: ScalarInt

    def bad_heir_probs(wealth: ContinuousState) -> FloatND:  # noqa: ARG001
        # Rows don't sum to 1 — should be caught even though heir_present
        # lives in the target regime, not the source.
        return jnp.array([0.5, 0.3])

    def next_wealth_passthrough(wealth: ContinuousState) -> ContinuousState:
        return wealth

    def _utility_alive(wealth: ContinuousState) -> FloatND:
        return wealth

    def _utility_dead(wealth: ContinuousState, heir_present: DiscreteState) -> FloatND:
        return wealth * heir_present

    def _to_dead(age: float) -> ScalarInt:  # noqa: ARG001
        return jnp.asarray(_RegId.dead)

    alive = UserRegime(
        functions={"utility": _utility_alive},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=3)},
        state_transitions={
            "wealth": next_wealth_passthrough,
            "heir_present": {"dead": MarkovTransition(bad_heir_probs)},
        },
        transition=_to_dead,
        active=lambda age: age < 1,
    )
    dead = UserRegime(
        transition=None,
        functions={"utility": _utility_dead},
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=3),
            "heir_present": DiscreteGrid(_Heir),
        },
        active=lambda age: age >= 1,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegId,
    )
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"):
        model.solve(params={"discount_factor": 0.95})


def test_model_with_no_markov_transitions_solves_normally() -> None:
    """A purely deterministic model is unaffected (fast-exit in validator)."""

    alive = UserRegime(
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
        transition=_next_regime,
        active=lambda age: age < 1,
    )
    model = Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
    )
    model.solve(params={"discount_factor": 0.95})
