"""Automatic state transition probability validation.

Exercises the pre-solve numerical sweep over `MarkovTransition` state
transitions, the process-time AST subscript-order check, and the way the
`log_level` validation policy turns failures into warnings or raises.
"""

import logging
from pathlib import Path

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
from lcm.utils.logging import LogLevel


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


def test_valid_state_probs_at_boundary_pass() -> None:
    """Inclusive [0, 1] bounds and a row sum within the 1e-6 tolerance pass.

    For `health == good` the row is exactly `[0.0, 1.0]` — values at the
    inclusive bounds. For `health == bad` the row sums to `1 - 5e-7`, just
    inside the `atol=1e-6` row-sum tolerance. Validation must accept both
    without raising.
    """

    def boundary_health_probs(health: DiscreteState) -> FloatND:
        return jnp.where(
            health == _Health.good,
            jnp.array([0.0, 1.0]),
            jnp.array([0.5, 0.4999995]),
        )

    model = _model_with_state_probs(boundary_health_probs)
    model.solve(log_level="debug", params={"discount_factor": 0.95})


def test_runtime_check_catches_invalidity_hidden_at_some_grid_points() -> None:
    """An ensemble valid at some continuous-grid points and invalid at others raises.

    The `MarkovTransition` for `health` is conditioned on the continuous
    `wealth` grid: it returns a valid row for `wealth <= 5` and a row summing
    to 0.7 for `wealth > 5`. Only sweeping the full `wealth` grid surfaces the
    failure — a spot check at the first grid point (`wealth == 1`) would pass.
    """

    def sneaky_health_probs(
        wealth: ContinuousState,
        health: DiscreteState,  # noqa: ARG001
    ) -> FloatND:
        return jnp.where(
            wealth > 5.0,
            jnp.array([0.5, 0.2]),
            jnp.array([0.3, 0.7]),
        )

    model = _model_with_state_probs(sneaky_health_probs)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def test_runtime_check_raises_on_wrong_outcome_axis_size() -> None:
    """Wrong outcome-axis size (length 3 instead of 2) surfaces at solve."""

    def too_many_outcomes(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.4, 0.4, 0.2])

    model = _model_with_state_probs(too_many_outcomes)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="outcome axis"):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def test_runtime_check_raises_on_values_out_of_range() -> None:
    """Negative or >1 probability values surface at solve."""

    def negative_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([1.2, -0.2])

    model = _model_with_state_probs(negative_probs)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match=r"\[0, 1\]"):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def test_runtime_check_raises_on_rows_not_summing_to_one() -> None:
    """A row that sums to 0.7 surfaces at solve."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def test_log_level_off_skips_runtime_check() -> None:
    """A model whose state probs violate sum-to-1 still solves at log_level='off'."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    # With log_level='off' the runtime numerical check is skipped — solve
    # returns a (numerically dubious) V_arr rather than raising.
    model.solve(params={"discount_factor": 0.95}, log_level="off")


@pytest.mark.parametrize("log_level", ["warning", "progress"])
def test_warn_levels_log_invalid_probs_and_continue(
    log_level: LogLevel, caplog: pytest.LogCaptureFixture
) -> None:
    """At 'warning'/'progress', invalid state probs log a warning; solve continues."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(params={"discount_factor": 0.95}, log_level=log_level)
    assert "sum to 1" in caplog.text


def test_debug_level_raises_on_invalid_probs() -> None:
    """At log_level='debug', invalid state probs raise rather than warn."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"):
        model.solve(params={"discount_factor": 0.95}, log_level="debug")


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
    model.solve(log_level="debug", params={"discount_factor": 0.95})


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
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def _good_health_probs(health: DiscreteState) -> FloatND:
    return jnp.where(
        health == _Health.good,
        jnp.array([0.2, 0.8]),
        jnp.array([0.6, 0.4]),
    )


@pytest.mark.parametrize(
    ("log_level", "expect_snapshot"),
    [
        ("off", False),
        ("warning", False),
        ("progress", False),
        ("debug", True),
    ],
)
def test_snapshot_written_only_at_debug_on_valid_solve(
    log_level: LogLevel,
    expect_snapshot: bool,  # noqa: FBT001
    tmp_path: Path,
) -> None:
    """With `log_path` set, a valid solve writes a snapshot only at `"debug"`.

    Pins the snapshot column of the `log_level` x `log_path` table for a solve
    that produces no NaN: `"debug"` snapshots every call, the warn/off levels
    do not.
    """
    model = _model_with_state_probs(_good_health_probs)
    model.solve(
        params={"discount_factor": 0.95}, log_level=log_level, log_path=tmp_path
    )
    snapshots = list(tmp_path.glob("solve_snapshot_*"))
    assert bool(snapshots) is expect_snapshot


def test_warn_mode_writes_snapshot_on_nan_failure(tmp_path: Path) -> None:
    """At `"warning"` with `log_path` set, a NaN solve writes a snapshot.

    Pins the "one per warned failure" snapshot-table cell: warn mode does not
    raise, so the snapshot is the only on-disk record of the failed solve.
    """
    model = _model_with_state_probs(_good_health_probs)
    model.solve(
        params={"discount_factor": float("nan")},
        log_level="warning",
        log_path=tmp_path,
    )
    assert list(tmp_path.glob("solve_snapshot_*"))


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
    model.solve(log_level="debug", params={"discount_factor": 0.95})


def _model_with_fixed_param_health_probs() -> Model:
    """Build a model whose `health` MarkovTransition reads from `fixed_params`.

    `transition_bias` lives in `fixed_params`, not the per-iteration
    `params` dict. Solve sees it via `regime.resolved_fixed_params`; the
    pre-solve numerical validator must do the same merge.
    """

    def health_probs(health: DiscreteState, transition_bias: float) -> FloatND:
        good_row = jnp.array([0.5 - transition_bias, 0.5 + transition_bias])
        bad_row = jnp.array([0.5, 0.5])
        return jnp.where(health == _Health.good, good_row, bad_row)

    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": MarkovTransition(health_probs),
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
        fixed_params={"transition_bias": 0.1},
    )


def test_state_validator_resolves_params_from_fixed_params(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A MarkovTransition reading a `fixed_params` entry is numerically validated.

    The skip-and-warn branch must not fire just because the parameter
    sits in `fixed_params` rather than the per-iteration `params` dict —
    both belong to the namespace solve resolves against.
    """
    model = _model_with_fixed_param_health_probs()

    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(log_level="warning", params={"discount_factor": 0.95})

    skips = [r for r in caplog.records if "not numerically validated" in r.message]
    assert not skips, f"Validator skipped: {skips[0].message}"


def _model_with_per_target_fixed_param_health_probs() -> Model:
    """Build a model whose per-target `health` transition reads from `fixed_params`.

    `health` exists only in the `terminal` regime; the source `alive`
    regime declares its initialisation via a per-target dict keyed by
    target regime name. The transition function takes `transition_bias`
    from `fixed_params`. Solve resolves it via the per-target qualified
    name `to_terminal_next_health__transition_bias`; the validator must
    strip the same prefix or fall through to the skip-and-warn branch.
    """

    def health_probs(transition_bias: float) -> FloatND:
        return jnp.array([0.5 - transition_bias, 0.5 + transition_bias])

    def _utility_terminal_with_health(
        wealth: ContinuousState,
        health: DiscreteState,  # noqa: ARG001
    ) -> FloatND:
        return jnp.log(wealth)

    alive = UserRegime(
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": {"terminal": MarkovTransition(health_probs)},
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
        transition=_next_regime,
        active=lambda age: age < 1,
    )
    terminal = UserRegime(
        transition=None,
        functions={"utility": _utility_terminal_with_health},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(_Health)},
        active=lambda age: age >= 1,
    )
    return Model(
        regimes={"alive": alive, "terminal": terminal},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        fixed_params={"transition_bias": 0.1},
    )


def test_per_target_state_validator_resolves_params_from_fixed_params(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-target dict MarkovTransitions are numerically validated like simple ones.

    The skip-and-warn branch must not fire just because the source regime
    declares the transition under a per-target dict — the validator
    strips the per-target qualified prefix
    `to_<target>_next_<state>__` the same way it strips the simple
    `next_<state>__` prefix for non-per-target transitions.
    """
    model = _model_with_per_target_fixed_param_health_probs()

    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(log_level="warning", params={"discount_factor": 0.95})

    skips = [r for r in caplog.records if "not numerically validated" in r.message]
    assert not skips, f"Validator skipped: {skips[0].message}"


def test_state_validator_catches_bad_probs_when_using_fixed_param() -> None:
    """Invalid probs are still surfaced when the transition reads from `fixed_params`.

    Proves the merged-namespace fix doesn't just silence the skip-warning
    but actually runs the numerical check.
    """

    def bad_health_probs(health: DiscreteState, transition_bias: float) -> FloatND:
        # Bias is added to row 0 only, so `transition_bias=0.6` makes the
        # `good` row sum to 1.6 — well outside the row-sum tolerance.
        return jnp.where(
            health == _Health.good,
            jnp.array([0.5 + transition_bias, 0.5]),
            jnp.array([0.5, 0.5]),
        )

    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": MarkovTransition(bad_health_probs),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
        transition=_next_regime,
        active=lambda age: age < 1,
    )
    model = Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=_RegimeId,
        fixed_params={"transition_bias": 0.6},
    )

    with pytest.raises(InvalidStateTransitionProbabilitiesError):
        model.solve(log_level="debug", params={"discount_factor": 0.95})
