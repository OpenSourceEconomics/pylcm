"""Automatic state transition probability validation.

Exercises the pre-solve numerical sweep over `StochasticTransition` state
transitions, the process-time AST subscript-order check, and the way the
Enabled runtime checks raise independently of console verbosity.
"""

import logging
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NotRequired, Unpack
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
from typing_extensions import TypedDict

from _lcm import transition_checks as transition_checks_module
from _lcm.engine import Regime, _StochasticStateTransition
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.simulation import initial_conditions as initial_conditions_module
from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.host_operations import ProfiledSimulationOperations
from _lcm.simulation.memory import SimulationMemory
from _lcm.simulation.residency import (
    DeviceBufferFootprint,
    measure_buffer_footprint,
    resident_bytes_by_device,
)
from _lcm.solution import backward_induction as backward_induction_module
from _lcm.time import TimeAxis
from _lcm.typing import (
    EconFunctionsMapping,
    FlatParams,
    FlatRegimeParams,
    FootprintTree,
    InitialConditions,
    RegimeNamesToIds,
    StateOrActionName,
)
from _lcm.utils.logging import LogLevel
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    DiscreteGrid,
    ExecutionConfig,
    JointTransition,
    LinSpacedGrid,
    Model,
    Phased,
    StochasticTransition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidStateTransitionProbabilitiesError,
    InvalidValueFunctionError,
)
from lcm.regime import FunctionEntry, StateEntry, StateTransitionEntry
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    RegimeName,
    ScalarInt,
)


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


def _next_wealth(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def _budget(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def _utility_alive(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_terminal(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


def _terminal_regime() -> UserRegime:
    return UserRegime(
        functions={"utility": _utility_terminal},
        states={"wealth": WEALTH_GRID},
    )


def _model_with_state_probs(next_health_func) -> Model:
    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(category_class=_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": StochasticTransition(func=next_health_func),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
    )
    return Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
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

    The `StochasticTransition` for `health` is conditioned on the continuous
    `wealth` grid: it returns a valid row for `wealth <= 5` and a row summing
    to 0.7 for `wealth > 5`. Only sweeping the full `wealth` grid surfaces the
    failure — a spot check at the first grid point (`wealth == 1`) would pass.
    """

    def sneaky_health_probs(
        *,
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


@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_runtime_checks_disabled_skip_probability_check(log_level: LogLevel) -> None:
    """Disabled runtime checks permit non-unit state probabilities at every level."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    model.solve(
        params={"discount_factor": 0.95},
        log_level=log_level,
        runtime_checks=False,
    )


@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_runtime_checks_reject_invalid_probs_at_every_level(
    *, log_level: LogLevel, caplog: pytest.LogCaptureFixture
) -> None:
    """Enabled checks reject non-unit state probabilities at every verbosity."""

    def bad_sum_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.5, 0.2])

    model = _model_with_state_probs(bad_sum_probs)
    with (
        caplog.at_level(logging.WARNING, logger="lcm"),
        pytest.raises(InvalidStateTransitionProbabilitiesError, match="sum to 1"),
    ):
        model.solve(params={"discount_factor": 0.95}, log_level=log_level)


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
        *,
        period: ScalarInt,
        health: DiscreteState,
        probs_array: FloatND,
    ) -> FloatND:
        # Subscripts in wrong order: signature is (period, health) but
        # body indexes as [health, period].
        return probs_array[health, period]

    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(category_class=_Local)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": StochasticTransition(func=swapped_probs),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
    )
    terminal = UserRegime(
        functions={"utility": _utility_terminal},
        states={"wealth": WEALTH_GRID},
    )

    with pytest.raises(InvalidStateTransitionProbabilitiesError, match="subscript"):
        Model(
            regimes={"alive": alive, "terminal": terminal},
            ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
            regime_id_class=_LocalRegimeId,
            initial_nodes={0: "alive"},
            edges={"alive": {"terminal": 0}},
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
    """Each StochasticTransition inside a per-target dict is validated independently."""

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

    def _utility_dead(
        *, wealth: ContinuousState, heir_present: DiscreteState
    ) -> FloatND:
        return wealth * heir_present

    alive = UserRegime(
        functions={"utility": _utility_alive},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=3)},
        state_transitions={
            "wealth": next_wealth_passthrough,
            "heir_present": {"dead": StochasticTransition(func=bad_heir_probs)},
        },
    )
    dead = UserRegime(
        functions={"utility": _utility_dead},
        states={
            "wealth": LinSpacedGrid(start=1, stop=10, n_points=3),
            "heir_present": DiscreteGrid(category_class=_Heir),
        },
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegId,
        initial_nodes={0: "alive"},
        edges={"alive": {"dead": 0}},
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
    *,
    log_level: LogLevel,
    expect_snapshot: bool,
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


@pytest.mark.parametrize("log_level", ["off", "warning", "progress", "debug"])
def test_runtime_checks_write_snapshot_on_nan_failure(
    *, tmp_path: Path, log_level: LogLevel
) -> None:
    """Enabled NaN checks save the failure snapshot even with silent logging."""
    model = _model_with_state_probs(_good_health_probs)
    with pytest.raises(InvalidValueFunctionError):
        model.solve(
            params={"discount_factor": float("nan")},
            log_level=log_level,
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
    )
    model = Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
    )
    model.solve(log_level="debug", params={"discount_factor": 0.95})


def _model_with_fixed_param_health_probs() -> Model:
    """Build a model whose `health` StochasticTransition reads from `fixed_params`.

    `transition_bias` lives in `fixed_params`, not the per-iteration
    `params` dict. Solve sees it via `regime.resolved_fixed_params`; the
    pre-solve numerical validator must do the same merge.
    """

    def health_probs(*, health: DiscreteState, transition_bias: float) -> FloatND:
        good_row = jnp.array([0.5 - transition_bias, 0.5 + transition_bias])
        bad_row = jnp.array([0.5, 0.5])
        return jnp.where(health == _Health.good, good_row, bad_row)

    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(category_class=_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": StochasticTransition(func=health_probs),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
    )
    return Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        fixed_params={"transition_bias": 0.1},
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
    )


def test_state_validator_resolves_params_from_fixed_params(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A StochasticTransition reading a `fixed_params` entry is numerically validated.

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
        *,
        wealth: ContinuousState,
        health: DiscreteState,  # noqa: ARG001
    ) -> FloatND:
        return jnp.log(wealth)

    alive = UserRegime(
        states={"wealth": WEALTH_GRID},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": {"terminal": StochasticTransition(func=health_probs)},
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
    )
    terminal = UserRegime(
        functions={"utility": _utility_terminal_with_health},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(category_class=_Health)},
    )
    return Model(
        regimes={"alive": alive, "terminal": terminal},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        fixed_params={"transition_bias": 0.1},
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
    )


def test_per_target_state_validator_resolves_params_from_fixed_params(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Per-target dict StochasticTransitions are numerically validated like simple ones.

    The skip-and-warn branch must not fire just because the source regime
    declares the transition under a per-target dict — the validator
    strips the per-target qualified prefix
    `<target>__next_<state>__` the same way it resolves the params of
    coarse transitions.
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

    def bad_health_probs(*, health: DiscreteState, transition_bias: float) -> FloatND:
        # Bias is added to row 0 only, so `transition_bias=0.6` makes the
        # `good` row sum to 1.6 — well outside the row-sum tolerance.
        return jnp.where(
            health == _Health.good,
            jnp.array([0.5 + transition_bias, 0.5]),
            jnp.array([0.5, 0.5]),
        )

    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(category_class=_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": StochasticTransition(func=bad_health_probs),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
    )
    model = Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        fixed_params={"transition_bias": 0.6},
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
    )

    with pytest.raises(InvalidStateTransitionProbabilitiesError):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def test_a_subnormal_probability_is_a_valid_row() -> None:
    """A probability too small for the dtype to carry is not a misspecification.

    Such a row sums to one and lies in `[0, 1]`, and the outcome it names is
    simply very unlikely. Its contribution to a finite continuation is below the
    last bit, so the solve prices the row as though the outcome were absent
    rather than rejecting a model whose answer is correct.
    """
    active = jnp.zeros(()).dtype
    tiny = jnp.finfo(active).tiny
    subnormal = np.nextafter(np.asarray(tiny), np.asarray(0.0, dtype=active))

    def subnormal_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.asarray([subnormal, 1.0 - subnormal], dtype=active)

    model = _model_with_state_probs(subnormal_probs)
    V = model.solve(log_level="debug", params={"discount_factor": 0.95}).values

    assert not bool(jnp.any(jnp.isnan(jnp.asarray(V[0]["alive"]))))


@categorical(ordered=False)
class _Effort:
    low: ScalarInt
    high: ScalarInt


def _effort(consumption: ContinuousAction) -> ScalarInt:
    # Never `high` on the consumption grid: the validator must still check the
    # law at `high`, because it sweeps every declared code of a derived
    # categorical rather than only the codes the regime functions reach.
    return jnp.where(consumption > 10.0, _Effort.high, _Effort.low)


def _income(wealth: ContinuousState) -> FloatND:
    return 0.1 * wealth


def _model_with_function_reading_health_probs(
    *, next_health_func, phased: bool = False, **model_options: Unpack[_ModelOptions]
) -> Model:
    """Build a model whose `health` law reads regime-function outputs.

    `effort` is a derived categorical (a regime function with a declared
    `DiscreteGrid`); `income` is a plain continuous function output. With
    `phased`, the solve and simulate phases each declare their own copy of the law.
    """
    law = StochasticTransition(func=next_health_func)
    alive = UserRegime(
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(category_class=_Health)},
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": Phased(solve=law, simulate=law) if phased else law,
        },
        functions={"utility": _utility_alive, "effort": _effort, "income": _income},
        derived_categoricals={"effort": DiscreteGrid(category_class=_Effort)},
        constraints={"budget": _budget},
    )
    return Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
        **model_options,
    )


def test_state_validator_checks_law_reading_derived_categorical(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A law reading a derived categorical is numerically validated, not skipped."""

    def health_probs(*, health: DiscreteState, effort: ScalarInt) -> FloatND:
        good = jnp.where(effort == _Effort.high, 0.9, 0.7)
        return jnp.where(
            health == _Health.good,
            jnp.stack([1.0 - good, good]),
            jnp.array([0.5, 0.5]),
        )

    model = _model_with_function_reading_health_probs(next_health_func=health_probs)
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(log_level="warning", params={"discount_factor": 0.95})

    skips = [r for r in caplog.records if "not numerically validated" in r.message]
    assert skips == []


def test_state_validator_raises_on_invalid_law_at_unreached_derived_code() -> None:
    """Invalid probabilities at any declared derived-categorical code raise."""

    def health_probs(
        *,
        health: DiscreteState,  # noqa: ARG001
        effort: ScalarInt,
    ) -> FloatND:
        return jnp.where(
            effort == _Effort.high, jnp.array([0.5, 0.2]), jnp.array([0.3, 0.7])
        )

    model = _model_with_function_reading_health_probs(next_health_func=health_probs)
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError,
        match=r"state 'health' in regime 'alive' .* do not sum to 1",
    ):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


# Each law below reads `health` without depending on it: a state that nothing
# reads is refused as unused, and the row under test reads only `income` or
# `carried_share`.
def _health_from_income(
    *,
    health: DiscreteState,  # noqa: ARG001
    income: FloatND,
) -> FloatND:
    return jnp.stack([1.0 - income, income])


def _health_beside_income(
    *,
    health: DiscreteState,  # noqa: ARG001
    income: FloatND,
) -> FloatND:
    return jnp.stack([jnp.asarray(0.5), income])


def test_state_validator_checks_law_reading_non_categorical_function_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A law reading a plain function output is evaluated through that function.

    `income = 0.1 * wealth` lies in `[0.1, 1]` on the wealth grid, so the row
    `[1 - income, income]` is valid and nothing is skipped.
    """
    model = _model_with_function_reading_health_probs(
        next_health_func=_health_from_income
    )
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(log_level="warning", params={"discount_factor": 0.95})

    skips = [r for r in caplog.records if "not numerically validated" in r.message]
    assert skips == []


def test_state_validator_rejects_invalid_law_reading_function_output() -> None:
    """The row `[0.5, income]` sums to `0.6` at `wealth = 1` and is refused."""
    model = _model_with_function_reading_health_probs(
        next_health_func=_health_beside_income
    )
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError,
        match=r"state 'health' in regime 'alive'",
    ):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _health_from_share(
    *,
    health: DiscreteState,  # noqa: ARG001
    carried_share: ContinuousState,
) -> FloatND:
    return jnp.stack([1.0 - carried_share, carried_share])


def _health_from_stray_share(
    *, health: DiscreteState, carried_share: ContinuousState
) -> FloatND:
    return _health_from_share(health=health, carried_share=2.0 * carried_share - 0.5)


def _model_with_carried_share_health_probs(
    *, simulate_law: Callable[..., FloatND]
) -> Model:
    """Build a model whose `health` laws read the carried state `carried_share`.

    `carried_share` is the function `0.5` in the solve phase and the grid
    `{0, 1}` in the simulate phase. The solve law is `[1 - s, s]`; the
    simulate law is `simulate_law`.
    """
    alive = UserRegime(
        states={
            "wealth": WEALTH_GRID,
            "health": DiscreteGrid(category_class=_Health),
            "carried_share": Phased(
                solve=_half, simulate=LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
            ),
        },
        actions={"consumption": CONSUMPTION_GRID},
        state_transitions={
            "wealth": _next_wealth,
            "health": Phased(
                solve=StochasticTransition(func=_health_from_share),
                simulate=StochasticTransition(func=simulate_law),
            ),
            "carried_share": fixed_transition("carried_share"),
        },
        functions={"utility": _utility_alive},
        constraints={"budget": _budget},
    )
    return Model(
        regimes={"alive": alive, "terminal": _terminal_regime()},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"terminal": 0}},
    )


def test_state_validator_checks_laws_reading_a_carried_state_in_both_phases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The solve law reads the carried state's function, the simulate law its grid."""
    model = _model_with_carried_share_health_probs(simulate_law=_health_from_share)
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.solve(log_level="warning", params={"discount_factor": 0.95})

    skips = [r for r in caplog.records if "not numerically validated" in r.message]
    assert skips == []


def test_state_validator_rejects_a_simulate_law_invalid_on_the_carried_grid() -> None:
    """At `carried_share = 0` the simulate row is `[1.5, -0.5]` and is refused."""
    model = _model_with_carried_share_health_probs(
        simulate_law=_health_from_stray_share
    )
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError,
        match=r"state 'health' \(simulate phase\) in regime 'alive'",
    ):
        model.solve(log_level="debug", params={"discount_factor": 0.95})


def _health_probs_reading_effort(
    *,
    health: DiscreteState,
    effort: ScalarInt,
) -> FloatND:
    good = jnp.where(effort == _Effort.high, 0.9, 0.7)
    return jnp.where(
        health == _Health.good, jnp.stack([1.0 - good, good]), jnp.array([0.5, 0.5])
    )


def _budgeted_model_with_derived_law(*, phased: bool) -> Model:
    return _model_with_function_reading_health_probs(
        next_health_func=_health_probs_reading_effort,
        phased=phased,
        execution_config=ExecutionConfig(device_memory_bytes=2**28),
    )


def _derived_code_footprint(*, model: Model, phased: bool) -> DeviceBufferFootprint:
    """Measure the validation code arrays each law keeps, read off its metadata."""
    codes = tuple(
        code
        for regime in model._regimes.values()
        for transition in regime.stochastic_state_transitions.values()
        for code in transition.derived_categorical_codes.values()
    )
    footprint = measure_buffer_footprint(tree=codes)
    # A plain law keeps one two-code int32 array; a phased law keeps one per phase.
    assert len(codes) == (2 if phased else 1)
    assert sum(
        end - start for spans in footprint.spans.values() for start, end in spans
    ) == 8 * len(codes)
    return footprint


def _uncovered_code_bytes(
    *, codes: DeviceBufferFootprint, inventory: DeviceBufferFootprint
) -> int:
    return sum(
        resident_bytes_by_device(
            live=codes, arguments=inventory, devices=tuple(codes.spans)
        ).values()
    )


_INITIAL_CONDITIONS = {
    "age": np.array([0.0]),
    "wealth": np.array([10.0]),
    "health": np.array([_Health.good], dtype=np.int32),
    "regime_id": np.array([_RegimeId.alive], dtype=np.int32),
}


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_simulate_entry_inventory_charges_derived_categorical_codes(
    *, phased: bool, log_level: LogLevel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every simulation entry snapshot covers the law validation code arrays."""
    model = _budgeted_model_with_derived_law(phased=phased)
    codes = _derived_code_footprint(model=model, phased=phased)
    snapshots: list[DeviceBufferFootprint] = []
    original = SimulationEntryAllocations.snapshot

    def record(self: SimulationEntryAllocations) -> DeviceBufferFootprint:
        snapshot = original(self)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(SimulationEntryAllocations, "snapshot", record)
    model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions=_INITIAL_CONDITIONS,
        seed=0,
        log_level=log_level,
    )

    assert snapshots
    assert [
        _uncovered_code_bytes(codes=codes, inventory=snapshot) for snapshot in snapshots
    ] == [0] * len(snapshots)


@pytest.mark.parametrize("phased", [False, True])
def test_simulation_preflight_charges_derived_categorical_codes(
    *, phased: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validation preflight admits its buffers against the code arrays."""
    model = _budgeted_model_with_derived_law(phased=phased)
    codes = _derived_code_footprint(model=model, phased=phased)
    memories: list[SimulationMemory] = []
    original = initial_conditions_module._preflight_memory

    def record(**arguments: Unpack[_PreflightKwargs]) -> SimulationMemory | None:
        memory = original(**arguments)
        assert memory is not None
        memories.append(memory)
        return memory

    monkeypatch.setattr(initial_conditions_module, "_preflight_memory", record)
    model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions=_INITIAL_CONDITIONS,
        seed=0,
        log_level="debug",
    )

    assert [
        _uncovered_code_bytes(codes=codes, inventory=memory.inputs)
        for memory in memories
    ] == [0]


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_solve_fixed_inventory_charges_derived_categorical_codes(
    *, phased: bool, log_level: LogLevel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A standalone budgeted solve counts the code arrays among its fixed owners."""
    model = _budgeted_model_with_derived_law(phased=phased)
    codes = _derived_code_footprint(model=model, phased=phased)
    trees: list[FootprintTree] = []
    original: Callable[..., MappingProxyType[int, int]] = (
        backward_induction_module.concrete_device_bytes
    )

    def record(*, tree: FootprintTree) -> MappingProxyType[int, int]:
        trees.append(tree)
        return original(tree=tree)

    monkeypatch.setattr(backward_induction_module, "concrete_device_bytes", record)
    model.solve(params={"discount_factor": 0.95}, log_level=log_level)

    assert trees
    assert [
        _uncovered_code_bytes(
            codes=codes, inventory=measure_buffer_footprint(tree=tree)
        )
        for tree in trees
    ] == [0] * len(trees)


class _ModelOptions(TypedDict):
    execution_config: NotRequired[ExecutionConfig]


class _PreflightKwargs(TypedDict, closed=True):
    execution: ResolvedExecution | None
    retained_footprint: DeviceBufferFootprint | None
    producers: NotRequired[ProfiledSimulationOperations | None]
    initial_conditions: InitialConditions
    flat_params: FlatParams
    regimes: MappingProxyType[RegimeName, Regime]
    regime_names_to_ids: RegimeNamesToIds
    ages: TimeAxis


def test_validate_transitions_returns_none_without_backward_induction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public preflight accepts valid laws without entering the solver."""

    def valid_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.array([0.4, 0.6])

    solver = Mock(side_effect=AssertionError("Transition validation entered solver"))
    monkeypatch.setattr(Model, "_solve_from_flat_params", solver)
    model = _model_with_state_probs(valid_probs)
    assert model.validate_transitions(params={"discount_factor": 0.95}) is None
    solver.assert_not_called()


@pytest.mark.parametrize(
    "probabilities", [[0.4, 0.5], [1.1, -0.1], [float("nan"), 1.0]]
)
def test_validate_transitions_matches_solve_failure(probabilities: list[float]) -> None:
    """Preflight and solve preserve the first failing law's diagnostic."""

    def bad_probs(health: DiscreteState) -> FloatND:  # noqa: ARG001
        return jnp.asarray(probabilities)

    model = _model_with_state_probs(bad_probs)
    params = {"discount_factor": 0.95}
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as preflight:
        model.validate_transitions(params=params)
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as solve:
        model.solve(params=params, log_level="off")
    assert str(preflight.value) == str(solve.value)
    assert "state 'health'" in str(preflight.value)
    assert "regime 'alive'" in str(preflight.value)
    assert "age 0" in str(preflight.value)


def test_validate_transitions_checks_later_periods_and_new_params() -> None:
    """A successful call does not hide an invalid later-period parameterization."""

    def health_probs(*, health: DiscreteState, period: int, mass: float) -> FloatND:  # noqa: ARG001
        return jnp.array([0.4, jnp.where(period == 1, mass - 0.4, 0.6)])

    prototype = _model_with_state_probs(health_probs)
    model = Model(
        regimes=prototype.user_regimes,
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": (0, 1), "terminal": 2}},
    )
    assert (
        model.validate_transitions(params={"discount_factor": 0.95, "mass": 1.0})
        is None
    )
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError, match=r"age 1.*sum to 1"
    ):
        model.validate_transitions(params={"discount_factor": 0.95, "mass": 0.9})


@categorical(ordered=False)
class _PhaseDemandRegimeId:
    source: ScalarInt
    perceived: ScalarInt
    realized: ScalarInt


_PHASE_DEMAND_AGES = (0, 1, 2)
_PHASE_DEMAND_EDGES = {
    "solve": {"source": {"perceived": (0, 1)}},
    "simulate": {"source": {"realized": (0,), "perceived": (1,)}},
}
_PHASE_DEMAND_VALID_ROW = (0.4, 0.6)
_PHASE_DEMAND_CARRIED_NODES = (0.0, 1.0)


def _phase_demand_utility(health: DiscreteState) -> FloatND:
    return jnp.asarray(health, dtype=float)


def _phase_demand_half() -> ContinuousState:
    return jnp.asarray(0.5)


def _phase_demand_model(
    *,
    health_law: StateTransitionEntry,
    functions: Mapping[str, FunctionEntry] | None = None,
    carried: bool = False,
) -> tuple[Model, frozenset[tuple[str, int, str]]]:
    """Build a model whose solve and simulate phases demand different targets.

    `source` reaches `perceived` at ages 0 and 1 in the solve phase, and
    `realized` at age 0 and `perceived` at age 1 in the simulate phase. Returns
    the model and its demanded `(phase, period, target)` triples, derived from
    the declared edges and checked against the model's own graphs.
    """
    states: dict[str, StateEntry] = {"health": DiscreteGrid(category_class=_Health)}
    transitions: dict[str, StateTransitionEntry] = {"health": health_law}
    if carried:
        states["carried_share"] = Phased(
            solve=_phase_demand_half,
            simulate=LinSpacedGrid(
                start=_PHASE_DEMAND_CARRIED_NODES[0],
                stop=_PHASE_DEMAND_CARRIED_NODES[-1],
                n_points=len(_PHASE_DEMAND_CARRIED_NODES),
            ),
        )
        transitions["carried_share"] = fixed_transition("carried_share")
    source = UserRegime(
        states=states,
        state_transitions=transitions,
        functions={"utility": _phase_demand_utility, **(functions or {})},
    )
    terminal = UserRegime(
        states={"health": DiscreteGrid(category_class=_Health)},
        functions={"utility": _phase_demand_utility},
    )
    model = Model(
        regimes={"source": source, "perceived": terminal, "realized": terminal},
        ages=AgeGrid(
            start=_PHASE_DEMAND_AGES[0],
            inclusive_stop=_PHASE_DEMAND_AGES[-1],
            step="Y",
        ),
        regime_id_class=_PhaseDemandRegimeId,
        initial_nodes={0: "source", 1: "source"},
        edges=Phased(
            solve=_PHASE_DEMAND_EDGES["solve"],
            simulate=_PHASE_DEMAND_EDGES["simulate"],
        ),
    )
    period_by_age = {age: period for period, age in enumerate(_PHASE_DEMAND_AGES)}
    demanded = frozenset(
        (phase, period_by_age[age], target)
        for phase, edges in _PHASE_DEMAND_EDGES.items()
        for target, ages in edges["source"].items()
        for age in ages
    )
    observed = frozenset(
        (phase, period, target)
        for phase, graph in (
            ("solve", model.graph.solution),
            ("simulate", model.graph.simulation),
        )
        for period in range(len(_PHASE_DEMAND_AGES) - 1)
        for target in graph.targets(period=period, source="source")
    )
    assert observed == demanded
    return model, demanded


def _phase_demand_row_is_valid(row: tuple[float, float]) -> bool:
    """Check a two-outcome row independently of the production validator."""
    array = np.asarray(row, dtype=float)
    return bool(
        np.isfinite(array).all()
        and (array >= 0).all()
        and (array <= 1).all()
        and np.isclose(array.sum(), 1.0, atol=1e-6, rtol=0.0)
    )


def _phase_demand_probabilities(
    *, row: tuple[float, float], bad_period: int
) -> Callable[..., FloatND]:
    def probabilities(
        *,
        health: DiscreteState,  # noqa: ARG001
        period: int,
    ) -> FloatND:
        return jnp.where(
            period == bad_period,
            jnp.asarray(row),
            jnp.asarray(_PHASE_DEMAND_VALID_ROW),
        )

    return probabilities


def _assert_phase_demand_result(
    *,
    model: Model,
    fails: bool,
    period: int,
    phase: str | None = None,
    target: RegimeName | None = None,
) -> None:
    if not fails:
        assert model.validate_transitions(params={"discount_factor": 0.95}) is None
        return
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as caught:
        model.validate_transitions(params={"discount_factor": 0.95})
    label = "state 'health'"
    qualifiers = [
        *([f"target regime '{target}'"] if target is not None else []),
        *([f"{phase} phase"] if phase is not None else []),
    ]
    if qualifiers:
        label += f" ({', '.join(qualifiers)})"
    assert f"{label} in regime 'source' at age {_PHASE_DEMAND_AGES[period]}" in str(
        caught.value
    )


def _phase_demand_from_row(*, probability_row: FloatND) -> FloatND:
    return probability_row


def _phase_demand_from_forwarded_row(*, forwarded_row: FloatND) -> FloatND:
    return forwarded_row


@pytest.mark.parametrize(
    ("law_phase", "bad_period", "bad_target", "row"),
    [
        pytest.param("solve", 0, "perceived", (0.4, 0.5), id="solve-retained"),
        pytest.param("simulate", 0, "realized", (0.4, 0.5), id="simulate-mass"),
        pytest.param("simulate", 0, "realized", (1.1, -0.1), id="simulate-range"),
        pytest.param("simulate", 0, "realized", (float("nan"), 1.0), id="simulate-nan"),
        pytest.param("simulate", 1, "perceived", (0.4, 0.5), id="simulate-later"),
        pytest.param("simulate", 0, "perceived", (0.4, 0.5), id="other-phase-only"),
        pytest.param("simulate", 1, "realized", (0.4, 0.5), id="other-period-only"),
        pytest.param(None, 0, "realized", (0.4, 0.5), id="shared-simulate-only"),
        pytest.param(None, 1, "realized", (0.4, 0.5), id="shared-unused-period"),
        pytest.param(None, 0, "perceived", (0.4, 0.5), id="shared-solve-only"),
        pytest.param("simulate", 0, "realized", (0.4, 0.6), id="phased-valid"),
        pytest.param(None, 0, "realized", (0.4, 0.6), id="shared-valid"),
    ],
)
def test_per_target_state_laws_are_checked_where_their_phase_demands_them(
    *,
    law_phase: str | None,
    bad_period: int,
    bad_target: RegimeName,
    row: tuple[float, float],
) -> None:
    """A per-target law is validated exactly at its phase's demanded edges.

    An explicitly phased law is checked against its own phase's targets at
    each period; a shared law against each phase's targets separately. An
    invalid row on an edge no phase demands is never a failure.
    """

    def law(*, target: RegimeName, phase: str | None) -> StochasticTransition:
        selected = (
            row
            if (target, phase) == (bad_target, law_phase)
            else _PHASE_DEMAND_VALID_ROW
        )
        return StochasticTransition(
            func=_phase_demand_probabilities(row=selected, bad_period=bad_period)
        )

    health_law: StateTransitionEntry
    if law_phase is None:
        health_law = {
            target: law(target=target, phase=None)
            for target in ("perceived", "realized")
        }
    else:
        health_law = Phased(
            solve={"perceived": law(target="perceived", phase="solve")},
            simulate={
                target: law(target=target, phase="simulate")
                for target in ("perceived", "realized")
            },
        )
    model, demanded = _phase_demand_model(health_law=health_law)
    phases = ("solve", "simulate") if law_phase is None else (law_phase,)
    affected = {(phase, bad_period, bad_target) for phase in phases}
    # A shared law failing on a simulate-only edge names the simulate phase.
    failing_phases = {phase for phase, _, _ in demanded & affected}
    _assert_phase_demand_result(
        model=model,
        fails=bool(demanded & affected) and not _phase_demand_row_is_valid(row),
        period=bad_period,
        phase=law_phase
        if law_phase is not None
        else ("simulate" if failing_phases == {"simulate"} else None),
        target=bad_target,
    )


@pytest.mark.parametrize(
    ("per_target", "indirect", "row"),
    [
        pytest.param(False, False, (0.4, 0.5), id="coarse-direct-helper"),
        pytest.param(True, False, (0.4, 0.5), id="target-direct-helper"),
        pytest.param(True, True, (0.4, 0.5), id="target-indirect-helper"),
        pytest.param(True, True, (0.4, 0.6), id="target-indirect-valid"),
    ],
)
def test_shared_state_law_reads_each_phases_own_functions(
    *,
    per_target: bool,
    indirect: bool,
    row: tuple[float, float],
) -> None:
    """A shared law reading a phased function is checked with each phase's variant.

    The simulate variant of `probability_row`, read directly or through another
    function, governs the simulate phase's demanded edge at age 0.
    """
    functions: dict[str, FunctionEntry] = {
        "probability_row": Phased(
            solve=_phase_demand_probabilities(
                row=_PHASE_DEMAND_VALID_ROW, bad_period=0
            ),
            simulate=_phase_demand_probabilities(row=row, bad_period=0),
        )
    }
    if indirect:
        functions["forwarded_row"] = _phase_demand_from_row
    transition = StochasticTransition(
        func=_phase_demand_from_forwarded_row if indirect else _phase_demand_from_row
    )
    health_law = (
        dict.fromkeys(("perceived", "realized"), transition)
        if per_target
        else transition
    )
    model, demanded = _phase_demand_model(health_law=health_law, functions=functions)
    affected = {
        (phase, period, target)
        for phase, period, target in demanded
        if phase == "simulate" and period == 0
    }
    assert affected == {("simulate", 0, "realized")}
    _assert_phase_demand_result(
        model=model,
        fails=not _phase_demand_row_is_valid(row),
        period=0,
        target="realized" if per_target else None,
        phase="simulate",
    )


@pytest.mark.parametrize("invalid", [False, True], ids=["valid-grid", "late-node"])
def test_shared_state_law_sweeps_the_carried_grid_beyond_the_imputation(
    *,
    invalid: bool,
) -> None:
    """A shared law reading a carried state is checked on its whole simulate grid.

    The solve imputation 0.5 and the first carried node 0.0 give valid rows; the
    invalid law fails only at the later carried node 1.0.
    """

    def health_probabilities(*, carried_share: ContinuousState) -> FloatND:
        good = (
            jnp.where(carried_share > 0.75, 1.2, carried_share)
            if invalid
            else carried_share
        )
        return jnp.stack([1.0 - good, good])

    model, _ = _phase_demand_model(
        health_law=StochasticTransition(func=health_probabilities), carried=True
    )
    good_at = {
        node: 1.2 if invalid and node > 0.75 else node
        for node in (0.5, *_PHASE_DEMAND_CARRIED_NODES)
    }
    assert _phase_demand_row_is_valid((1.0 - good_at[0.5], good_at[0.5]))
    assert _phase_demand_row_is_valid((1.0 - good_at[0.0], good_at[0.0]))
    fails = any(
        not _phase_demand_row_is_valid((1.0 - good_at[node], good_at[node]))
        for node in _PHASE_DEMAND_CARRIED_NODES
    )
    _assert_phase_demand_result(model=model, fails=fails, period=0, phase="simulate")


type _AgeGridLawKind = Literal["state", "joint"]
type _AgeGridCase = Literal[
    "invalid_current_nodes", "invalid_representative_nodes", "valid_everywhere"
]


def _age_grid_wealth(age: float) -> LinSpacedGrid:
    """Age 0 -> [1, 2], age 1 -> [11, 12], age 2 -> [21, 22]."""
    return LinSpacedGrid(start=1.0 + 10.0 * age, stop=2.0 + 10.0 * age, n_points=2)


_AGE_GRID_WEALTH = AgeSpecializedGrid(build=_age_grid_wealth, signature=lambda age: age)


def _age_grid_probabilities(case: _AgeGridCase) -> Callable[..., FloatND]:
    def law(*, wealth: ContinuousState, period: int) -> FloatND:
        if case == "invalid_current_nodes":
            bad = (period == 1) & (wealth > 10.0)
        elif case == "invalid_representative_nodes":
            bad = (period == 1) & (wealth < 10.0)
        else:
            bad = jnp.zeros((), dtype=bool)
        return jnp.stack((jnp.asarray(0.4), jnp.where(bad, 0.5, 0.6)))

    return law


def _age_grid_next_wealth(*, wealth: ContinuousState) -> FloatND:
    return wealth + 10.0


def _age_grid_health_from_draw(draw: dict[str, ScalarInt]) -> ScalarInt:
    return draw["health"]


def _age_grid_utility(
    *, wealth: ContinuousState, health: DiscreteState, consumption: ContinuousAction
) -> FloatND:
    return wealth + 0.01 * consumption + jnp.where(health == _Health.good, 0.1, 0.0)


def _age_grid_terminal_utility(
    *, wealth: ContinuousState, health: DiscreteState
) -> FloatND:
    return wealth + jnp.where(health == _Health.good, 0.1, 0.0)


def _age_grid_model(*, law_kind: _AgeGridLawKind, law: Callable[..., FloatND]) -> Model:
    """Build a model whose wealth nodes move with age and land on the next grid."""
    states = {
        "wealth": _AGE_GRID_WEALTH,
        "health": DiscreteGrid(category_class=_Health),
    }
    actions = {"consumption": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)}
    functions = {"utility": _age_grid_utility}
    if law_kind == "state":
        alive = UserRegime(
            states=states,
            actions=actions,
            functions=functions,
            state_transitions={
                "wealth": _age_grid_next_wealth,
                "health": StochasticTransition(func=law),
            },
        )
    else:
        joint = JointTransition(
            support_size=2,
            support={
                "health": jnp.asarray([_Health.bad, _Health.good], dtype=jnp.int32)
            },
            probabilities=law,
            outputs={"health": _age_grid_health_from_draw},
        )
        alive = UserRegime(
            states=states,
            actions=actions,
            functions=functions,
            state_transitions={"wealth": _age_grid_next_wealth},
            joint_transitions={"alive": {"draw": joint}, "terminal": {"draw": joint}},
        )
    terminal = UserRegime(
        states={
            "wealth": _age_grid_wealth(3.0),
            "health": DiscreteGrid(category_class=_Health),
        },
        functions={"utility": _age_grid_terminal_utility},
    )
    return Model(
        regimes={"alive": alive, "terminal": terminal},
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        edges={"alive": {"alive": (0, 1), "terminal": 2}},
    )


def _age_grid_invalid_rows(
    *, law: Callable[..., FloatND], use_representative_grid: bool
) -> list[tuple[int, float]]:
    """Enumerate the factory's nodes and check each row with `math.fsum`."""
    invalid = []
    for period in (0, 1, 2):
        age = 0.0 if use_representative_grid else float(period)
        for node in _AGE_GRID_WEALTH.build(age).to_jax().tolist():
            row = law(wealth=jnp.asarray(node), period=period).tolist()
            valid = all(
                math.isfinite(value) and 0.0 <= value <= 1.0 for value in row
            ) and math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-6)
            if not valid:
                invalid.append((period, float(node)))
    return invalid


@pytest.mark.parametrize("law_kind", ["state", "joint"])
def test_validate_transitions_rejects_law_invalid_only_on_current_age_nodes(
    *,
    law_kind: _AgeGridLawKind,
) -> None:
    """A law invalid only at age 1's own wealth nodes is refused."""
    law = _age_grid_probabilities("invalid_current_nodes")
    assert _age_grid_invalid_rows(law=law, use_representative_grid=False) == [
        (1, 11.0),
        (1, 12.0),
    ]
    assert _age_grid_invalid_rows(law=law, use_representative_grid=True) == []
    with pytest.raises(InvalidStateTransitionProbabilitiesError):
        _age_grid_model(law_kind=law_kind, law=law).validate_transitions(
            params={"discount_factor": 0.95}
        )


@pytest.mark.parametrize("law_kind", ["state", "joint"])
def test_validate_transitions_accepts_law_invalid_only_off_the_current_grid(
    *,
    law_kind: _AgeGridLawKind,
) -> None:
    """A law invalid only at another age's wealth nodes is accepted at age 1."""
    law = _age_grid_probabilities("invalid_representative_nodes")
    assert _age_grid_invalid_rows(law=law, use_representative_grid=False) == []
    assert _age_grid_invalid_rows(law=law, use_representative_grid=True) == [
        (1, 1.0),
        (1, 2.0),
    ]
    model = _age_grid_model(law_kind=law_kind, law=law)
    assert model.validate_transitions(params={"discount_factor": 0.95}) is None


@pytest.mark.parametrize("law_kind", ["state", "joint"])
def test_validate_transitions_accepts_law_valid_on_every_age_grid(
    *,
    law_kind: _AgeGridLawKind,
) -> None:
    """A law with unit mass at every node of every age's grid is accepted."""
    law = _age_grid_probabilities("valid_everywhere")
    assert _age_grid_invalid_rows(law=law, use_representative_grid=False) == []
    model = _age_grid_model(law_kind=law_kind, law=law)
    assert model.validate_transitions(params={"discount_factor": 0.95}) is None


@categorical(ordered=False)
class _LateEntryRegimeId:
    source: ScalarInt
    late: ScalarInt
    terminal: ScalarInt


_LATE_ENTRY_AGES = (0, 1, 2, 3)


def _late_entry_model(*, bad_period: int) -> Model:
    """Build a model whose regime `late` is solved earlier than it is simulated.

    The solve phase reaches `late` from `source` at ages 0 and 1, so `late` is
    solved at ages 1 and 2. The simulate phase reaches `late` only at age 1, so
    simulated subjects occupy it at age 2 alone. The shared health law of `late`
    reads `probability_row`, whose simulate variant is invalid at `bad_period`.
    """
    health = DiscreteGrid(category_class=_Health)
    valid = StochasticTransition(
        func=_phase_demand_probabilities(row=_PHASE_DEMAND_VALID_ROW, bad_period=0)
    )
    source = UserRegime(
        states={"health": health},
        state_transitions={"health": valid},
        functions={"utility": _phase_demand_utility},
    )
    late = UserRegime(
        states={"health": health},
        state_transitions={"health": StochasticTransition(func=_phase_demand_from_row)},
        functions={
            "utility": _phase_demand_utility,
            "probability_row": Phased(
                solve=_phase_demand_probabilities(
                    row=_PHASE_DEMAND_VALID_ROW, bad_period=bad_period
                ),
                simulate=_phase_demand_probabilities(
                    row=(0.4, 0.5), bad_period=bad_period
                ),
            ),
        },
    )
    terminal = UserRegime(
        states={"health": health}, functions={"utility": _phase_demand_utility}
    )
    model = Model(
        regimes={"source": source, "late": late, "terminal": terminal},
        ages=AgeGrid(
            start=_LATE_ENTRY_AGES[0], inclusive_stop=_LATE_ENTRY_AGES[-1], step="Y"
        ),
        regime_id_class=_LateEntryRegimeId,
        initial_nodes={0: "source", 1: "source"},
        edges=Phased(
            solve={"source": {"late": (0, 1)}, "late": {"terminal": (1, 2)}},
            simulate={
                "source": {"terminal": (0,), "late": (1,)},
                "late": {"terminal": (2,)},
            },
        ),
    )
    assert {
        period
        for period in range(len(_LATE_ENTRY_AGES))
        if "late" in model.graph.simulation.active_regimes_by_period[period]
    } == {2}
    return model


def test_shared_state_law_is_not_simulate_checked_where_no_subject_is() -> None:
    """A simulate variant invalid only where no simulated subject is passes.

    `late` is solved at age 1, but the simulation first reaches it at age 2,
    so the simulate evaluation of its shared law has no demand at age 1.
    """
    assert (
        _late_entry_model(bad_period=1).validate_transitions(
            params={"discount_factor": 0.95}
        )
        is None
    )


def test_shared_state_law_simulate_failure_names_the_phase() -> None:
    """A shared law failing only in its simulate evaluation names that phase."""
    with pytest.raises(
        InvalidStateTransitionProbabilitiesError,
        match=r"state 'health' \(simulate phase\) in regime 'late' at age 2",
    ):
        _late_entry_model(bad_period=2).validate_transitions(
            params={"discount_factor": 0.95}
        )


def test_validate_transitions_reports_an_unvalidated_law(
    *, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A law the preflight cannot evaluate is reported, not silently accepted."""
    through_functions = transition_checks_module._through_functions

    def hide_bias(
        *,
        transition: _StochasticStateTransition,
        regime_params: FlatRegimeParams,
        functions: EconFunctionsMapping,
        function_params: FlatRegimeParams,
        inputs: frozenset[StateOrActionName],
    ) -> tuple[_StochasticStateTransition, FlatRegimeParams]:
        transition, regime_params = through_functions(
            transition=transition,
            regime_params=regime_params,
            functions=functions,
            function_params=function_params,
            inputs=inputs,
        )
        return transition, MappingProxyType(
            {
                name: value
                for name, value in regime_params.items()
                if name != "transition_bias"
            }
        )

    monkeypatch.setattr(transition_checks_module, "_through_functions", hide_bias)
    model = _model_with_fixed_param_health_probs()
    with caplog.at_level(logging.WARNING, logger="lcm"):
        model.validate_transitions(params={"discount_factor": 0.95})
    skips = [r for r in caplog.records if "not numerically validated" in r.message]
    assert len(skips) == 1


def test_validate_transitions_leaves_the_shared_logger_level() -> None:
    """The preflight does not change the level later `lcm` warnings pass through."""
    shared = logging.getLogger("lcm")
    shared.setLevel(logging.WARNING)
    _model_with_state_probs(_good_health_probs).validate_transitions(
        params={"discount_factor": 0.95}
    )
    assert shared.level == logging.WARNING
