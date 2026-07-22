"""Regression test for F5: E2 value-aware feasibility bypassed the guard.

`_get_deterministic_transitions` merges each `next_<state>` law across a
regime's target bundles and returns the set of names whose implementation
DIFFERS across targets (`conflicting_deterministic_transition_names`, see
`Q_and_F.py`). `_fail_if_conflicting_transition_is_read` rejects a model whose
within-period decision reads one of those names, because the merged decision
DAG would silently bind one target's law while the simulate state-update uses
the per-target one
(`test_conflicting_target_specific_deterministic_law_read_by_utility_is_rejected`
in `test_Q_and_F.py` pins this for ordinary `utility`/`feasibility`).

F5 reported that the guard was applied to ordinary same-period-ref reads but
NOT threaded into the E2 value-aware feasibility machinery: neither the
`value_constraints` predicate evaluators nor the `same_period_refs` projection
functions built in `_build_value_constraint_machinery` /
`_build_same_period_ref_reader` received
`conflicting_deterministic_transition_names`, so both could read a
target-dependent conflicting `next_<state>` law and silently bind one
target's implementation.

A standalone unit-level check (bypassing full model construction) confirmed
this at HEAD before the fix: `_build_value_constraint_machinery` with a
conflicting `deterministic_transitions` mapping built and evaluated a
predicate reading the conflicting name WITHOUT error, while the equivalent
`_get_U_and_F` call (ordinary utility/feasibility path) raised `ValueError`
for the identical input. The fix threads
`conflicting_deterministic_transition_names` into both builders and applies
`_fail_if_conflicting_transition_is_read` to each value-constraint predicate
and each same-period-ref projection, mirroring the existing ordinary-path
call site.

This file proves it end-to-end through real model construction
(`process_regimes`): a collective regime with two transition targets whose
`next_wage` law differs, read either by a `value_constraints` predicate or by
a `same_period_refs` projection, must be rejected exactly like the ordinary
path.
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from lcm import DiscreteGrid, LinSpacedGrid, categorical, fixed_transition
from lcm.ages import AgeGrid
from lcm.regime import Regime, SamePeriodRef
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


_WAGE_GRID = LinSpacedGrid(start=1.0, stop=3.0, n_points=3)


def _prob_half(age: FloatND) -> FloatND:
    return 0.5 * jnp.ones_like(age, dtype=float)


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.5 * wage * work


def _utility_zero_collective(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return 0.0 * wage * work


def _next_wage_identity(wage: ContinuousState) -> ContinuousState:
    return wage


def _next_wage_zero(wage: ContinuousState) -> ContinuousState:
    return 0.0 * wage


def _vc_ignores_next_wage(Q_f: FloatND) -> BoolND:
    """A value constraint that never reads the conflicting name (control leg)."""
    return Q_f >= -jnp.inf


def _make_conflict_couple(*, value_constraint, same_period_refs=None) -> Regime:
    return Regime(
        transition={
            "terminal_a": MarkovTransition(_prob_half),
            "terminal_b": MarkovTransition(_prob_half),
        },
        active=lambda age: age < 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        state_transitions={
            "wage": {
                "terminal_a": _next_wage_identity,
                "terminal_b": _next_wage_zero,
            },
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
        value_constraints={"vc": value_constraint},
        same_period_refs=same_period_refs or {},
    )


def _terminal_collective() -> Regime:
    return Regime(
        transition=None,
        active=lambda age: age >= 1,
        stakeholders=("f", "m"),
        states={"wage": _WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility_f": _utility_zero_collective,
            "utility_m": _utility_zero_collective,
        },
    )


def _process(regimes: dict[str, Regime]) -> None:
    ages = AgeGrid(start=0, stop=1, step="Y")
    process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {name: jnp.int32(i) for i, name in enumerate(regimes)}
        ),
        enable_jit=False,
    )


# --------------------------------------------------------------------------------------
# value_constraints predicate reading a conflicting `next_<state>` law
# --------------------------------------------------------------------------------------


def _vc_reads_next_wage(next_wage: ContinuousState) -> BoolND:
    return next_wage >= -jnp.inf


def test_value_constraint_reading_conflicting_transition_is_rejected():
    """F5: an E2 predicate reading a target-dependent conflicting law is rejected.

    `couple` transitions to `terminal_a` (next_wage = wage) or `terminal_b`
    (next_wage = 0), a genuine target-dependent `next_wage`. The value
    constraint reads `next_wage` directly, exactly the read pattern the
    ordinary-path guard already rejects for `utility`/`feasibility`.
    """
    regimes = {
        "couple": _make_conflict_couple(value_constraint=_vc_reads_next_wage),
        "terminal_a": _terminal_collective(),
        "terminal_b": _terminal_collective(),
    }
    with pytest.raises(ValueError, match="next_wage"):
        _process(regimes)


def test_value_constraint_reading_non_conflicting_transition_is_accepted():
    """Negative control: identical `next_wage` laws across targets build fine.

    Same shape as the rejected model, but both targets share the SAME
    `next_wage` function object, so `_get_deterministic_transitions` reports
    no conflict and the build must succeed.
    """
    couple = _make_conflict_couple(value_constraint=_vc_reads_next_wage).replace(
        state_transitions={
            "wage": {
                "terminal_a": _next_wage_identity,
                "terminal_b": _next_wage_identity,
            },
        },
    )
    regimes = {
        "couple": couple,
        "terminal_a": _terminal_collective(),
        "terminal_b": _terminal_collective(),
    }
    _process(regimes)  # must not raise


def test_value_constraint_not_reading_conflicting_transition_is_accepted():
    """A conflicting law that's never read by the predicate is harmless."""
    regimes = {
        "couple": _make_conflict_couple(value_constraint=_vc_ignores_next_wage),
        "terminal_a": _terminal_collective(),
        "terminal_b": _terminal_collective(),
    }
    _process(regimes)  # must not raise


# --------------------------------------------------------------------------------------
# same_period_refs projection reading a conflicting `next_<state>` law
# --------------------------------------------------------------------------------------


def _utility_single(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _utility_single_terminal(wage: ContinuousState) -> FloatND:
    return 0.0 * wage


def _project_reads_next_wage(next_wage: ContinuousState) -> ContinuousState:
    """A same-period-ref projection reading the SOURCE regime's `next_wage`."""
    return next_wage


def _single_regime() -> Regime:
    return Regime(
        transition={"single_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wage": _WAGE_GRID},
        state_transitions={"wage": fixed_transition("wage")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_single},
    )


def _single_terminal_regime() -> Regime:
    return Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": _WAGE_GRID},
        functions={"utility": _utility_single_terminal},
    )


def test_same_period_ref_projection_reading_conflicting_transition_is_rejected():
    """F5: an E2 projection reading a target-dependent conflicting law is rejected.

    The value-constraint predicate itself (`_vc_ignores_next_wage`) never
    touches `next_wage`; ONLY the `same_period_refs` projection does. Before
    the fix, `_build_same_period_ref_reader` never received
    `conflicting_deterministic_transition_names`, so this leg alone isolates
    the projection-path bypass from the predicate-path one covered above.
    """
    couple = _make_conflict_couple(
        value_constraint=_vc_ignores_next_wage,
        same_period_refs={
            "V_ref": SamePeriodRef(
                regime="single", projection={"wage": _project_reads_next_wage}
            )
        },
    )
    regimes = {
        "couple": couple,
        "terminal_a": _terminal_collective(),
        "terminal_b": _terminal_collective(),
        "single": _single_regime(),
        "single_terminal": _single_terminal_regime(),
    }
    with pytest.raises(ValueError, match="next_wage"):
        _process(regimes)


def test_same_period_ref_projection_not_reading_conflicting_transition_is_accepted():
    """Negative control: an ordinary (non-conflicting) projection builds fine."""

    def _project_wage(wage: ContinuousState) -> ContinuousState:
        return wage

    couple = _make_conflict_couple(
        value_constraint=_vc_ignores_next_wage,
        same_period_refs={
            "V_ref": SamePeriodRef(regime="single", projection={"wage": _project_wage})
        },
    )
    regimes = {
        "couple": couple,
        "terminal_a": _terminal_collective(),
        "terminal_b": _terminal_collective(),
        "single": _single_regime(),
        "single_terminal": _single_terminal_regime(),
    }
    _process(regimes)  # must not raise
