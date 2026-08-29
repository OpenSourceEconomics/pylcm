"""Ahead-of-time compiled simulation of a regime reading same-period values.

A collective regime declaring `same_period_refs` chooses its action against a
value-aware feasibility mask whose operands are another regime's value function
*in the same period*. Simulate dispatches those arrays — and each reference
regime's own flat params — alongside the continuation, so a program compiled
ahead of time for `Model(n_subjects=N)` has to be lowered with them too.

The model below is exact arithmetic. Wages are $1$ at low education and $2$ at
high; the wife earns three times her wage when she works, the husband values
his leisure at $5$ and earns his wife's education level when she works. The
terminal couple pays nothing, so every period-0 value is the flow payoff:

| education | leisure $(u^f, u^m)$ | work $(u^f, u^m)$ | household mean |
| --------- | -------------------- | ----------------- | -------------- |
| low       | $(0, 5)$             | $(3, 0)$          | $2.5$ vs $1.5$ |
| high      | $(0, 5)$             | $(6, 1)$          | $2.5$ vs $3.5$ |

Unconstrained, the low-education household would take leisure. The wife's
participation constraint $Q^f \\ge V^{single}$ with $V^{single} = (1, 2)$ rules
leisure out at both education levels, so both households work and the
period-0 values are $(3, 0)$ and $(6, 1)$.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    ValueDependentConstraint,
    categorical,
    fixed_transition,
)
from lcm.regime import ProjectedRegimeValue, Regime
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_N_SUBJECTS = 2

_DISCOUNT_FACTOR = 0.95

# The couple's period-0 `(value_f, value_m)` at low and high education.
COUPLE_V_PERIOD_0 = ((3.0, 0.0), (6.0, 1.0))


@pytest.mark.parametrize("declared_n_subjects", [None, _N_SUBJECTS])
def test_same_period_ref_model_simulates_to_its_participation_constrained_values(
    declared_n_subjects: int | None,
):
    """A collective regime reading same-period values keeps them when AOT-compiled.

    Each household works because the wife's participation constraint rules
    leisure out, and both stakeholders' period-0 values are the ones the model
    solves to — whether the decision runs interpreted (`Model(n_subjects=None)`)
    or as a program compiled ahead of time for the simulated population.
    """
    model = _make_participation_model(n_subjects=declared_n_subjects)
    params = {
        "couple": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "couple_terminal": {},
        "single_f": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "single_f_terminal": {},
    }
    initial_conditions = {
        "education": jnp.array([Education.low, Education.high], dtype=jnp.int32),
        "age": jnp.zeros(_N_SUBJECTS),
        "regime_id": jnp.full(
            _N_SUBJECTS, ParticipationRegimeId.couple, dtype=jnp.int32
        ),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    if declared_n_subjects is not None:
        _fail_if_aot_programs_missing(model=model, n_subjects=declared_n_subjects)

    simulated = result.to_dataframe()
    period_0 = simulated.loc[simulated["period"] == 0, ["value_f", "value_m"]]
    aaae(
        period_0.to_numpy(),
        np.asarray(COUPLE_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )


def _fail_if_aot_programs_missing(*, model: Model, n_subjects: int) -> None:
    """Raise unless every dispatched decision function is an AOT program.

    The test is a claim about the compiled programs, so a model that quietly
    fell back to the interpreted path would make it assert nothing.
    """
    cached = model._simulate_compile_cache.get(n_subjects)
    if cached is None:
        msg = (
            f"no simulate program was compiled for {n_subjects} subjects; "
            f"compiled batch shapes: {sorted(model._simulate_compile_cache)}"
        )
        raise AssertionError(msg)
    interpreted = [
        f"{regime_name}/argmax_and_max_Q_over_a[{period}]"
        for regime_name, regime in cached.items()
        for period in regime.active_periods
        if not isinstance(
            regime.simulation.argmax_and_max_Q_over_a[period], jax.stages.Compiled
        )
    ]
    if interpreted:
        msg = f"these decision functions were not AOT-compiled: {interpreted}"
        raise AssertionError(msg)


@categorical(ordered=True)
class Education:
    """The single state of the participation model, and its own wage level."""

    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


@categorical(ordered=True)
class Work:
    """The binary action of the participation model's regimes."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class ParticipationRegimeId:
    """Regime ids of the participation model."""

    couple: ScalarInt  # code 0
    couple_terminal: ScalarInt  # code 1
    single_f: ScalarInt  # code 2
    single_f_terminal: ScalarInt  # code 3


def _make_participation_model(*, n_subjects: int | None) -> Model:
    """Build a collective regime whose feasibility reads a single's value.

    `couple` and `single_f` are both active at age 0, so `single_f`'s value is
    available in the same period the couple chooses its action — the reference
    the wife's participation constraint compares her own $Q^f$ against.

    Args:
        n_subjects: Simulate batch size to compile ahead of time, or `None` to
            compile at runtime.

    Returns:
        The model.

    """
    couple = Regime(
        transition={"couple_terminal": MarkovTransition(_certain_transition)},
        active=lambda age: age < 1,
        states={"education": DiscreteGrid(Education)},
        state_transitions={"education": fixed_transition("education")},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _couple_utility_f, "m": _couple_utility_m}
            )
        },
        constraints={
            "participation_f": ValueDependentConstraint(
                predicate=_participation_f,
                references={
                    "V_single_f_ref": ProjectedRegimeValue(
                        regime="single_f", projection={"education": _identity_education}
                    )
                },
            )
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(Education)},
        actions={"work": DiscreteGrid(Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _zero_utility, "m": _zero_utility}
            )
        },
    )
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_certain_transition)},
        active=lambda age: age < 1,
        states={"education": DiscreteGrid(Education)},
        state_transitions={"education": fixed_transition("education")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _single_f_utility},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(Education)},
        functions={"utility": _zero_terminal_utility},
    )
    return Model(
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal,
            "single_f": single_f,
            "single_f_terminal": single_f_terminal,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=ParticipationRegimeId,
        n_subjects=n_subjects,
    )


def _wage(education: DiscreteState) -> FloatND:
    """Wage by education: 1 for the low level, 2 for the high one."""
    return jnp.where(education == Education.low, 1.0, 2.0)


def _certain_transition(age: FloatND) -> FloatND:
    """Regime transition probability: the successor regime is reached for sure."""
    return jnp.ones_like(age, dtype=float)


def _couple_utility_f(education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The wife's share of household income: three times her own wage."""
    return 3.0 * _wage(education) * work


def _couple_utility_m(education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The husband values his leisure at 5, and her education when she works."""
    return 5.0 * (1.0 - work) + education * work


def _zero_utility(education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The terminal couple pays nothing, whatever it does."""
    return 0.0 * education * work


def _single_f_utility(education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The outside option: 1 at low education, 2 at high, and only when working."""
    return (1.0 + education) * work


def _zero_terminal_utility(education: DiscreteState) -> FloatND:
    """The terminal single pays nothing."""
    return 0.0 * education


def _identity_education(education: DiscreteState) -> DiscreteState:
    """Education is carried unchanged into the reference regime."""
    return education


def _participation_f(Q_f: FloatND, V_single_f_ref: FloatND) -> BoolND:
    """The wife accepts only an action worth at least her outside option."""
    return Q_f >= V_single_f_ref
