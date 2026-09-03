"""Ahead-of-time compiled simulation of collective and gated-edge models.

`Model(n_subjects=N)` compiles every simulate function for batch shape `N`
before the first forward pass and swaps the compiled programs in whenever the
simulated population is exactly `N`. A compiled program is fixed to the
argument shapes it was lowered against, so the templates it is lowered with
have to be the arrays simulate actually dispatches:

- a collective regime's value function carries a trailing stakeholder axis, so
  every continuation template built for a collective target carries it too;
- a regime declaring `gated_edges` chooses its action against the gated
  continuation `Wbar`, which is one stakeholder's leg of the target's value
  and therefore carries no stakeholder axis at all;
- routing that regime's rows re-evaluates the gate at the realized candidate
  target state, and that evaluator is compiled ahead of time too, so the
  first routed period dispatches a program rather than tracing one.

Both models below are small enough that every simulated value is an exact
arithmetic expression, stated in the factory's docstring.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.simulation.gated_routing import population_call
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentTransition,
    categorical,
    fixed_transition,
)
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.collective_fixtures import (
    TWO_STAKEHOLDER_V_PERIOD_0,
    make_couple_initial_conditions,
    make_two_stakeholder_model,
)
from tests.conftest import DECIMAL_PRECISION

_N_SUBJECTS = 2

_DISCOUNT_FACTOR = 0.95

# `single`'s period-0 value at education low and high, in subject order.
CONSENT_V_SINGLE_PERIOD_0 = (3.85, 8.65)

# The regime each subject occupies at period 1: the low-education subject
# marries, the high-education one keeps her outside option.
CONSENT_PERIOD_1_REGIMES = ("married_terminal", "single_terminal")


def test_collective_model_simulates_under_the_aot_program():
    """A collective model simulated at its declared `n_subjects` keeps its values.

    Each subject's period-0 row carries both stakeholders' values at the
    household argmax, and they are the same numbers the model solves to.
    """
    model, params = make_two_stakeholder_model(n_subjects=_N_SUBJECTS)
    initial_conditions = make_couple_initial_conditions(n_subjects=_N_SUBJECTS)

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="debug",
    )
    _fail_if_aot_programs_missing(model=model, n_subjects=_N_SUBJECTS)

    simulated = result.to_dataframe()
    period_0 = simulated.loc[simulated["period"] == 0, ["value_f", "value_m"]]
    aaae(
        period_0.to_numpy(),
        np.asarray(TWO_STAKEHOLDER_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )


def test_gated_edge_model_simulates_under_the_aot_program():
    """A gated-edge model simulated at its declared `n_subjects` keeps its values.

    The source regime's own action is chosen against the consent-gated
    continuation, so its period-0 value is the flow payoff plus the discounted
    gated continuation — the married value where consent holds and the single
    fallback where it does not — and each subject then routes to the regime
    its own gate selects.
    """
    model = _make_consent_model(n_subjects=_N_SUBJECTS)
    params = {"discount_factor": _DISCOUNT_FACTOR}
    initial_conditions = {
        "education": jnp.array([Education.low, Education.high], dtype=jnp.int32),
        "age": jnp.zeros(_N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, ConsentRegimeId.single, dtype=jnp.int32),
    }

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="debug",
    )
    _fail_if_aot_programs_missing(model=model, n_subjects=_N_SUBJECTS)

    simulated = result.to_dataframe()
    aaae(
        simulated.loc[simulated["period"] == 0, "value"].to_numpy(),
        np.asarray(CONSENT_V_SINGLE_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )
    routed = simulated.loc[simulated["period"] == 1, "regime_name"]
    assert tuple(routed) == CONSENT_PERIOD_1_REGIMES


def test_gate_evaluators_are_compiled_before_the_first_simulated_period():
    """A gated model's gate evaluators are AOT programs, not traced on first use.

    The router re-evaluates the gate at the realized candidate target state
    once per edge per period, and that evaluation is a compiled program of its
    own. Leaving it out of ahead-of-time compilation would move a trace and a
    compile into the first routed period — the one place a fixed batch size is
    meant to have paid for already.
    """
    model = _make_consent_model(n_subjects=_N_SUBJECTS)

    model.simulate(
        params={"discount_factor": _DISCOUNT_FACTOR},
        initial_conditions={
            "education": jnp.array([Education.low, Education.high], dtype=jnp.int32),
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.full(_N_SUBJECTS, ConsentRegimeId.single, dtype=jnp.int32),
        },
        log_level="debug",
    )

    assert _uncompiled_gate_evaluators(model=model, n_subjects=_N_SUBJECTS) == []


def _fail_if_aot_programs_missing(*, model: Model, n_subjects: int) -> None:
    """Raise unless every dispatched decision function is an AOT program.

    Both tests are claims about the compiled programs, so a model that quietly
    fell back to the interpreted path would make them assert nothing.
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


def _uncompiled_gate_evaluators(*, model: Model, n_subjects: int) -> list[str]:
    """Return a label per gate evaluator that is not an AOT program."""
    cached = model._simulate_compile_cache[n_subjects]
    return [
        f"{regime_name} -> {target_name} (fold period {fold_period})"
        for regime_name, regime in cached.items()
        for target_name, edge in regime.gated_edges.items()
        for fold_period in (
            period + 1
            for period in regime.active_periods
            if period + 1 < model.n_periods
        )
        if not isinstance(
            population_call(
                func=edge.simulate_gate_evaluator_at(period=fold_period),
                axis_size=n_subjects,
            ),
            jax.stages.Compiled,
        )
    ]


@categorical(ordered=True)
class Education:
    """The single state of the consent model, and its own wage level."""

    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


@categorical(ordered=True)
class Work:
    """The binary action of the consent model's source and target regimes."""

    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class ConsentRegimeId:
    """Regime ids of the consent model."""

    single: ScalarInt  # code 0
    single_terminal: ScalarInt  # code 1
    married_terminal: ScalarInt  # code 2


def _make_consent_model(*, n_subjects: int | None) -> Model:
    """Build a singleton source consenting into a collective target.

    `single` is active at age 0 and transitions with probability one into the
    two-stakeholder `married_terminal`, but only where the wife consents:
    her own value in the marriage has to beat the value of staying single.
    Where it does not, the gated edge routes her to `single_terminal` with her
    education carried across.

    Hand computation, wage $\\{1, 2\\}$ by education and $\\beta = 0.95$:

    - `married_terminal`, both payoffs increasing in `work`, so work wins:
      $V^f = (3, 6)$ and $V^m = (1, 2)$ by education.
    - `single_terminal`: $V = (2, 7)$ by education.
    - Consent, $V^f > V^{single}$: $3 > 2$ holds at low education, $6 > 7$
      fails at high, so $\\bar{W}^f = (3, 7)$.
    - `single` at period 0, work again optimal: $V = (1 + 0.95 \\cdot 3,
      2 + 0.95 \\cdot 7) = (3.85, 8.65)$.

    Args:
        n_subjects: Simulate batch size to compile ahead of time, or `None` to
            compile at runtime.

    Returns:
        The model.

    """
    single = Regime(
        transition={
            "married_terminal": ValueDependentTransition(
                probability=MarkovTransition(_certain_transition),
                gate=_consent_gate,
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_terminal",
                            projection={"education": _identity_education},
                        ),
                    )
                },
                gate_references={
                    "V_single_ref": ProjectedRegimeValue(
                        regime="single_terminal",
                        projection={"education": _identity_education},
                    )
                },
            )
        },
        active=lambda age: age < 1,
        states={"education": DiscreteGrid(category_class=Education)},
        state_transitions={"education": fixed_transition("education")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _single_utility},
    )
    single_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(category_class=Education)},
        functions={"utility": _single_terminal_utility},
    )
    married_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(category_class=Education)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _married_utility_f, "m": _married_utility_m}
            )
        },
    )
    return Model(
        regimes={
            "single": single,
            "single_terminal": single_terminal,
            "married_terminal": married_terminal,
        },
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=ConsentRegimeId,
        n_subjects=n_subjects,
    )


def _wage(education: DiscreteState) -> FloatND:
    """Wage by education: 1 for the low level, 2 for the high one."""
    return jnp.where(education == Education.low, 1.0, 2.0)


def _certain_transition(age: FloatND) -> FloatND:
    """Regime transition probability: the marriage edge is offered every period."""
    return jnp.ones_like(age, dtype=float)


def _single_utility(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """A single woman earns her wage when she works and nothing otherwise."""
    return _wage(education) * work


def _single_terminal_utility(education: DiscreteState) -> FloatND:
    """The outside option, steep in education: 2 at the low level, 7 at the high."""
    return 2.0 + 5.0 * education


def _married_utility_f(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The wife's share of household income: three times her own wage."""
    return 3.0 * _wage(education) * work


def _married_utility_m(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The husband's share of household income: his wife's wage."""
    return _wage(education) * work


def _identity_education(education: DiscreteState) -> DiscreteState:
    """Education is carried unchanged into the fallback regime."""
    return education


def _consent_gate(*, V_target_f: FloatND, V_single_ref: FloatND) -> BoolND:
    """The wife marries only when the marriage beats staying single."""
    return V_target_f > V_single_ref
