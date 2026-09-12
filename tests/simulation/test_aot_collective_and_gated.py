"""Runtime-compiled simulation of collective and gated-edge models.

Simulation prepares programs from the population and live argument shapes on each
call. Collective continuation values retain their trailing stakeholder axis;
gated continuation values select one stakeholder's leg, and routing re-evaluates
the gate at the realized candidate target state. Repeated calls reuse the
compiled decisions and population gate evaluators for those shapes.

Both models below are small enough that every simulated value is an exact
arithmetic expression, stated in the factory's docstring.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.simulation import gated_routing
from _lcm.simulation.runtime import CompiledSimulationProgram
from benchmarks.asv._compile_counters import count_compile_requests
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


def test_collective_model_simulates_under_the_runtime_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collective runtime program keeps both stakeholders' values.

    Each subject's period-0 row carries both stakeholders' values at the
    household argmax, and they are the same numbers the model solves to.
    """
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    model, params = make_two_stakeholder_model()
    initial_conditions = make_couple_initial_conditions(n_subjects=_N_SUBJECTS)

    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="debug",
    )
    assert compiled, "Simulation must dispatch an actual compiled program."

    simulated = result.to_dataframe()
    period_0 = simulated.loc[simulated["period"] == 0, ["value_f", "value_m"]]
    aaae(
        period_0.to_numpy(),
        np.asarray(TWO_STAKEHOLDER_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )


def test_gated_edge_model_simulates_under_the_runtime_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gated-edge runtime program keeps its values and consent routes.

    The source regime's own action is chosen against the consent-gated
    continuation, so its period-0 value is the flow payoff plus the discounted
    gated continuation — the married value where consent holds and the single
    fallback where it does not — and each subject then routes to the regime
    its own gate selects.
    """
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    model = _make_consent_model()
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
    assert compiled, "Simulation must dispatch an actual compiled program."

    simulated = result.to_dataframe()
    aaae(
        simulated.loc[simulated["period"] == 0, "value"].to_numpy(),
        np.asarray(CONSENT_V_SINGLE_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )
    routed = simulated.loc[simulated["period"] == 1, "regime_name"]
    assert tuple(routed) == CONSENT_PERIOD_1_REGIMES


def test_gate_evaluators_reuse_compilation_for_a_repeated_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second forward call reuses the gate programs exercised by the first."""
    model = _make_consent_model()
    params = {"discount_factor": _DISCOUNT_FACTOR}
    initial_conditions = {
        "education": jnp.array([Education.low, Education.high], dtype=jnp.int32),
        "age": jnp.zeros(_N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, ConsentRegimeId.single, dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="debug")
    calls: list[Callable] = []
    original = gated_routing.population_call

    def observe(*, func: Callable, axis_size: int) -> Callable:
        call = original(func=func, axis_size=axis_size)
        calls.append(call)
        return call

    monkeypatch.setattr(gated_routing, "population_call", observe)
    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="debug",
    )
    first_calls = tuple(calls)
    assert first_calls, "The positive control must exercise the gate evaluators."
    calls.clear()
    with count_compile_requests() as counts:
        result = model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="debug",
        )
        for periods in result.raw_results.values():
            for period in periods.values():
                jax.block_until_ready(period.V_arr)
    assert tuple(calls) == first_calls
    assert (
        counts.trace_requests,
        counts.lowering_requests,
        counts.compile_requests,
    ) == (
        0,
        0,
        0,
    )
    simulated = result.to_dataframe()
    assert tuple(simulated.loc[simulated["period"] == 1, "regime_name"]) == (
        CONSENT_PERIOD_1_REGIMES
    )


def _capture_compiled_dispatches(
    *, monkeypatch: pytest.MonkeyPatch
) -> list[jax.stages.Compiled]:
    """Observe actual runtime dispatch, refusing a silent eager fallback."""
    observed: list[jax.stages.Compiled] = []
    original = CompiledSimulationProgram.__call__

    def observe(self: CompiledSimulationProgram, **arguments: object) -> object:
        assert isinstance(self.executable, jax.stages.Compiled)
        observed.append(self.executable)
        return original(self, **arguments)

    monkeypatch.setattr(CompiledSimulationProgram, "__call__", observe)
    return observed


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


def _make_consent_model() -> Model:
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
