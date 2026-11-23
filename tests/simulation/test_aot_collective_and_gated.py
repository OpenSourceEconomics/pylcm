"""Runtime-compiled simulation of collective and gated-edge models.

Simulation prepares programs from the population and live argument shapes on each
call. Collective continuation values retain their trailing member axis; a
consented exit moves a subject within the period into the collective target.
Repeated calls reuse the compiled decisions and exit programs for those shapes.

Both models below are small enough that every simulated value is an exact
arithmetic expression, stated in the factory's docstring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.simulation.runtime import CompiledSimulationProgram
from _lcm.typing import PytreeValue, ShapeDtypePytree
from benchmarks.asv._compile_counters import count_compile_requests
from lcm import (
    AgeGrid,
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exits import Exit, ExitTarget
from lcm.typing import DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.collective_fixtures import (
    TWO_MEMBER_V_PERIOD_0,
    make_couple_initial_conditions,
    make_two_member_model,
)
from tests.conftest import DECIMAL_PRECISION

_N_SUBJECTS = 2

_DISCOUNT_FACTOR = 0.95

# Each subject's own period-0 value at education low and high, in subject order:
# her married value where she moves, her single value where she stays.
CONSENT_OWN_VALUE_PERIOD_0 = (5.85, 13.4)

# The regime each subject acts in at period 0: the low-education subject moves
# into the couple, the high-education one stays single.
CONSENT_PERIOD_0_REGIMES = ("married", "single")

# The regime each subject occupies at period 1: the low-education subject
# stays married, the high-education one keeps her outside option.
CONSENT_PERIOD_1_REGIMES = ("married_terminal", "single_terminal")


def test_collective_model_simulates_under_the_runtime_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A collective runtime program keeps both members' values.

    Each subject's period-0 row carries both members' values at the
    household argmax, and they are the same numbers the model solves to.
    """
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    model, params = make_two_member_model()
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
        np.asarray(TWO_MEMBER_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )


def test_consent_exit_model_simulates_under_the_runtime_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consented-exit runtime program keeps its values and routes.

    Each subject's own period-0 value is her married value where both consent
    and her single value where she does not, and each subject then acts in the
    regime the consent selects.
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
    period_0 = simulated.loc[simulated["period"] == 0]
    assert tuple(period_0["regime_name"]) == CONSENT_PERIOD_0_REGIMES
    aaae(
        np.where(
            period_0["regime_name"] == "married",
            period_0["value_f"],
            period_0["value"],
        ),
        np.asarray(CONSENT_OWN_VALUE_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )
    routed = simulated.loc[simulated["period"] == 1, "regime_name"]
    assert tuple(routed) == CONSENT_PERIOD_1_REGIMES


def test_exit_programs_reuse_compilation_for_a_repeated_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second forward call reuses the exit programs the first one built.

    The consented exit is evaluated inside the compiled simulation program,
    so the first call must compile and the repeated one must dispatch the same
    executables without tracing, lowering or compiling anything.
    """
    model = _make_consent_model()
    params = {"discount_factor": _DISCOUNT_FACTOR}
    initial_conditions = {
        "education": jnp.array([Education.low, Education.high], dtype=jnp.int32),
        "age": jnp.zeros(_N_SUBJECTS),
        "regime_id": jnp.full(_N_SUBJECTS, ConsentRegimeId.single, dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="debug")
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    with count_compile_requests() as first:
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="debug",
        )
    assert compiled, "The first population must dispatch a compiled program."
    assert first.compile_requests > 0, "The positive control must compile."
    compiled.clear()
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
    assert compiled, "The repeated population must dispatch a compiled program."
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

    def observe(
        self: CompiledSimulationProgram, **arguments: PytreeValue | ShapeDtypePytree
    ) -> PytreeValue:
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
    married: ScalarInt  # code 2
    married_terminal: ScalarInt  # code 3
    partner: ScalarInt  # code 4


def _make_consent_model() -> Model:
    """Build a singleton source whose consented exit enters a collective target.

    `single` may move at age 0 into the two-member `married` as its member `f`,
    whose counterparty `m` compares with the value of `partner`. Both consent
    only where each one's married value strictly beats the own outside value.

    Hand computation, wage $\\{1, 2\\}$ by education and $\beta = 0.95$:

    - `married` then `married_terminal`, both payoffs increasing in `work`, so work
      wins: $V^f = 3 w (1 + 0.95) = (5.85, 11.7)$ and $V^m = (1.95, 3.9)$.
    - `single` without its exit: $V = w + 0.95 (2 + 10 e) = (2.9, 13.4)$.
    - `partner`: $V = -10 + 0.95 (2 + 10 e) = (-8.1, 1.4)$, so he always consents.
    - She consents where $5.85 > 2.9$, at low education, and not where
      $11.7 < 13.4$, at high: the low subject moves, the high one stays.

    Returns:
        The model.

    """
    single = Regime(
        states={"education": DiscreteGrid(category_class=Education)},
        state_transitions={"education": fixed_transition("education")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _single_utility},
        exit=Exit(
            targets={
                "self": ExitTarget(
                    regime="married",
                    role="f",
                    projection={"education": _identity_education},
                )
            },
            rule=Consent(
                counterparty=Counterparty(
                    regime="partner",
                    role="m",
                    projection={"education": _identity_education},
                )
            ),
        ),
    )
    single_terminal = Regime(
        states={"education": DiscreteGrid(category_class=Education)},
        functions={"utility": _single_terminal_utility},
    )
    married_terminal = Regime(
        states={"education": DiscreteGrid(category_class=Education)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _married_utility_f, "m": _married_utility_m}
            )
        },
    )
    married = married_terminal.replace(
        state_transitions={"education": fixed_transition("education")}
    )
    partner = Regime(
        states={"education": DiscreteGrid(category_class=Education)},
        state_transitions={"education": fixed_transition("education")},
        functions={"utility": _partner_utility},
    )
    return Model(
        regimes={
            "single": single,
            "single_terminal": single_terminal,
            "married": married,
            "married_terminal": married_terminal,
            "partner": partner,
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=ConsentRegimeId,
        initial_nodes={0: ("single", "partner")},
        edges={
            "single": {"single_terminal": 0},
            "married": {"married_terminal": 0},
            "partner": {"single_terminal": 0},
        },
    )


def _wage(education: DiscreteState) -> FloatND:
    """Wage by education: 1 for the low level, 2 for the high one."""
    return jnp.where(education == Education.low, 1.0, 2.0)


def _single_utility(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """A single woman earns her wage when she works and nothing otherwise."""
    return _wage(education) * work


def _single_terminal_utility(education: DiscreteState) -> FloatND:
    """The outside option, steep in education: 2 at the low level, 12 at the high."""
    return 2.0 + 10.0 * education


def _partner_utility(education: DiscreteState) -> FloatND:
    """The counterparty's single flow, low enough that he always consents."""
    return -10.0 + 0.0 * education


def _married_utility_f(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The wife's share of household income: three times her own wage."""
    return 3.0 * _wage(education) * work


def _married_utility_m(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The husband's share of household income: his wife's wage."""
    return _wage(education) * work


def _identity_education(education: DiscreteState) -> DiscreteState:
    """Education is carried unchanged into the regime the exit reads or enters."""
    return education
