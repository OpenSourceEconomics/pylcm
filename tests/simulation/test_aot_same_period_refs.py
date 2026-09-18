"""Runtime-compiled simulation of a regime reading same-period values.

A collective regime declaring `same_period_refs` chooses its action against a
value-aware feasibility mask whose operands are another regime's value function
*in the same period*. Simulate dispatches those arrays — and each reference
regime's own flat params — alongside the continuation, so a program compiled
at runtime for the current population has to be lowered with them too.

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

from collections.abc import Mapping
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.simulation.simulate import simulate as simulate_with_replay_readers
from _lcm.utils.logging import get_logger
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
from lcm.solver_api import ActionOutput, ValueStore
from lcm.transition import MarkovTransition
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION
from tests.simulation.test_aot_collective_and_gated import _capture_compiled_dispatches

_N_SUBJECTS = 2

_DISCOUNT_FACTOR = 0.95

# The couple's period-0 `(value_f, value_m)` at low and high education.
COUPLE_V_PERIOD_0 = ((3.0, 0.0), (6.0, 1.0))


class _AlwaysWorkReplayReader:
    """JAX-transformable external reader selecting the constrained action."""

    def __call__(
        self,
        *,
        states: Mapping[str, object],
        fallback_actions: Mapping[str, object],  # noqa: ARG002
    ) -> ActionOutput:
        """Return one categorical work code per simulated subject."""
        education = jnp.asarray(states["education"])
        return ActionOutput(
            actions={
                "work": jnp.full(education.shape, Work.work, dtype=jnp.int32),
            }
        )


@pytest.mark.parametrize("n_subjects", [2, 4])
def test_same_period_ref_model_simulates_to_its_participation_constrained_values(
    *, n_subjects: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime programs preserve participation-constrained values at each population.

    Each household works because the wife's participation constraint rules
    leisure out, and both stakeholders' period-0 values are the ones the model
    solves to, including when the runtime population repeats both education levels.
    """
    compiled = _capture_compiled_dispatches(monkeypatch=monkeypatch)
    model = _make_participation_model()
    params = {
        "couple": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "couple_terminal": {},
        "single_f": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "single_f_terminal": {},
    }
    initial_conditions = {
        "education": jnp.tile(
            jnp.array([Education.low, Education.high], dtype=jnp.int32),
            n_subjects // 2,
        ),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.full(
            n_subjects, ParticipationRegimeId.couple, dtype=jnp.int32
        ),
    }

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
        np.tile(np.asarray(COUPLE_V_PERIOD_0), (n_subjects // 2, 1)),
        decimal=DECIMAL_PRECISION,
    )


def test_external_replay_scores_actions_with_same_period_reference_inputs() -> None:
    """External canonical-Q replay receives the reference arrays and their params."""
    model = _make_participation_model()
    params = {
        "couple": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "couple_terminal": {},
        "single_f": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "single_f_terminal": {},
    }
    initial_conditions = MappingProxyType(
        {
            "education": jnp.array([Education.low, Education.high], dtype=jnp.int32),
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.full(
                _N_SUBJECTS, ParticipationRegimeId.couple, dtype=jnp.int32
            ),
        }
    )
    solution = model.solve(params=params, log_level="off")
    assert isinstance(solution.values, ValueStore)

    result = simulate_with_replay_readers(
        flat_params=model._process_params(params),
        initial_conditions=initial_conditions,
        regimes=model._runtime_regimes_for_shape(compile_batch_size=_N_SUBJECTS),
        regime_names_to_ids=model.regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution.values.materialize(),
        ages=model.ages,
        simulation_output_dtypes=model.simulation_output_dtypes,
        period_to_regime_to_replay_reader=MappingProxyType(
            {0: MappingProxyType({"couple": _AlwaysWorkReplayReader()})}
        ),
        seed=0,
    )

    period_0 = result.raw_results["couple"][0]
    np.testing.assert_array_equal(period_0.actions["work"], [Work.work, Work.work])
    aaae(
        np.asarray(period_0.V_arr),
        np.asarray(COUPLE_V_PERIOD_0),
        decimal=DECIMAL_PRECISION,
    )


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


def _make_participation_model() -> Model:
    """Build a collective regime whose feasibility reads a single's value.

    `couple` and `single_f` are both active at age 0, so `single_f`'s value is
    available in the same period the couple chooses its action — the reference
    the wife's participation constraint compares her own $Q^f$ against.

    Returns:
        The model.

    """
    couple = Regime(
        transition={"couple_terminal": MarkovTransition(_certain_transition)},
        active=lambda age: age < 1,
        states={"education": DiscreteGrid(category_class=Education)},
        state_transitions={"education": fixed_transition("education")},
        actions={"work": DiscreteGrid(category_class=Work)},
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
        states={"education": DiscreteGrid(category_class=Education)},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _zero_utility, "m": _zero_utility}
            )
        },
    )
    single_f = Regime(
        transition={"single_f_terminal": MarkovTransition(_certain_transition)},
        active=lambda age: age < 1,
        states={"education": DiscreteGrid(category_class=Education)},
        state_transitions={"education": fixed_transition("education")},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _single_f_utility},
    )
    single_f_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"education": DiscreteGrid(category_class=Education)},
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
    )


def _wage(education: DiscreteState) -> FloatND:
    """Wage by education: 1 for the low level, 2 for the high one."""
    return jnp.where(education == Education.low, 1.0, 2.0)


def _certain_transition(age: FloatND) -> FloatND:
    """Regime transition probability: the successor regime is reached for sure."""
    return jnp.ones_like(age, dtype=float)


def _couple_utility_f(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The wife's share of household income: three times her own wage."""
    return 3.0 * _wage(education) * work


def _couple_utility_m(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The husband values his leisure at 5, and her education when she works."""
    return 5.0 * (1.0 - work) + education * work


def _zero_utility(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The terminal couple pays nothing, whatever it does."""
    return 0.0 * education * work


def _single_f_utility(*, education: DiscreteState, work: DiscreteAction) -> FloatND:
    """The outside option: 1 at low education, 2 at high, and only when working."""
    return (1.0 + education) * work


def _zero_terminal_utility(education: DiscreteState) -> FloatND:
    """The terminal single pays nothing."""
    return 0.0 * education


def _identity_education(education: DiscreteState) -> DiscreteState:
    """Education is carried unchanged into the reference regime."""
    return education


def _participation_f(*, Q_f: FloatND, V_single_f_ref: FloatND) -> BoolND:
    """The wife accepts only an action worth at least her outside option."""
    return Q_f >= V_single_f_ref
