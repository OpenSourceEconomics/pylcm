"""Regime-transition mass in a collective continuation.

A regime's continuation is a lottery over the targets it can reach, weighted by
the regime-transition probabilities. Two properties hold whether or not the
regime carries stakeholders:

- The single target's unit mass prices each stakeholder as its singleton twin.
- Regime selection refuses a non-unit mass and a negative probability at the
  `off`, `warning` and `debug` log levels, including probabilities that sum to one
  only because one is negative.

Every model here pairs a two-stakeholder regime with singleton twins carrying one
stakeholder's utility each. The husband's payoff is twice the wife's, so the
household argmax of the equally weighted scalarization and each twin's own argmax
select the same action, and each stakeholder slice of the collective value must
equal its twin's value.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from _lcm.utils.logging import LogLevel
from lcm import (
    ByAge,
    CollectiveUtility,
    DiscreteGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import (
    ContinuousState,
    DiscreteAction,
    FloatND,
    FunctionName,
    RegimeName,
    UserFunction,
    UserParams,
)
from tests.collective_fixtures import (
    AGES,
    DISCOUNT_FACTOR,
    WAGE_GRID,
    CoupleRegimeId,
    Work,
)
from tests.conftest import DECIMAL_PRECISION

# The stakeholders every collective regime in this module carries, wife first.
STAKEHOLDERS = ("f", "m")

# A regime-transition probability just above one, which regime selection refuses.
INFLATED_MASS = 1.0 + 1e-4

# Probabilities of the two-target source regime's targets at age 0.
# They sum to one, so only the sign disqualifies them as a distribution.
STAY_PROBABILITY = 1.5
LEAVE_PROBABILITY = -0.5

# Period-0 value of each singleton twin of the single-target model, by wage node.
# Working is optimal at both nodes, so the wife's value is
# `wage + 0.95 * 100 * 40` and the husband's is twice that.
EXPECTED_TWIN_V = {"f": (3808.0, 3840.0), "m": (7616.0, 7680.0)}


def test_collective_continuation_prices_each_stakeholder_as_its_singleton_twin():
    """Every stakeholder slice of a collective value equals its singleton twin's.

    The single outgoing edge carries the regime's whole unit mass, so each twin's
    value is the hand-computed `EXPECTED_TWIN_V`.
    """
    rtol = 10.0**-DECIMAL_PRECISION

    collective = _build_single_target_model(household=STAKEHOLDERS)
    collective_V = collective.solve(
        params=_single_target_params(),
        log_level="off",
    ).values[0]["couple"]

    twin_values = []
    for stakeholder in STAKEHOLDERS:
        twin = _build_single_target_model(household=None, stakeholder=stakeholder)
        twin_values.append(
            twin.solve(
                params=_single_target_params(),
                log_level="off",
            ).values[0]["couple"]
        )
    expected_V = jnp.stack(twin_values, axis=-1)

    # Certify the reference against the hand-computed values, so a defect in the
    # twins cannot hide one in the collective path.
    assert_allclose(
        expected_V,
        jnp.asarray([EXPECTED_TWIN_V["f"], EXPECTED_TWIN_V["m"]]).T,
        rtol=rtol,
    )
    # A collective value function carries one axis per solve state plus the
    # trailing stakeholder axis, so the comparison is aligned.
    assert collective_V.shape == (WAGE_GRID.n_points, len(STAKEHOLDERS))

    assert_allclose(collective_V, expected_V, rtol=rtol)


_OUTSIDE_UNIT_INTERVAL = (
    r"^Regime transition probabilities from 'couple' between ages 0 and 1 "
    r"contain values outside \[0, 1\]\. "
)
_NOT_SUMMING_TO_ONE = (
    r"^Regime transition probabilities from 'couple' between ages 0 and 1 "
    r"do not sum to 1\.0\. 1 of 1 probability vectors do not sum to 1\.0\."
)


@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
@pytest.mark.parametrize(
    ("stay_probability", "leave_probability", "match"),
    [
        (INFLATED_MASS, 0.0, _OUTSIDE_UNIT_INTERVAL),
        (0.5, 0.5 - 1e-4, _NOT_SUMMING_TO_ONE),
    ],
)
def test_collective_regime_selection_refuses_a_non_unit_mass(
    *,
    stay_probability: float,
    leave_probability: float,
    match: str,
    log_level: LogLevel,
) -> None:
    """A regime-transition mass other than one is refused at the source's age 0."""
    collective = _build_two_target_model(household=STAKEHOLDERS)
    params = _two_target_params(
        stay_probability=stay_probability, leave_probability=leave_probability
    )
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError, match=match):
        collective.solve(params=params, log_level=log_level)


@pytest.mark.parametrize("log_level", ["off", "warning", "debug"])
@pytest.mark.parametrize("household", [STAKEHOLDERS, None])
def test_collective_regime_selection_refuses_a_negative_probability(
    *, household: tuple[str, ...] | None, log_level: LogLevel
) -> None:
    """Probabilities 1.5 and -0.5 sum to one but are refused at the source's age 0.

    The collective regime and its singleton twin refuse them alike.
    """
    model = _build_two_target_model(household=household)
    with pytest.raises(
        InvalidRegimeTransitionProbabilitiesError, match=_OUTSIDE_UNIT_INTERVAL
    ):
        model.solve(params=_two_target_params(), log_level=log_level)


def test_two_target_model_with_a_distribution_has_a_finite_last_source_period():
    """With probabilities 0.5 and 0.5 the source's last period solves to finite values.

    At age 1 the source reaches its terminal target with probability one, so the
    refusal above is about the age-0 probabilities alone.
    """
    params = _two_target_params(stay_probability=0.5, leave_probability=0.5)
    values = (
        _build_two_target_model(household=STAKEHOLDERS)
        .solve(params=params, log_level="off")
        .values
    )
    assert bool(jnp.isfinite(values[1]["couple"]).all())


def _build_single_target_model(
    *, household: tuple[str, ...] | None, stakeholder: str = "f"
) -> Model:
    """Build a source regime whose only target carries the whole mass.

    `couple` reaches `couple_terminal` — solved from age 1 — along its single
    outgoing edge at age 0, so the edge is the regime's entire transition mass.

    Args:
        household: Stakeholder names of both regimes, or `None` for the
            singleton twin.
        stakeholder: Whose utility the singleton twin carries. Ignored when
            `household` is not `None`.

    Returns:
        The model, which `_single_target_params` supplies the numbers for.

    """
    return _build_model(
        law=None,
        household=household,
        stakeholder=stakeholder,
        source_ends_at_age=1,
    )


def _build_two_target_model(
    *, household: tuple[str, ...] | None, stakeholder: str = "f"
) -> Model:
    """Build a source regime reaching two targets, one of them itself.

    `couple` has a transition law at ages 0 and 1 and `couple_terminal` is solved
    from age 1 on, so at age 0 both are reachable and the transition splits its
    mass between them, while at age 1 only the terminal regime is left and takes
    all of it.

    Args:
        household: Stakeholder names of both regimes, or `None` for the
            singleton twin.
        stakeholder: Whose utility the singleton twin carries. Ignored when
            `household` is not `None`.

    Returns:
        The model, which `_two_target_params` supplies the numbers for.

    """
    return _build_model(
        law={
            "couple": StochasticTransition(func=_stay_probability),
            "couple_terminal": StochasticTransition(func=_leave_probability),
        },
        household=household,
        stakeholder=stakeholder,
        source_ends_at_age=2,
    )


def _build_model(
    *,
    law: Mapping[RegimeName, StochasticTransition] | None,
    household: tuple[str, ...] | None,
    stakeholder: str,
    source_ends_at_age: int,
) -> Model:
    """Build the collective model, or the singleton twin of one stakeholder.

    Both regimes carry the same two-point wage grid and the same binary action;
    only the flow payoffs and the regime transition differ between the models
    this builds.

    Args:
        law: Regime transition law of the source regime, as a per-target dict
            whose targets are reachable before its last age, or `None` when
            the terminal regime is its only target.
        household: Stakeholder names of both regimes, or `None` for a
            singleton twin.
        stakeholder: Whose utility a singleton twin carries.
        source_ends_at_age: First age at which the source regime is not solved.

    Returns:
        The model, ready to solve once its params are supplied.

    """
    couple = Regime(
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions=_source_functions(household=household, stakeholder=stakeholder),
    )
    couple_terminal = Regime(
        states={"wage": WAGE_GRID},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions=_terminal_functions(household=household, stakeholder=stakeholder),
    )
    # Every target is reachable before the last source age; only the terminal
    # regime is reachable at it, where it takes the whole mass.
    targets = {
        **dict.fromkeys(law or {}, tuple(range(source_ends_at_age - 1))),
        "couple_terminal": tuple(range(source_ends_at_age)),
    }
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
        initial_nodes={0: "couple"},
        edges={
            "couple": (
                targets
                if law is None
                else Transition(
                    targets=targets,
                    law=ByAge.until(
                        stop_age_exclusive=source_ends_at_age,
                        law=law,
                        then="couple_terminal",
                    ),
                )
            )
        },
    )


def _source_functions(
    *, household: tuple[str, ...] | None, stakeholder: str
) -> Mapping[FunctionName, UserFunction | CollectiveUtility]:
    """Return the source regime's flow payoffs, as a household or as one agent."""
    per_stakeholder = {"f": _source_utility_f, "m": _source_utility_m}
    if household is None:
        return {"utility": per_stakeholder[stakeholder]}
    return {
        "utility": CollectiveUtility(
            utilities={name: per_stakeholder[name] for name in household}
        )
    }


def _terminal_functions(
    *, household: tuple[str, ...] | None, stakeholder: str
) -> Mapping[FunctionName, UserFunction | CollectiveUtility]:
    """Return the terminal regime's payoffs, as a household or as one agent."""
    per_stakeholder = {"f": _terminal_utility_f, "m": _terminal_utility_m}
    if household is None:
        return {"utility": per_stakeholder[stakeholder]}
    return {
        "utility": CollectiveUtility(
            utilities={name: per_stakeholder[name] for name in household}
        )
    }


def _single_target_params() -> UserParams:
    """Return the single-target model's params."""
    return {
        "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "couple_terminal": {},
    }


def _two_target_params(
    *,
    stay_probability: float = STAY_PROBABILITY,
    leave_probability: float = LEAVE_PROBABILITY,
) -> UserParams:
    """Return the two-target model's params, with the split probabilities."""
    return {
        "couple": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "couple_terminal": {},
        "edges": {
            "couple": {
                "couple": {"stay_probability": stay_probability},
                "couple_terminal": {"leave_probability": leave_probability},
            }
        },
    }


def _source_utility_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife's flow payoff in the source regime: her wage while she works."""
    return wage * work


def _source_utility_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband's flow payoff in the source regime: twice the wife's."""
    return 2.0 * wage * work


def _terminal_utility_f(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife's terminal payoff: a hundred times her wage while she works.

    Large next to the source regime's own flow payoff, so the source value is
    dominated by its continuation and a continuation scaled by the transition
    mass moves that value by the mass's whole deviation from one.
    """
    return 100.0 * wage * work


def _terminal_utility_m(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband's terminal payoff: twice the wife's."""
    return 200.0 * wage * work


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Deterministic wage law: working today yields the high wage tomorrow."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _stay_probability(*, age: float, stay_probability: float) -> FloatND:
    """Probability of staying collective, zero once the source regime ends."""
    return jnp.where(age < 1, stay_probability, 0.0)


def _leave_probability(*, age: float, leave_probability: float) -> FloatND:
    """Probability of entering the terminal regime, certain once the source ends."""
    return jnp.where(age < 1, leave_probability, 1.0)
