"""Mass normalization and probability poisoning in a collective continuation.

A regime's continuation is a lottery over the targets it can reach, weighted by
the regime-transition probabilities. Two properties of that lottery hold whether
or not the regime carries stakeholders:

- The aggregated continuation is divided by the probability mass that was summed,
  so a mass the arithmetic accepts as a distribution but that is not exactly one
  does not scale the continuation.
- Probabilities that sum to one only because one of them is negative are not a
  distribution, and the continuation publishes `NaN` rather than a finite value.

Every model here pairs a two-stakeholder regime with singleton twins carrying one
stakeholder's utility each. The husband's payoff is twice the wife's, so the
household argmax of the equally weighted scalarization and each twin's own argmax
select the same action, and each stakeholder slice of the collective value must
equal its twin's value.
"""

from collections.abc import Mapping

import jax.numpy as jnp
from numpy.testing import assert_allclose

from lcm import CollectiveUtility, DiscreteGrid, MarkovTransition, Model, Regime
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

# Regime-transition mass of the single-target source regime. Inside the band the
# continuation arithmetic accepts as a distribution, and far enough above one
# that an undivided sum shows up at either float precision.
INFLATED_MASS = 1.0 + 1e-4

# Probabilities of the two-target source regime's targets while it is active.
# They sum to one, so only the sign disqualifies them as a distribution.
STAY_PROBABILITY = 1.5
LEAVE_PROBABILITY = -0.5

# Period-0 value of each singleton twin of the single-target model, by wage node.
# Working is optimal at both nodes, so the wife's value is
# `wage + 0.95 * 100 * 40` and the husband's is twice that.
EXPECTED_TWIN_V = {"f": (3808.0, 3840.0), "m": (7616.0, 7680.0)}


def test_collective_continuation_divides_by_represented_mass():
    """A transition mass inside the accepted band is renormalized.

    The continuation arithmetic accepts any regime-transition mass within a
    thousandth of one, and divides the aggregated continuation by the mass it
    summed. A mass of `1 + 1e-4` therefore prices a collective regime exactly as
    a mass of one does, and every stakeholder slice of its value equals the value
    of the singleton regime carrying that stakeholder's utility.
    """
    rtol = 10.0**-DECIMAL_PRECISION

    collective = _build_single_target_model(household=STAKEHOLDERS)
    # Runtime validation rejects a probability above one. The assertion below is
    # about the continuation arithmetic, which carries its own, wider guard on
    # the mass it sums.
    collective_V = collective.solve(
        params=_single_target_params(mass=INFLATED_MASS),
        log_level="off",
    )[0]["couple"]

    twin_values = []
    for stakeholder in STAKEHOLDERS:
        twin = _build_single_target_model(household=None, stakeholder=stakeholder)
        twin_values.append(
            twin.solve(
                params=_single_target_params(mass=INFLATED_MASS),
                log_level="off",
            )[0]["couple"]
        )
    expected_V = jnp.stack(twin_values, axis=-1)

    # Certify the reference against the hand-computed unit-mass values, so a
    # defect in the twins cannot hide one in the collective path.
    assert_allclose(
        expected_V,
        jnp.asarray([EXPECTED_TWIN_V["f"], EXPECTED_TWIN_V["m"]]).T,
        rtol=rtol,
    )
    # A collective value function carries one axis per solve state plus the
    # trailing stakeholder axis, so the comparison is aligned.
    assert collective_V.shape == (WAGE_GRID.n_points, len(STAKEHOLDERS))

    assert_allclose(collective_V, expected_V, rtol=rtol)


def test_collective_continuation_poisons_a_negative_probability():
    """A negative regime-transition probability publishes `NaN`.

    Unit mass alone does not make a set of transition probabilities a
    distribution: 1.5 and -0.5 sum to one. A collective regime whose two targets
    carry those probabilities publishes `NaN` at every state and for every
    stakeholder, the same answer a singleton regime publishes there.
    """
    collective = _build_two_target_model(household=STAKEHOLDERS)
    # Runtime validation rejects a probability outside `[0, 1]`. The assertion
    # below is about the continuation arithmetic, which reads each probability's
    # sign off its own bits.
    solution = collective.solve(
        params=_two_target_params(),
        log_level="off",
    )

    twin = _build_two_target_model(household=None, stakeholder="f")
    twin_solution = twin.solve(params=_two_target_params(), log_level="off")
    # The singleton contract the collective regime states as well.
    assert bool(jnp.isnan(twin_solution[0]["couple"]).all())
    # The source regime's last period reaches its terminal target with
    # probability one, so a `NaN` there would mean an ill-posed model rather than
    # the negative probability under test.
    assert bool(jnp.isfinite(solution[1]["couple"]).all())

    assert bool(jnp.isnan(solution[0]["couple"]).all())


def _build_single_target_model(
    *, household: tuple[str, ...] | None, stakeholder: str = "f"
) -> Model:
    """Build a source regime whose only target carries the whole declared mass.

    `couple` is active at age 0 and reaches `couple_terminal` — active from age 1
    — with the probability named by the `regime_mass` parameter, which is the
    regime's entire transition mass.

    Args:
        household: Stakeholder names of both regimes, or `None` for the
            singleton twin.
        stakeholder: Whose utility the singleton twin carries. Ignored when
            `household` is not `None`.

    Returns:
        The model, which `_single_target_params` supplies the numbers for.

    """
    return _build_model(
        transition={"couple_terminal": MarkovTransition(_target_probability)},
        household=household,
        stakeholder=stakeholder,
        source_ends_at_age=1,
    )


def _build_two_target_model(
    *, household: tuple[str, ...] | None, stakeholder: str = "f"
) -> Model:
    """Build a source regime reaching two targets, one of them itself.

    `couple` is active at ages 0 and 1 and `couple_terminal` from age 1 on, so at
    age 0 both are reachable and the transition splits its mass between them,
    while at age 1 only the terminal regime is left and takes all of it.

    Args:
        household: Stakeholder names of both regimes, or `None` for the
            singleton twin.
        stakeholder: Whose utility the singleton twin carries. Ignored when
            `household` is not `None`.

    Returns:
        The model, which `_two_target_params` supplies the numbers for.

    """
    return _build_model(
        transition={
            "couple": MarkovTransition(_stay_probability),
            "couple_terminal": MarkovTransition(_leave_probability),
        },
        household=household,
        stakeholder=stakeholder,
        source_ends_at_age=2,
    )


def _build_model(
    *,
    transition: Mapping[RegimeName, MarkovTransition],
    household: tuple[str, ...] | None,
    stakeholder: str,
    source_ends_at_age: int,
) -> Model:
    """Build the collective model, or the singleton twin of one stakeholder.

    Both regimes carry the same two-point wage grid and the same binary action;
    only the flow payoffs and the regime transition differ between the models
    this builds.

    Args:
        transition: Regime transition of the source regime, as a per-target dict.
        household: Stakeholder names of both regimes, or `None` for a
            singleton twin.
        stakeholder: Whose utility a singleton twin carries.
        source_ends_at_age: First age at which the source regime is inactive.

    Returns:
        The model, ready to solve once its params are supplied.

    """
    couple = Regime(
        transition=transition,
        active=lambda age: age < source_ends_at_age,
        states={"wage": WAGE_GRID},
        state_transitions={"wage": _next_wage},
        actions={"work": DiscreteGrid(Work)},
        functions=_source_functions(household=household, stakeholder=stakeholder),
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage": WAGE_GRID},
        actions={"work": DiscreteGrid(Work)},
        functions=_terminal_functions(household=household, stakeholder=stakeholder),
    )
    return Model(
        regimes={"couple": couple, "couple_terminal": couple_terminal},
        ages=AGES,
        regime_id_class=CoupleRegimeId,
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


def _single_target_params(*, mass: float) -> UserParams:
    """Return the single-target model's params, with `mass` as its whole mass."""
    return {
        "couple": {
            "koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR},
            "couple_terminal": {"next_regime": {"regime_mass": mass}},
        },
        "couple_terminal": {},
    }


def _two_target_params() -> UserParams:
    """Return the two-target model's params, with the split probabilities."""
    return {
        "couple": {
            "koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR},
            "couple": {"next_regime": {"stay_probability": STAY_PROBABILITY}},
            "couple_terminal": {
                "next_regime": {"leave_probability": LEAVE_PROBABILITY}
            },
        },
        "couple_terminal": {},
    }


def _source_utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife's flow payoff in the source regime: her wage while she works."""
    return wage * work


def _source_utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband's flow payoff in the source regime: twice the wife's."""
    return 2.0 * wage * work


def _terminal_utility_f(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Wife's terminal payoff: a hundred times her wage while she works.

    Large next to the source regime's own flow payoff, so the source value is
    dominated by its continuation and a continuation scaled by the transition
    mass moves that value by the mass's whole deviation from one.
    """
    return 100.0 * wage * work


def _terminal_utility_m(wage: ContinuousState, work: DiscreteAction) -> FloatND:
    """Husband's terminal payoff: twice the wife's."""
    return 200.0 * wage * work


def _next_wage(work: DiscreteAction) -> ContinuousState:
    """Deterministic wage law: working today yields the high wage tomorrow."""
    return 40.0 * work + 8.0 * (1.0 - work)


def _target_probability(regime_mass: float) -> FloatND:
    """Probability of the only target: the source regime's whole mass."""
    return jnp.asarray(regime_mass)


def _stay_probability(age: float, stay_probability: float) -> FloatND:
    """Probability of staying collective, zero once the source regime ends."""
    return jnp.where(age < 1, stay_probability, 0.0)


def _leave_probability(age: float, leave_probability: float) -> FloatND:
    """Probability of entering the terminal regime, certain once the source ends."""
    return jnp.where(age < 1, leave_probability, 1.0)
