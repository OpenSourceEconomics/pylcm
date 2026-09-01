"""A marriage market: consent to marry, participate, or dissolve.

Three things a single decision maker never does happen here, and each is one
declaration:

- a household of two takes ONE consumption decision, and each partner's own
  value at that common choice is published — `CollectiveUtility`;
- the household is feasible only where each partner is at least as well off in
  it as alone in the same period — `ValueDependentConstraint`, reading the
  singles' current-period value functions;
- a single marries only under mutual consent, and a couple whose feasible set
  is empty dissolves back to two singles — `ValueDependentTransition`, whose
  gate compares values on the target regime's grid.

The dissolution edge is keyed by the CONTINUING collective regime under
`gate = ~D_couple`: the gate-open branch is staying married, and each partner's
route fallback is her or his own single regime. Keying it by one partner's
single regime would send both partners there whenever the couple stays
together.

Everything is CRRA over consumption with a fixed marriage premium, so the model
is small, has no free structure, and scales along its grid knobs —
`n_periods`, `wealth_n_points` and `consumption_n_points` — which is what makes
it usable as a benchmark workload as well as a worked example.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    CollectiveUtility,
    LinSpacedGrid,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    ValueDependentConstraint,
    ValueDependentTransition,
    categorical,
)
from lcm.transition import MarkovTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

# Lowest wealth node, kept strictly positive so log utility stays finite.
WEALTH_MIN = 1.0

# Highest wealth node.
WEALTH_MAX = 20.0

# Chance a single meets a partner in a period, before consent is decided.
MEETING_PROBABILITY = 0.4


@categorical(ordered=False)
class RegimeId:
    """The six regimes a subject of this model can occupy."""

    couple: ScalarInt
    couple_terminal: ScalarInt
    single_f: ScalarInt
    single_m: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt


def get_model(
    *,
    n_periods: int = 5,
    wealth_n_points: int = 20,
    consumption_n_points: int = 20,
    n_subjects: int | None = None,
) -> Model:
    """Build the marriage-market model.

    Args:
        n_periods: Number of lifecycle periods. The last one is terminal.
        wealth_n_points: Nodes on every regime's wealth grid.
        consumption_n_points: Nodes on every regime's consumption grid.
        n_subjects: Batch size to compile simulation for ahead of time, or
            `None` to compile at runtime.

    Returns:
        The model, ready to `solve()` with `get_params()`.

    """
    wealth = LinSpacedGrid(start=WEALTH_MIN, stop=WEALTH_MAX, n_points=wealth_n_points)
    couple_wealth = LinSpacedGrid(
        start=2.0 * WEALTH_MIN, stop=2.0 * WEALTH_MAX, n_points=wealth_n_points
    )
    consumption = LinSpacedGrid(
        start=WEALTH_MIN, stop=2.0 * WEALTH_MAX, n_points=consumption_n_points
    )
    last_age = n_periods - 1
    probability = _transition_probabilities(last_age=last_age)

    couple = Regime(
        transition={
            "couple": ValueDependentTransition(
                probability=MarkovTransition(probability["stays_married"]),
                gate=_no_dissolution,
                routes={
                    "f": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f",
                            projection={"wealth": _half_of_couple_wealth},
                        ),
                    ),
                    "m": StakeholderRoute(
                        target_stakeholder="m",
                        fallback=ProjectedRegimeValue(
                            regime="single_m",
                            projection={"wealth": _half_of_couple_wealth},
                        ),
                    ),
                },
            ),
            "couple_terminal": MarkovTransition(probability["reaches_last_age"]),
        },
        active=lambda age: age < last_age,
        states={"wealth": couple_wealth},
        state_transitions={"wealth": _next_couple_wealth},
        actions={"consumption": consumption},
        functions={
            "utility": CollectiveUtility(
                utilities={"f": _couple_utility_f, "m": _couple_utility_m}
            )
        },
        constraints={
            "affordable": _consumption_within_couple_wealth,
            "participation_f": ValueDependentConstraint(
                predicate=_participation_f,
                references={
                    "V_alone_f": ProjectedRegimeValue(
                        regime="single_f",
                        projection={"wealth": _half_of_couple_wealth},
                    )
                },
            ),
            "participation_m": ValueDependentConstraint(
                predicate=_participation_m,
                references={
                    "V_alone_m": ProjectedRegimeValue(
                        regime="single_m",
                        projection={"wealth": _half_of_couple_wealth},
                    )
                },
            ),
        },
    )
    couple_terminal = Regime(
        transition=None,
        active=lambda age: age >= last_age,
        states={"wealth": couple_wealth},
        actions={"consumption": consumption},
        functions={
            "utility": CollectiveUtility(
                utilities={
                    "f": _terminal_utility_collective,
                    "m": _terminal_utility_collective,
                }
            )
        },
        constraints={"affordable": _consumption_within_couple_wealth},
    )
    single_f = Regime(
        transition={
            "couple": ValueDependentTransition(
                probability=MarkovTransition(probability["meets_a_partner"]),
                gate=_mutual_consent,
                routes={
                    "her": StakeholderRoute(
                        target_stakeholder="f",
                        fallback=ProjectedRegimeValue(
                            regime="single_f",
                            projection={"wealth": _half_of_couple_wealth},
                        ),
                    )
                },
                gate_references={
                    "V_alone_f": ProjectedRegimeValue(
                        regime="single_f",
                        projection={"wealth": _half_of_couple_wealth},
                    ),
                    "V_alone_m": ProjectedRegimeValue(
                        regime="single_m",
                        projection={"wealth": _half_of_couple_wealth},
                    ),
                },
            ),
            "single_f": MarkovTransition(probability["meets_nobody"]),
            "single_f_terminal": MarkovTransition(probability["reaches_last_age"]),
        },
        active=lambda age: age < last_age,
        states={"wealth": wealth},
        # Marrying pools two people's wealth, so the coordinate the couple's
        # grid is entered at is not the one her own regime would carry
        # forward. A law that ignored that would land every bride at the
        # bottom of the couple's grid, where the household is infeasible and
        # the consent gate shuts for reasons that are an artifact.
        state_transitions={
            "wealth": {
                "couple": _pooled_wealth,
                "single_f": _next_single_wealth,
                "single_f_terminal": _next_single_wealth,
            }
        },
        actions={"consumption": consumption},
        functions={"utility": _single_utility},
        constraints={"affordable": _consumption_within_single_wealth},
    )
    single_f_terminal = _single_terminal(
        wealth=wealth, consumption=consumption, last_age=last_age
    )
    # Spelled out rather than replaced off `single_f`: his transition names his
    # own regimes and enters the household in his own role. Replacing her
    # transition would carry her already-lowered gated edge forward and meet
    # his declaration of the same target, which is refused because the two
    # disagree on the routes.
    single_m = Regime(
        transition={
            "couple": ValueDependentTransition(
                probability=MarkovTransition(probability["meets_a_partner"]),
                gate=_mutual_consent,
                routes={
                    "his": StakeholderRoute(
                        target_stakeholder="m",
                        fallback=ProjectedRegimeValue(
                            regime="single_m",
                            projection={"wealth": _half_of_couple_wealth},
                        ),
                    )
                },
                gate_references={
                    "V_alone_f": ProjectedRegimeValue(
                        regime="single_f",
                        projection={"wealth": _half_of_couple_wealth},
                    ),
                    "V_alone_m": ProjectedRegimeValue(
                        regime="single_m",
                        projection={"wealth": _half_of_couple_wealth},
                    ),
                },
            ),
            "single_m": MarkovTransition(probability["meets_nobody"]),
            "single_m_terminal": MarkovTransition(probability["reaches_last_age"]),
        },
        active=lambda age: age < last_age,
        states={"wealth": wealth},
        state_transitions={
            "wealth": {
                "couple": _pooled_wealth,
                "single_m": _next_single_wealth,
                "single_m_terminal": _next_single_wealth,
            }
        },
        actions={"consumption": consumption},
        functions={"utility": _single_utility},
        constraints={"affordable": _consumption_within_single_wealth},
    )
    return Model(
        regimes={
            "couple": couple,
            "couple_terminal": couple_terminal,
            "single_f": single_f,
            "single_m": single_m,
            "single_f_terminal": single_f_terminal,
            "single_m_terminal": _single_terminal(
                wealth=wealth, consumption=consumption, last_age=last_age
            ),
        },
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        regime_id_class=RegimeId,
        n_subjects=n_subjects,
    )


def get_params(
    *,
    discount_factor: float = 0.95,
    marriage_premium: float = 1.0,
    interest_rate: float = 0.03,
    participation_slack: float = 0.0,
) -> dict:
    """Return the params dict `get_model()`'s model takes.

    Args:
        discount_factor: Per-period discount factor.
        marriage_premium: Additive flow gain from being in the household, in
            utils, identical for both partners.
        interest_rate: Return on wealth carried forward.
        participation_slack: How far below the outside option a partner will
            still stay. Zero is the strict participation constraint.

    Returns:
        Dict of per-regime parameter mappings.

    """
    return {
        "couple": {
            "koopmans_aggregator": {"discount_factor": discount_factor},
            "utility_f": {"marriage_premium": marriage_premium},
            "utility_m": {"marriage_premium": marriage_premium},
            "next_wealth": {"interest_rate": interest_rate},
            "participation_f": {"slack": participation_slack},
            "participation_m": {"slack": participation_slack},
        },
        "couple_terminal": {},
        "single_f": {
            "koopmans_aggregator": {"discount_factor": discount_factor},
            "next_wealth": {"interest_rate": interest_rate},
        },
        "single_m": {
            "koopmans_aggregator": {"discount_factor": discount_factor},
            "next_wealth": {"interest_rate": interest_rate},
        },
        "single_f_terminal": {},
        "single_m_terminal": {},
    }


def get_initial_conditions(*, n_subjects: int, model: Model) -> dict:
    """Return a cohort of singles, half women and half men.

    Args:
        n_subjects: Number of subjects.
        model: The model whose regime ids and role codes the cohort is coded
            against.

    Returns:
        Dict of the `wealth`, `age` and `regime_id` columns `simulate()` takes.
        No `own_stakeholder`: the whole cohort starts in a singleton regime, so
        nobody occupies a household role yet, and each subject is given one by
        the route it takes on entering the couple.

    """
    is_woman = jnp.arange(n_subjects) % 2 == 0
    return {
        "wealth": jnp.linspace(WEALTH_MIN, WEALTH_MAX, n_subjects),
        "age": jnp.zeros(n_subjects),
        "regime_id": jnp.where(
            is_woman,
            model.regime_names_to_ids["single_f"],
            model.regime_names_to_ids["single_m"],
        ).astype(jnp.int32),
    }


def _log_utility(consumption: ContinuousAction) -> FloatND:
    """Flow utility of consumption, finite on the whole grid."""
    return jnp.log(consumption)


def _single_utility(consumption: ContinuousAction) -> FloatND:
    return _log_utility(consumption)


def _terminal_utility(consumption: ContinuousAction) -> FloatND:
    return _log_utility(consumption)


def _couple_utility_f(
    *, consumption: ContinuousAction, marriage_premium: float
) -> FloatND:
    """Her flow utility: half of the household's consumption, plus the premium."""
    return _log_utility(0.5 * consumption) + marriage_premium


def _couple_utility_m(
    *, consumption: ContinuousAction, marriage_premium: float
) -> FloatND:
    """His flow utility, symmetric to hers."""
    return _log_utility(0.5 * consumption) + marriage_premium


def _terminal_utility_collective(consumption: ContinuousAction) -> FloatND:
    return _log_utility(0.5 * consumption)


def _next_single_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption)


def _pooled_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, interest_rate: float
) -> ContinuousState:
    """Household wealth on entry: two partners' savings, pooled."""
    return 2.0 * (1.0 + interest_rate) * (wealth - consumption)


def _next_couple_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption)


def _consumption_within_single_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> BoolND:
    return consumption <= wealth


def _consumption_within_couple_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> BoolND:
    return consumption <= wealth


def _half_of_couple_wealth(wealth: ContinuousState) -> ContinuousState:
    """A partner's share of household wealth, at the couple's own coordinate."""
    return 0.5 * wealth


def _participation_f(*, Q_f: FloatND, V_alone_f: FloatND, slack: float) -> BoolND:
    """She stays where the household is worth at least her outside option."""
    return Q_f >= V_alone_f - slack


def _participation_m(*, Q_m: FloatND, V_alone_m: FloatND, slack: float) -> BoolND:
    """He stays on the same terms."""
    return Q_m >= V_alone_m - slack


def _mutual_consent(
    *,
    V_target_f: FloatND,
    V_target_m: FloatND,
    V_alone_f: FloatND,
    V_alone_m: FloatND,
) -> BoolND:
    """Both partners must prefer the household; either one can refuse."""
    return (V_target_f > V_alone_f) & (V_target_m > V_alone_m)


def _no_dissolution(D_target: BoolND) -> BoolND:
    """The couple continues wherever its feasible set is not empty."""
    return ~D_target


def _single_terminal(
    *, wealth: LinSpacedGrid, consumption: LinSpacedGrid, last_age: int
) -> Regime:
    """Build one single's terminal regime, identical for her and for him."""
    return Regime(
        transition=None,
        active=lambda age: age >= last_age,
        states={"wealth": wealth},
        actions={"consumption": consumption},
        functions={"utility": _terminal_utility},
        constraints={"affordable": _consumption_within_single_wealth},
    )


def _transition_probabilities(*, last_age: int) -> dict:
    """Build the four age-dependent transition probabilities of one model.

    Every non-terminal regime is active up to `last_age` and hands its rows to
    its own terminal regime there, so the probabilities need the age at which
    that happens. Closing over it keeps the split a property of the model
    rather than a parameter a caller could set inconsistently with `ages`.

    Args:
        last_age: The age at which the terminal regimes take over.

    Returns:
        Dict of the four probability functions, keyed by what each one says:
        `meets_a_partner`, `meets_nobody`, `stays_married`, `reaches_last_age`.

    """

    def _before_last(*, age: FloatND, probability: float) -> FloatND:
        return jnp.where(age < last_age - 1, probability, 0.0)

    def meets_a_partner(age: FloatND) -> FloatND:
        return _before_last(age=age, probability=MEETING_PROBABILITY)

    def meets_nobody(age: FloatND) -> FloatND:
        return _before_last(age=age, probability=1.0 - MEETING_PROBABILITY)

    def stays_married(age: FloatND) -> FloatND:
        return _before_last(age=age, probability=1.0)

    def reaches_last_age(age: FloatND) -> FloatND:
        return jnp.where(age < last_age - 1, 0.0, 1.0)

    return {
        "meets_a_partner": meets_a_partner,
        "meets_nobody": meets_nobody,
        "stays_married": stays_married,
        "reaches_last_age": reaches_last_age,
    }
