"""A marriage market: singles marry under mutual consent, couples dissolve unilaterally.

Two partners, her (`f`) and him (`m`), differ only in income, $I_f > I_m$. Every
regime chooses a consumption share $s$ of its wealth, so consumption $c = s\\,w$ is
affordable everywhere and a regime's value moves smoothly between wealth nodes.

Single $i$ at wealth $w$:

$$
u_i(c) = \\log c, \\qquad w' = (1 + r)(w - c) + I_i .
$$

Couple at household wealth $W$, one common choice, each partner's own value:

$$
u_{\\mathrm{couple}, i}(c) = \\log(c / 2) + \\mu, \\qquad
W' = (1 + r)(W - c) + I_f + I_m .
$$

Moves within a period, each the `Exit` of the regime that decides:

- a single meets a partner with probability $p$ (the current shock `meet_i`; no
  meeting at the first age). Meeting, she or he moves into the couple only if both
  consent (`Consent`): each partner's couple value, with the couple's own rule for
  staying applied, strictly exceeds staying single. The partner met holds the same
  savings, so the couple starts at $W = 2(w - I_i) + I_f + I_m$; the counterparty
  compares with its own single value at $W / 2$. A refused single keeps its own
  wealth $w$.
- a couple dissolves if either partner prefers to leave (`Unilateral`, an exactly
  indifferent partner leaves): partner $i$ compares the couple's value with her or
  his single value at $W / 2$ minus the divorce cost.

Her income is the higher one, so for her the marriage premium $\\mu$ has to outweigh
giving up part of it, and it does so only once the household is rich enough that
income matters less than wealth: her outside option, not his, rules out the poorest
matches.

Everything is log utility over consumption with a fixed marriage premium, so the
model is small, has no free structure, and scales along its grid knobs —
`n_periods`, `wealth_n_points` and `consumption_n_points` — which is what makes it
usable as a benchmark workload as well as a worked example.
"""

from collections.abc import Callable, Mapping
from typing import Literal

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    ByAge,
    CollectiveUtility,
    Consent,
    Counterparty,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exits import Exit, ExitTarget, Unilateral
from lcm.shocks import Discrete, ShockGroup
from lcm.transition import StochasticTransition, Transition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    IntND,
    ScalarInt,
    StateName,
    UserParamsNode,
)
from lcm_examples import with_lottery_contraction

# Lowest wealth node, kept strictly positive so log utility stays finite.
WEALTH_MIN = 1.0

# Highest wealth node.
WEALTH_MAX = 20.0

# Income she receives every period, single or married. Higher than his, so
# living alone keeps all of it while the household splits the pooled income
# evenly: hers is the outside option that rules out the poorest matches.
INCOME_F = 3.0

# Income he receives every period, single or married.
INCOME_M = WEALTH_MIN

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


@categorical(ordered=False)
class Meet:
    """Whether a single meets a partner this period."""

    no: ScalarInt
    yes: ScalarInt


def get_model(
    *,
    n_periods: int = 5,
    wealth_n_points: int = 20,
    consumption_n_points: int = 20,
    execution_config: ExecutionConfig | None = None,
    lottery_contraction: bool = False,
) -> Model:
    """Build the marriage-market model.

    Args:
        n_periods: Number of lifecycle periods. The last one is terminal.
        wealth_n_points: Nodes on every regime's wealth grid.
        consumption_n_points: Nodes on every regime's consumption-share grid.
        execution_config: Optional hardware-local execution controls.
        lottery_contraction: Passed to every regime's `GridSearch` solver.

    Returns:
        The model, ready to `solve()` with `get_params()`.

    """
    wealth = LinSpacedGrid(start=WEALTH_MIN, stop=WEALTH_MAX, n_points=wealth_n_points)
    couple_wealth = LinSpacedGrid(
        start=2.0 * WEALTH_MIN, stop=2.0 * WEALTH_MAX, n_points=wealth_n_points
    )
    # Consumption is chosen as a share of wealth, so every choice is affordable
    # at every wealth, on the grid or between its nodes. A choice in levels is
    # affordable only from some wealth on, which makes a household's value
    # jump between the nodes its outside option is interpolated across.
    consumption_share = LinSpacedGrid(
        start=1.0 / consumption_n_points, stop=1.0, n_points=consumption_n_points
    )
    last_age = n_periods - 1
    probability = _transition_probabilities(last_age=last_age)
    couple = Regime(
        states={"wealth": couple_wealth},
        state_transitions={"wealth": _next_couple_wealth},
        actions={"consumption_share": consumption_share},
        functions={
            "consumption": _consumption,
            "utility": CollectiveUtility(
                utilities={"f": _couple_utility_f, "m": _couple_utility_m}
            ),
        },
        exit=Exit(
            targets={
                member: ExitTarget(
                    regime=f"single_{member}",
                    projection={"wealth": _half_of_couple_wealth},
                    utility_adjustment=_divorce_cost,
                )
                for member in ("f", "m")
            },
            rule=Unilateral(tie="leave"),
        ),
    )
    couple_terminal = Regime(
        states={"wealth": couple_wealth},
        actions={"consumption_share": consumption_share},
        functions={
            "consumption": _consumption,
            "utility": CollectiveUtility(
                utilities={
                    "f": _terminal_utility_collective,
                    "m": _terminal_utility_collective,
                }
            ),
        },
    )
    meeting = (1.0 - MEETING_PROBABILITY, MEETING_PROBABILITY)
    return Model(
        edges={
            regime: Transition(
                law=_until_last_age(
                    {
                        regime: StochasticTransition(func=probability["continues"]),
                        f"{regime}_terminal": StochasticTransition(
                            func=probability["reaches_last_age"]
                        ),
                    },
                    terminal=f"{regime}_terminal",
                    last_age=last_age,
                )
            )
            for regime in ("couple", "single_f", "single_m")
        },
        regimes=with_lottery_contraction(
            regimes={
                "couple": couple,
                "couple_terminal": couple_terminal,
                "single_f": _single(
                    member="f", wealth=wealth, consumption_share=consumption_share
                ),
                "single_m": _single(
                    member="m", wealth=wealth, consumption_share=consumption_share
                ),
                "single_f_terminal": _single_terminal(
                    wealth=wealth, consumption_share=consumption_share
                ),
                "single_m_terminal": _single_terminal(
                    wealth=wealth, consumption_share=consumption_share
                ),
            },
            lottery_contraction=lottery_contraction,
        ),
        shocks={
            f"meeting_{member}": ShockGroup(
                shocks={f"meet_{member}": Discrete(categories=Meet, probs=meeting)}
            )
            for member in ("f", "m")
        },
        ages=AgeGrid(start=0, inclusive_stop=n_periods - 1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: ("couple", "single_f", "single_m")},
        execution_config=(
            ExecutionConfig() if execution_config is None else execution_config
        ),
    )


def get_params(
    *,
    discount_factor: float = 0.95,
    marriage_premium: float = 0.3,
    interest_rate: float = 0.03,
    divorce_cost: float = 0.0,
) -> dict[str, UserParamsNode]:
    """Return the params dict `get_model()`'s model takes.

    Args:
        discount_factor: Per-period discount factor.
        marriage_premium: Additive flow gain from being in the household, in
            utils, identical for both partners.
        interest_rate: Return on wealth carried forward.
        divorce_cost: Utils a partner gives up on leaving the household, which
            also lowers the outside option the partner compares with staying.

    Returns:
        Dict of per-regime parameter mappings.

    """
    return {
        "couple": {
            "koopmans_aggregator": {"discount_factor": discount_factor},
            "utility_f": {"marriage_premium": marriage_premium},
            "utility_m": {"marriage_premium": marriage_premium},
            "next_wealth": {"interest_rate": interest_rate},
            "exit": {
                "targets": {
                    member: {"utility_adjustment": {"divorce_cost": divorce_cost}}
                    for member in ("f", "m")
                }
            },
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


def get_initial_conditions(
    *, n_subjects: int, model: Model
) -> dict[StateName | Literal["regime_id"], FloatND | IntND]:
    """Return a cohort of singles, half women and half men.

    Args:
        n_subjects: Number of subjects.
        model: The model whose regime ids and role codes the cohort is coded
            against.

    Returns:
        Dict of the `wealth`, `age` and `regime_id` columns `simulate()` takes.
        No `own_member`: the whole cohort starts in a singleton regime, so
        nobody occupies a household role yet, and each subject is given one by
        the exit it takes into the couple.

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


def _single(
    *,
    member: Literal["f", "m"],
    wealth: LinSpacedGrid,
    consumption_share: LinSpacedGrid,
) -> Regime:
    """Build single `member`'s regime, with formation as a consented exit."""
    spouse = "m" if member == "f" else "f"
    return Regime(
        states={"wealth": wealth},
        state_transitions={
            "wealth": _next_single_wealth_f if member == "f" else _next_single_wealth_m
        },
        actions={"consumption_share": consumption_share},
        functions={"consumption": _consumption, "utility": _single_utility},
        shocks=(f"meeting_{member}",),
        exit=Exit(
            targets={
                "self": ExitTarget(
                    regime="couple",
                    role=member,
                    projection={
                        "wealth": _pooled_from_f if member == "f" else _pooled_from_m
                    },
                )
            },
            rule=Consent(
                counterparty=Counterparty(
                    regime=f"single_{spouse}",
                    role=spouse,
                    projection={
                        "wealth": _half_pooled_from_f
                        if member == "f"
                        else _half_pooled_from_m
                    },
                )
            ),
            available=_meets_f if member == "f" else _meets_m,
            observes=(f"meet_{member}",),
        ),
    )


def _log_utility(consumption: FloatND) -> FloatND:
    """Flow utility of consumption, finite on the whole grid."""
    return jnp.log(consumption)


def _single_utility(consumption: FloatND) -> FloatND:
    return _log_utility(consumption)


def _terminal_utility(consumption: FloatND) -> FloatND:
    return _log_utility(consumption)


def _couple_utility_f(*, consumption: FloatND, marriage_premium: float) -> FloatND:
    """Her flow utility: half of the household's consumption, plus the premium."""
    return _log_utility(0.5 * consumption) + marriage_premium


def _couple_utility_m(*, consumption: FloatND, marriage_premium: float) -> FloatND:
    """His flow utility, symmetric to hers."""
    return _log_utility(0.5 * consumption) + marriage_premium


def _terminal_utility_collective(consumption: FloatND) -> FloatND:
    return _log_utility(0.5 * consumption)


def _consumption(
    *, wealth: ContinuousState, consumption_share: ContinuousAction
) -> FloatND:
    return consumption_share * wealth


def _next_single_wealth_f(
    *, wealth: ContinuousState, consumption: FloatND, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption) + INCOME_F


def _next_single_wealth_m(
    *, wealth: ContinuousState, consumption: FloatND, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption) + INCOME_M


def _next_couple_wealth(
    *, wealth: ContinuousState, consumption: FloatND, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption) + INCOME_F + INCOME_M


def _half_of_couple_wealth(wealth: ContinuousState) -> ContinuousState:
    """A partner's share of household wealth, at the couple's own coordinate."""
    return 0.5 * wealth


def _pooled(*, wealth: ContinuousState, own_income: float) -> ContinuousState:
    """Household wealth of a match: both partners' savings and both incomes.

    The partner met holds the same savings `wealth - own_income`. The result is
    clipped from below to the couple's lowest node, which only singles poorer
    than their own income reach.
    """
    pooled = 2.0 * (wealth - own_income) + INCOME_F + INCOME_M
    return jnp.maximum(pooled, 2.0 * WEALTH_MIN)


def _pooled_from_f(wealth: ContinuousState) -> ContinuousState:
    return _pooled(wealth=wealth, own_income=INCOME_F)


def _pooled_from_m(wealth: ContinuousState) -> ContinuousState:
    return _pooled(wealth=wealth, own_income=INCOME_M)


def _half_pooled_from_f(wealth: ContinuousState) -> ContinuousState:
    """The partner's single coordinate when she proposes: half the pooled wealth."""
    return jnp.maximum(0.5 * _pooled_from_f(wealth), WEALTH_MIN)


def _half_pooled_from_m(wealth: ContinuousState) -> ContinuousState:
    """The partner's single coordinate when he proposes: half the pooled wealth."""
    return jnp.maximum(0.5 * _pooled_from_m(wealth), WEALTH_MIN)


def _meets(*, meet: IntND, age: FloatND) -> BoolND:
    """A single can marry after meeting a partner, from the second age on."""
    return (meet == Meet.yes) & (age >= 1)


def _meets_f(*, meet_f: IntND, age: FloatND) -> BoolND:
    return _meets(meet=meet_f, age=age)


def _meets_m(*, meet_m: IntND, age: FloatND) -> BoolND:
    return _meets(meet=meet_m, age=age)


def _divorce_cost(divorce_cost: float) -> FloatND:
    """A partner who leaves gives up `divorce_cost` utils of the single payoff."""
    return jnp.asarray(-divorce_cost)


# keyword-only-exempt: primary-argument=law
def _until_last_age(
    law: Mapping[str, StochasticTransition], *, terminal: str, last_age: int
) -> ByAge:
    """Apply `law` until the age before `last_age`, then only its `terminal` cell."""
    return ByAge.until(
        stop_age_exclusive=last_age, law=law, then={terminal: law[terminal]}
    )


def _single_terminal(
    *, wealth: LinSpacedGrid, consumption_share: LinSpacedGrid
) -> Regime:
    """Build one single's terminal regime, identical for her and for him."""
    return Regime(
        states={"wealth": wealth},
        actions={"consumption_share": consumption_share},
        functions={"consumption": _consumption, "utility": _terminal_utility},
    )


def _transition_probabilities(
    *, last_age: int
) -> dict[str, Callable[[FloatND], FloatND]]:
    """Build the two age-dependent transition probabilities of one model.

    Every non-terminal regime is solved up to `last_age` and hands its rows to
    its own terminal regime there, so the probabilities need the age at which
    that happens. Closing over it keeps the split a property of the model
    rather than a parameter a caller could set inconsistently with `ages`.

    Args:
        last_age: The age at which the terminal regimes take over.

    Returns:
        Dict of the two probability functions, keyed by what each one says:
        `continues` (the regime carries on) and `reaches_last_age`.

    """

    def continues(age: FloatND) -> FloatND:
        return jnp.where(age < last_age - 1, 1.0, 0.0)

    def reaches_last_age(age: FloatND) -> FloatND:
        return jnp.where(age < last_age - 1, 0.0, 1.0)

    return {"continues": continues, "reaches_last_age": reaches_last_age}
