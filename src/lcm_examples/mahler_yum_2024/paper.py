"""Paper-mode Mahler & Yum (2024): continuous effort through NNBEGM.

The canonical (`implementation="paper"`) configuration replaces three
discretizations of the brute-force example with their exact counterparts:

- **Continuous effort.** The 40-class effort action becomes the continuous
  outer action of an `NNBEGM` solve: the habit (`lagged_effort`) is a
  continuous state on `[0, 1]`, the outer post-decision is
  `next_lagged_effort = effort`, and the keeper holds the habit through
  `keep_effort`. Functions that used the effort *class* read the bound
  post-decision instead (`effort_value = next_lagged_effort`), so both the
  keeper and every adjuster candidate evaluate the same DAG.
- **Analytic adjustment cost.** The five-node `adjustment_cost` solve state
  disappears; the uniform observed fixed cost is integrated in closed form
  by `UniformObservedFixedCost` (scale `adjustment_cost_envelope[period]`),
  and the analytic adjustment probability is published through the solver
  diagnostics.
- **Consumption as the Euler action.** The savings-grid action becomes an
  inner NB-EGM consumption-saving solve on `wealth`: `cash_on_hand` is the
  budget target and `saving = cash_on_hand - consumption` the post-decision.
  The paper's guaranteed minimum consumption (transfers top consumption up
  to 10% of average earnings; Section 3.1 and `mincon` in the authors'
  Fortran) is a *declared* flat budget piece:
  `cash_on_hand = max(raw_cash_on_hand, min_consumption)`, kinked in the
  derived `raw_cash_on_hand` at the `min_consumption` threshold.
  KNOWN DEVIATION from the Fortran's
  `c = max(coh - a', mincon)`: on the floor the Fortran tops up consumption
  no matter how much is saved (saving capped at own resources), while the
  floored budget here lets the agent split `min_consumption` between
  consumption and saving. The two differ only at states with
  `net_income + R*wealth < min_consumption` (bottom wealth node, zero-income
  branches); resolving the floor exactly needs a kinked-utility (case-piece)
  Euler action, which the ride-along route does not yet compose with.

Everything else — income, taxes, pensions, benefits, health transitions,
survival, preference heterogeneity — is imported unchanged from the
brute-force module, so the two configurations cannot drift apart silently.
"""

from functools import partial

import jax.numpy as jnp

from lcm import (
    NBEGM,
    NNBEGM,
    AdaptiveOuterMesh,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    UniformObservedFixedCost,
    affine_breakpoint,
    fixed_transition,
    piecewise_affine,
)
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    Period,
)
from lcm_examples.mahler_yum_2024 import (
    _WEALTH_GRID_POINTS,
    DiscountType,
    Education,
    Health,
    HealthType,
    LaborSupply,
    ProductivityType,
    RegimeId,
    _load_survival_probs,
    ages,
    alive_is_active,
    base_income,
    benefits,
    college_coefficient,
    consumption_utility,
    dead_is_active,
    discount_factor,
    effort_cost,
    good_health_coefficient,
    health_age_effects,
    health_effort_coefficient,
    health_intercept,
    health_type_coefficient,
    income,
    lagged_health_effort_coefficient,
    net_income,
    next_health,
    next_regime,
    pension,
    prod_shock_grid,
    productivity_type_multiplier,
    retirement_constraint,
    risk_aversion,
    scaled_productivity_shock,
    taxed_income,
    work_disutility,
)

N_HABIT_GRID = 17
N_EFFORT_GRID = 17
N_CONSUMPTION_GRID = 50


def effort_value(next_lagged_effort: ContinuousState) -> FloatND:
    """The continuous effort choice, read through the bound outer node.

    Inside the nested solve the outer action itself is not visible to the
    inner problems; the outer post-decision (`next_lagged_effort = effort`)
    is — bound per adjuster candidate, and to `keep_effort` for the keeper.
    """
    return next_lagged_effort


def lagged_effort_value(lagged_effort: ContinuousState) -> FloatND:
    """The continuous habit state (identity — the state is the value)."""
    return lagged_effort


def keep_effort(lagged_effort: ContinuousState) -> FloatND:
    """The keeper's no-adjustment candidate: hold the habit."""
    return lagged_effort


def next_lagged_effort(effort: ContinuousAction) -> ContinuousState:
    """The habit law of motion — the outer post-decision (unit slope)."""
    return effort


def utility(
    effort_cost: FloatND,
    work_disutility: FloatND,
    consumption_utility: FloatND,
) -> FloatND:
    """Flow utility; the adjustment cost is folded analytically, not here."""
    return consumption_utility - work_disutility - effort_cost


def raw_cash_on_hand(
    net_income: FloatND,
    wealth: ContinuousState,
    gross_interest_rate: FloatND,
) -> FloatND:
    """Own liquid resources before transfers — affine in wealth per branch."""
    return net_income + wealth * gross_interest_rate


@piecewise_affine(
    "cash_on_hand",
    variable="raw_cash_on_hand",
    breakpoints=(affine_breakpoint("min_consumption", kind="continuous_kink"),),
)
def cash_on_hand(
    raw_cash_on_hand: FloatND,
    min_consumption: FloatND,
) -> FloatND:
    """Liquid resources the inner problem divides, floored by transfers.

    The paper's guaranteed minimum consumption enters as a flat budget piece
    where own resources fall below `min_consumption` (see the module
    docstring for the exact Fortran-semantics deviation).
    """
    return jnp.maximum(raw_cash_on_hand, min_consumption)


def saving(cash_on_hand: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance."""
    return cash_on_hand - consumption


def next_wealth(saving: FloatND) -> ContinuousState:
    return saving


def adjustment_cost_scale(period: Period, adjustment_cost_envelope: FloatND) -> FloatND:
    """Scale `B` of the uniform observed fixed adjustment cost, per period."""
    return adjustment_cost_envelope[period]


def dead_utility(
    wealth: ContinuousState,
    discount_type: DiscreteState,  # noqa: ARG001
) -> FloatND:
    """Dead-regime utility: identically zero, on an explicit wealth axis.

    An EGM parent reads its terminal target's *carry* — value and marginal
    on the target's Euler axis — so unlike the brute-force dead regime this
    one must declare `wealth` (marginal is exactly zero: no bequests).
    `discount_type` mirrors the alive regime's fixed state, as in the brute
    module.
    """
    return jnp.zeros_like(wealth)


def build_dead_regime() -> Regime:
    """The paper-mode dead regime (terminal, with the Euler axis declared)."""
    return Regime(
        transition=None,
        active=partial(dead_is_active, initial_age=int(ages.values[0])),
        states={
            "wealth": IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
            "discount_type": DiscreteGrid(DiscountType),
        },
        functions={"utility": dead_utility},
    )


def build_paper_solver(*, outer_search: AdaptiveOuterMesh | None = None) -> NNBEGM:
    """The paper-mode NNBEGM solver (plan section 12.1's target interface)."""
    return NNBEGM(
        inner=NBEGM(
            continuous_state="wealth",
            post_decision_function="saving",
            budget_target="cash_on_hand",
            savings_grid=IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
        ),
        outer_action="effort",
        outer_post_decision="next_lagged_effort",
        outer_no_adjustment_candidate="keep_effort",
        outer_search=outer_search
        if outer_search is not None
        else AdaptiveOuterMesh(
            initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=17),
            max_nodes=129,
            max_refinement_rounds=6,
        ),
        branch_aggregator=UniformObservedFixedCost(
            shock_name="adjustment_cost",
            scale_function="adjustment_cost_scale",
            lower=0.0,
            upper=1.0,
        ),
    )


def build_alive_regime(*, outer_search: AdaptiveOuterMesh | None = None) -> Regime:
    """The paper-mode alive regime (continuous effort and habit)."""
    return Regime(
        transition=MarkovTransition(next_regime),
        active=partial(alive_is_active, final_age_alive=int(ages.values[-2])),
        states={
            "wealth": IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
            "health": DiscreteGrid(Health),
            "productivity_shock": prod_shock_grid,
            "lagged_effort": LinSpacedGrid(start=0.0, stop=1.0, n_points=N_HABIT_GRID),
            "education": DiscreteGrid(Education),
            "productivity": DiscreteGrid(ProductivityType),
            "health_type": DiscreteGrid(HealthType),
            "discount_type": DiscreteGrid(DiscountType),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(next_health),
            "lagged_effort": next_lagged_effort,
            "education": fixed_transition("education"),
            "productivity": fixed_transition("productivity"),
            "health_type": fixed_transition("health_type"),
            "discount_type": fixed_transition("discount_type"),
        },
        actions={
            "labor_supply": DiscreteGrid(LaborSupply),
            "consumption": LinSpacedGrid(
                start=0.01, stop=30.0, n_points=N_CONSUMPTION_GRID
            ),
            "effort": LinSpacedGrid(start=0.0, stop=1.0, n_points=N_EFFORT_GRID),
        },
        functions={
            "utility": utility,
            "effort_value": effort_value,
            "lagged_effort_value": lagged_effort_value,
            "keep_effort": keep_effort,
            "work_disutility": work_disutility,
            "effort_cost": effort_cost,
            "consumption_utility": consumption_utility,
            "cash_on_hand": cash_on_hand,
            "raw_cash_on_hand": raw_cash_on_hand,
            "saving": saving,
            "base_income": base_income,
            "income": income,
            "benefits": benefits,
            "net_income": net_income,
            "taxed_income": taxed_income,
            "pension": pension,
            "scaled_productivity_shock": scaled_productivity_shock,
            "adjustment_cost_scale": adjustment_cost_scale,
            "discount_factor": discount_factor,
        },
        constraints={"retirement_constraint": retirement_constraint},
        solver=build_paper_solver(outer_search=outer_search),
    )


def create_mahler_yum_model(
    *,
    implementation: str = "paper",
    outer_search: AdaptiveOuterMesh | None = None,
) -> Model:
    """Build the Mahler-Yum model in the requested implementation.

    `"paper"` is the canonical continuous-outer configuration;
    `"brute"` returns the historical grid-search model unchanged;
    `"legacy_fortran"` (historical-algorithm compatibility) is not yet
    implemented.
    """
    if implementation == "brute":
        from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL  # noqa: PLC0415

        return MAHLER_YUM_MODEL
    if implementation == "legacy_fortran":
        msg = "the legacy_fortran configuration is not implemented yet"
        raise NotImplementedError(msg)
    if implementation != "paper":
        msg = f"unknown implementation: {implementation!r}"
        raise ValueError(msg)
    return Model(
        regimes={
            "alive": build_alive_regime(outer_search=outer_search),
            "dead": build_dead_regime(),
        },
        ages=ages,
        regime_id_class=RegimeId,
        fixed_params={
            "alive": {
                "productivity_type_multiplier": productivity_type_multiplier,
                "consumption_utility": {"sigma": risk_aversion},
                "next_health": {
                    "health_intercept": health_intercept,
                    "health_age_effects": health_age_effects,
                    "good_health_coefficient": good_health_coefficient,
                    "health_type_coefficient": health_type_coefficient,
                    "college_coefficient": college_coefficient,
                    "health_effort_coefficient": health_effort_coefficient,
                    "lagged_health_effort_coefficient": (
                        lagged_health_effort_coefficient
                    ),
                },
                "next_regime": {"transition_probs": _load_survival_probs()},
            },
        },
    )


def adapt_params_to_paper_mode(model_params: dict) -> dict:
    """Rewrite brute-mode `create_inputs` params for the paper configuration.

    One mechanical change: the adjustment-cost envelope moves from the
    dropped `adjustment_cost_penalty` function to `adjustment_cost_scale`.
    Everything else — including `min_consumption`, consumed here by the
    floored `cash_on_hand` schedule — passes through unchanged.
    """
    adapted = dict(model_params)
    penalty = adapted.pop("adjustment_cost_penalty", None)
    if penalty is not None:
        adapted["adjustment_cost_scale"] = {
            "adjustment_cost_envelope": penalty["adjustment_cost_envelope"],
        }
    return adapted
