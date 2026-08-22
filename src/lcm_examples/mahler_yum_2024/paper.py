"""Paper-mode Mahler & Yum (2024): continuous effort through NNBEGM.

The canonical (`implementation="paper"`) configuration replaces three
discretizations of the brute-force example with their exact counterparts:

- **Continuous effort.** The 40-class effort action becomes the continuous
  outer action of an `NNBEGM` solve: the habit (`lagged_effort`) is a
  continuous state on `[0, 1]`, the outer post-decision is
  `new_lagged_effort = effort`, and the keeper holds the habit through
  `keep_effort`. Functions that used the effort *class* read the bound
  post-decision instead (`effort_value = new_lagged_effort`), so both the
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
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Regime,
    affine_breakpoint,
    fixed_transition,
    piecewise_affine,
)
from lcm.consumption_savings_regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
)
from lcm.solvers import NBEGM, NNBEGM, AdaptiveOuterMesh, UniformObservedFixedCost
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
    legacy,
    next_health,
    pension,
    prod_shock_grid,
    productivity_type_multiplier,
    retirement_is_active,
    retirement_net_income,
    retirement_to_dead_probability,
    retirement_to_retirement_probability,
    risk_aversion,
    scaled_productivity_shock,
    taxed_income,
    work_disutility,
    working_is_active,
    working_net_income,
    working_to_dead_probability,
    working_to_retirement_probability,
    working_to_working_probability,
)

# Streaming knobs for the paper-mode configuration.
#
# `NBEGM`'s three streaming knobs all default to `0`, meaning every ride cell,
# discrete branch and liquid interval has its buffers in flight simultaneously.
# This model's product of non-liquid state axes and outer-mesh nodes makes that
# allocation impractical, so cells and discrete branches are streamed.
#
# Blocking is result-preserving: `test_value_is_invariant_to_envelope_cell_blocking`,
# `test_nbegm_branch_batch_size.py` and `test_nbegm_interval_batch.py` pin the
# solved value against the unstreamed reference to `atol=rtol=1e-10` -- agreement
# at that tolerance, not bit-for-bit, because blocking changes the XLA reduction
# order.
_PAPER_CELL_BLOCK_SIZE = 256

# `labor_supply` is a branch axis of the large candidate array. Streaming it
# reduces peak memory while the branch body remains compiled once.
_PAPER_BRANCH_BATCH_SIZE = 1

# `interval_batch_size` is left at `0`: the per-interval continuation buffers are
# not the binding term here, and streaming a dimension that is not the problem
# only costs time.

N_HABIT_GRID = 17
N_EFFORT_GRID = 17
N_CONSUMPTION_GRID = 50


def effort_value(new_lagged_effort: ContinuousState) -> FloatND:
    """The continuous effort choice, read through the bound outer node.

    Inside the nested solve the outer action itself is not visible to the
    inner problems; the outer post-decision (`new_lagged_effort = effort`)
    is — bound per adjuster candidate, and to `keep_effort` for the keeper.

    `new_lagged_effort` is the habit chosen THIS period: an ordinary function
    of this period's action, which the habit law then carries forward. It is
    not a `next_<state>` — that name is reserved for a transition's output and
    may not be read within the period.
    """
    return new_lagged_effort


def lagged_effort_value(lagged_effort: ContinuousState) -> FloatND:
    """The continuous habit state (identity — the state is the value)."""
    return lagged_effort


def keep_effort(lagged_effort: ContinuousState) -> FloatND:
    """The keeper's no-adjustment candidate: hold the habit."""
    return lagged_effort


def new_lagged_effort(effort: ContinuousAction) -> ContinuousState:
    """The habit chosen this period — the outer post-decision (unit slope)."""
    return effort


def next_lagged_effort(new_lagged_effort: ContinuousState) -> ContinuousState:
    """The habit law of motion: carry this period's chosen habit forward."""
    return new_lagged_effort


def working_utility(
    effort_cost: FloatND,
    work_disutility: FloatND,
    consumption_utility: FloatND,
) -> FloatND:
    """Flow utility while working."""
    return consumption_utility - work_disutility - effort_cost


def retirement_utility(effort_cost: FloatND, consumption_utility: FloatND) -> FloatND:
    """Flow utility after the work-only margin has disappeared."""
    return consumption_utility - effort_cost


def raw_cash_on_hand(
    net_income: FloatND,
    wealth: ContinuousState,
    gross_interest_rate: FloatND,
) -> FloatND:
    """Own liquid resources before transfers — affine in wealth per branch."""
    return net_income + wealth * gross_interest_rate


@piecewise_affine(
    output="cash_on_hand",
    variable="raw_cash_on_hand",
    breakpoints=(
        affine_breakpoint(threshold="min_consumption", kind="continuous_kink"),
    ),
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


def build_paper_solver(
    *,
    outer_search: AdaptiveOuterMesh | None = None,
    cell_block_size: int = _PAPER_CELL_BLOCK_SIZE,
    branch_batch_size: int = _PAPER_BRANCH_BATCH_SIZE,
    interval_batch_size: int = 0,
) -> NNBEGM:
    """Construct the paper-mode NNBEGM solver.

    The three streaming knobs are set rather than left at their `0` default;
    see `_PAPER_CELL_BLOCK_SIZE` for why. Pass `0` explicitly to restore the
    unstreamed behaviour (every cell and branch buffer in flight at once).
    """
    return NNBEGM(
        inner=NBEGM(
            savings_grid=IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
            cell_block_size=cell_block_size,
            branch_batch_size=branch_batch_size,
            interval_batch_size=interval_batch_size,
        ),
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


def build_working_regime(
    *, outer_search: AdaptiveOuterMesh | None = None
) -> NestedConsumptionSavingsRegime:
    """The paper-mode working regime with continuous effort and habit."""
    return NestedConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(working_to_working_probability),
            "retirement": MarkovTransition(working_to_retirement_probability),
            "dead": MarkovTransition(working_to_dead_probability),
        },
        active=working_is_active,
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
            "utility": working_utility,
            "new_lagged_effort": new_lagged_effort,
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
            "net_income": working_net_income,
            "taxed_income": taxed_income,
            "scaled_productivity_shock": scaled_productivity_shock,
            "adjustment_cost_scale": adjustment_cost_scale,
            "discount_factor": discount_factor,
        },
        constraints={},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="cash_on_hand",
            post_decision_state="saving",
        ),
        outer_continuous=OuterContinuousMargin(
            state="lagged_effort",
            action="effort",
            post_decision_state="new_lagged_effort",
            no_adjustment="keep_effort",
        ),
        solver=build_paper_solver(outer_search=outer_search),
    )


def build_retirement_regime(
    *, outer_search: AdaptiveOuterMesh | None = None
) -> NestedConsumptionSavingsRegime:
    """The paper-mode retirement regime without work-only dimensions."""
    return NestedConsumptionSavingsRegime(
        transition={
            "retirement": MarkovTransition(retirement_to_retirement_probability),
            "dead": MarkovTransition(retirement_to_dead_probability),
        },
        active=partial(retirement_is_active, final_age_alive=int(ages.values[-2])),
        states={
            "wealth": IrregSpacedGrid(points=_WEALTH_GRID_POINTS),
            "health": DiscreteGrid(Health),
            "lagged_effort": LinSpacedGrid(start=0.0, stop=1.0, n_points=N_HABIT_GRID),
            "education": DiscreteGrid(Education),
            "health_type": DiscreteGrid(HealthType),
            "discount_type": DiscreteGrid(DiscountType),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(next_health),
            "lagged_effort": next_lagged_effort,
            "education": fixed_transition("education"),
            "health_type": fixed_transition("health_type"),
            "discount_type": fixed_transition("discount_type"),
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.01, stop=30.0, n_points=N_CONSUMPTION_GRID
            ),
            "effort": LinSpacedGrid(start=0.0, stop=1.0, n_points=N_EFFORT_GRID),
        },
        functions={
            "utility": retirement_utility,
            "new_lagged_effort": new_lagged_effort,
            "effort_value": effort_value,
            "lagged_effort_value": lagged_effort_value,
            "keep_effort": keep_effort,
            "effort_cost": effort_cost,
            "consumption_utility": consumption_utility,
            "cash_on_hand": cash_on_hand,
            "raw_cash_on_hand": raw_cash_on_hand,
            "saving": saving,
            "net_income": retirement_net_income,
            "pension": pension,
            "adjustment_cost_scale": adjustment_cost_scale,
            "discount_factor": discount_factor,
        },
        constraints={},
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="cash_on_hand",
            post_decision_state="saving",
        ),
        outer_continuous=OuterContinuousMargin(
            state="lagged_effort",
            action="effort",
            post_decision_state="new_lagged_effort",
            no_adjustment="keep_effort",
        ),
        solver=build_paper_solver(outer_search=outer_search),
    )


def create_mahler_yum_model(
    *,
    implementation: str = "paper",
    outer_search: AdaptiveOuterMesh | None = None,
    compatibility: legacy.MahlerLegacyCompatibility | None = None,
    enable_jit: bool = True,
) -> Model:
    """Build the Mahler-Yum model in the requested implementation.

    `"paper"` is the canonical continuous-outer configuration; `"brute"`
    returns the historical grid-search model unchanged; `"legacy_fortran"`
    builds toward the historical-algorithm compatibility set in
    `compatibility` (defaulting to the full `legacy.LEGACY_FORTRAN`).

    Legacy support is per switch and honest: the historical finite-grid
    effort/saving searches over the five-node adjustment-cost grid ARE the
    brute configuration and are returned for that switch subset; a request
    enabling any solver switch the engine cannot yet build (the old-habit
    continuation needs solve-phase-only transition support) raises
    `NotImplementedError` naming exactly those switches instead of
    returning a model that does not implement its own manifest.
    Measurement-side switches never affect the model object and are
    consumed by the replication's `legacy_fortran` module.
    """
    if implementation == "brute":
        from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL  # noqa: PLC0415

        return MAHLER_YUM_MODEL
    if implementation == "legacy_fortran":
        compat = legacy.LEGACY_FORTRAN if compatibility is None else compatibility
        unbuildable = legacy.unimplemented_solver_switches(compat)
        if unbuildable:
            msg = (
                "legacy_fortran solver switches not implemented yet: "
                + ", ".join(unbuildable)
                + " (the historical finite-grid searches ARE implemented; "
                "restrict the compatibility object to those switches for a "
                "partial historical build)"
            )
            raise NotImplementedError(msg)
        from lcm_examples.mahler_yum_2024 import MAHLER_YUM_MODEL  # noqa: PLC0415

        return MAHLER_YUM_MODEL
    if implementation != "paper":
        msg = f"unknown implementation: {implementation!r}"
        raise ValueError(msg)
    if compatibility is not None and compatibility != legacy.CANONICAL:
        msg = (
            "the paper implementation is canonical by definition; historical "
            "switches require implementation='legacy_fortran': "
            + ", ".join(legacy.enabled_switches(compatibility))
        )
        raise ValueError(msg)
    return Model(
        regimes={
            "working": build_working_regime(outer_search=outer_search),
            "retirement": build_retirement_regime(outer_search=outer_search),
            "dead": build_dead_regime(),
        },
        ages=ages,
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
        fixed_params={
            "productivity_type_multiplier": productivity_type_multiplier,
            "sigma": risk_aversion,
            "health_intercept": health_intercept,
            "health_age_effects": health_age_effects,
            "good_health_coefficient": good_health_coefficient,
            "health_type_coefficient": health_type_coefficient,
            "college_coefficient": college_coefficient,
            "health_effort_coefficient": health_effort_coefficient,
            "lagged_health_effort_coefficient": (lagged_health_effort_coefficient),
            "survival_probs": _load_survival_probs(),
        },
    )


def adapt_params_to_paper_mode(model_params: dict) -> dict:
    """Copy the optimized model parameter mapping for paper mode."""
    return dict(model_params)
