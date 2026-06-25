"""Dobrescu-Shanker (2024) housing model as a discrete-housing DC-EGM model.

The RFC column of the DS-2024 RFC-vs-NEGM housing comparison. In the source
(`InverseDCDP` `housing.py`) the keeper's endogenous liquid grid is refined
**per housing column** by the 1-D rooftop cut (`RFC.RFCSimple.rfc` called inside
`_refineKeeper` for each `(z, h)`), and the housing margin is handled by nesting
over the housing grid — it is *not* a two-dimensional inverse-Euler. So in pylcm
the RFC column is the discrete-choice DC-EGM with the 1-D RFC upper-envelope
backend, exactly as the DS-2026 App.2 EGM-FUES column
(`ds_app2_housing_fues.py`): discretise the next-housing choice onto the housing
grid, treat it as a discrete action, and the inner liquid DC-EGM plus the
discrete-choice envelope (RFC/FUES/MSS/LTM) selects the optimal `H'`.

The economics are the DS-2024 housing model (`ds2024_housing.py`): CRRA-plus-log
utility, two-state Markov income, the proportional house-trade cost. The
grid-search (VFI) twin solves the same discrete-housing problem by brute force —
the accuracy oracle. Faithful at `delta = 0` (keep is `H' = H`); the paper's
`delta = 0.10` keeper depreciates the held stock off the housing grid, awaiting
the housing-axis carry extension shared with the NEGM column.
"""

from typing import Literal

import jax.numpy as jnp

from lcm import (
    DCEGM,
    AgeGrid,
    DiscreteGrid,
    GridSearch,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models.ds2024_housing import (
    Income,
    income_transition,
    income_value,
)

START_AGE = 60


def _make_housing_levels(*, n_housing: int) -> type:
    """Create an ordered categorical with one field per discrete housing level.

    The class name is model-unique so it never collides with another model's
    dynamically built housing-level categorical when both are imported together.
    """
    annotations = {f"h{i}": ScalarInt for i in range(n_housing)}
    cls = type("DS2024HousingLevels", (), {"__annotations__": annotations})
    return categorical(ordered=True)(cls)


@categorical(ordered=False)
class DS2024HousingFuesRegimeId:
    """Lifecycle regimes: an alive housing regime and the terminal bequest."""

    alive: ScalarInt
    dead: ScalarInt


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    """Inner post-decision liquid balance `a' = resources - c`."""
    return resources - consumption


def next_liquid(savings: FloatND) -> ContinuousState:
    """Euler-state law: next liquid assets equal post-decision savings."""
    return savings


def next_housing(housing_choice: DiscreteAction) -> DiscreteState:
    """Discrete housing law: next housing equals the chosen code."""
    return housing_choice


def inverse_marginal_utility(marginal_continuation: FloatND, gamma_c: float) -> FloatND:
    """Invert the consumption marginal utility `u'(c) = c^{-gamma_C}`.

    `c = mc^{-1/gamma_C}`; the log housing-service term is separable and drops out.
    """
    return marginal_continuation ** (-1.0 / gamma_c)


def build_model(  # noqa: C901
    *,
    variant: Literal["dcegm", "brute"] = "dcegm",
    n_grid: int,
    n_housing: int | None = None,
    n_consumption: int = 60,
    n_savings: int | None = None,
    liquid_max: float = 50.0,
    housing_max: float = 50.0,
    housing_min: float = 0.01,
    n_periods: int = 4,
    upper_envelope: Literal["fues", "mss", "ltm", "rfc"] = "rfc",
) -> Model:
    """Build the DS-2024 discrete-housing model.

    Args:
        variant: `"dcegm"` builds the discrete-choice DC-EGM (the RFC/FUES column);
            `"brute"` builds the grid-search (VFI) twin — the accuracy oracle.
        n_grid: Number of liquid grid points (and clustered savings nodes).
        n_housing: Number of discrete housing levels; defaults to `n_grid`.
        n_consumption: Number of consumption-grid points (brute search).
        n_savings: Number of savings-grid nodes; defaults to `n_grid`.
        liquid_max: Upper bound of the liquid grid.
        housing_max: Upper bound of the housing-level grid.
        housing_min: Lower bound `b` of the housing levels (and liquid floor).
        n_periods: Number of model periods (the last is the terminal bequest).
        upper_envelope: DC-EGM upper-envelope backend; the RFC column is `"rfc"`.

    Returns:
        The alive discrete-housing regime plus the terminal bequest regime.
    """
    n_housing = n_grid if n_housing is None else n_housing
    n_savings = n_grid if n_savings is None else n_savings

    ages = AgeGrid(start=START_AGE, stop=START_AGE + n_periods - 1, step="Y")
    final_age = int(ages.exact_values[-1])

    stock_levels = jnp.asarray(
        [
            housing_min + (housing_max - housing_min) * i / (n_housing - 1)
            for i in range(n_housing)
        ]
    )
    housing_class = _make_housing_levels(n_housing=n_housing)
    housing_grid = DiscreteGrid(housing_class)
    liquid_grid = LinSpacedGrid(start=0.0, stop=liquid_max, n_points=n_grid)
    consumption_grid = LinSpacedGrid(
        start=0.05, stop=liquid_max, n_points=n_consumption
    )
    savings_grid = IrregSpacedGrid(
        points=tuple(
            housing_min
            + (liquid_max + housing_max - housing_min) * (i / (n_savings - 1)) ** 2
            for i in range(n_savings)
        )
    )

    def housing_stock(housing: DiscreteState) -> FloatND:
        """Held housing stock `h` of the current discrete housing state."""
        return stock_levels[housing]

    def serviced_housing(housing_choice: DiscreteAction) -> FloatND:
        """Serviced housing this period — the chosen next stock `H'`."""
        return stock_levels[housing_choice]

    def housing_cost(
        housing: DiscreteState,
        housing_choice: DiscreteAction,
        delta: float,
        return_housing: float,
        tau: float,
    ) -> FloatND:
        """Net liquid cost of moving the house from `h` to `H'`.

        - keep (`H' = h`): cost `0`;
        - adjust (`H' != h`): cost `(1 + tau)·H' - (1 + r_H)·h·(1 - delta)`.
        """
        round_trip = (1.0 + tau) * stock_levels[housing_choice] - (
            1.0 + return_housing
        ) * stock_levels[housing] * (1.0 - delta)
        return jnp.where(housing_choice == housing, 0.0, round_trip)

    def resources(
        liquid: ContinuousState,
        housing_cost: FloatND,
        income_value: FloatND,
        return_liquid: float,
    ) -> FloatND:
        """Liquid resources `(1 + r)·a + y - housing_cost`."""
        return (1.0 + return_liquid) * liquid + income_value - housing_cost

    def next_liquid_brute(
        liquid: ContinuousState,
        housing_cost: FloatND,
        income_value: FloatND,
        consumption: ContinuousAction,
        return_liquid: float,
    ) -> ContinuousState:
        """Brute-force liquid law: resources minus consumption."""
        return (
            (1.0 + return_liquid) * liquid + income_value - housing_cost - consumption
        )

    def borrowing_constraint(
        liquid: ContinuousState,
        housing_cost: FloatND,
        income_value: FloatND,
        consumption: ContinuousAction,
        return_liquid: float,
    ) -> BoolND:
        """Keep post-decision liquid assets non-negative (`a' >= 0`)."""
        return (
            (1.0 + return_liquid) * liquid + income_value - housing_cost - consumption
        ) >= 0.0

    def utility(
        consumption: ContinuousAction,
        serviced_housing: FloatND,
        gamma_c: float,
        alpha: float,
    ) -> FloatND:
        """CRRA consumption utility plus a log housing-service flow."""
        consumption_utility = (consumption ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)
        return consumption_utility + alpha * jnp.log(serviced_housing)

    def bequest(
        liquid: ContinuousState,
        housing: DiscreteState,
        return_liquid: float,
        theta: float,
        bequest_shift: float,
        gamma_c: float,
    ) -> FloatND:
        """Terminal bequest `theta·((K + (1+r)a + h)^{1-gamma} - 1)/(1 - gamma)`."""
        estate = bequest_shift + (1.0 + return_liquid) * liquid + stock_levels[housing]
        return theta * (estate ** (1.0 - gamma_c) - 1.0) / (1.0 - gamma_c)

    def next_regime(age: int) -> ScalarInt:
        """Stay alive until the final age, then enter the terminal bequest."""
        return jnp.where(
            age + 1 >= final_age,
            DS2024HousingFuesRegimeId.dead,
            DS2024HousingFuesRegimeId.alive,
        )

    dead = UserRegime(
        transition=None,
        active=lambda age, fa=final_age: age >= fa,
        states={"liquid": liquid_grid, "housing": housing_grid},
        functions={"utility": bequest},
    )

    shared = {
        "utility": utility,
        "housing_cost": housing_cost,
        "serviced_housing": serviced_housing,
        "housing_stock": housing_stock,
        "income_value": income_value,
    }

    if variant == "brute":
        alive = UserRegime(
            transition=next_regime,
            active=lambda age, fa=final_age: age < fa,
            states={
                "liquid": liquid_grid,
                "housing": housing_grid,
                "income": DiscreteGrid(Income),
            },
            state_transitions={
                "liquid": next_liquid_brute,
                "housing": next_housing,
                "income": MarkovTransition(income_transition),
            },
            actions={"consumption": consumption_grid, "housing_choice": housing_grid},
            constraints={"borrowing_constraint": borrowing_constraint},
            functions=shared,
            solver=GridSearch(),
        )
        return Model(
            regimes={"alive": alive, "dead": dead},
            ages=ages,
            regime_id_class=DS2024HousingFuesRegimeId,
        )

    inner_solver = DCEGM(
        continuous_state="liquid",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=savings_grid,
        upper_envelope=upper_envelope,
        n_constrained_points=32,
    )
    alive = UserRegime(
        transition=next_regime,
        active=lambda age, fa=final_age: age < fa,
        states={
            "liquid": liquid_grid,
            "housing": housing_grid,
            "income": DiscreteGrid(Income),
        },
        state_transitions={
            "liquid": next_liquid,
            "housing": next_housing,
            "income": MarkovTransition(income_transition),
        },
        actions={"consumption": consumption_grid, "housing_choice": housing_grid},
        functions={
            **shared,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=inner_solver,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=DS2024HousingFuesRegimeId,
    )


def build_params(
    *,
    variant: Literal["dcegm", "brute"] = "dcegm",
    tau: float = 0.20,
    delta: float = 0.0,
    discount_factor: float = 0.945,
    gamma_c: float = 1.458,
    alpha: float = 0.66,
    return_liquid: float = 0.024,
    return_housing: float = 0.10,
    theta: float = 2.0,
    bequest_shift: float = 200.0,
) -> dict:
    """Calibration parameters for the DS-2024 discrete-housing model.

    Args:
        variant: Must match `build_model`.
        tau: Proportional housing-transaction cost.
        delta: Housing depreciation.
        discount_factor: Discount factor `beta`.
        gamma_c: Consumption CRRA `gamma_C`.
        alpha: Housing-service weight.
        return_liquid: Liquid return `r`.
        return_housing: Housing return `r_H`.
        theta: Terminal bequest weight.
        bequest_shift: Terminal bequest shift `K`.

    Returns:
        The nested parameter template keyed by regime then function.
    """
    utility_params = {"gamma_c": gamma_c, "alpha": alpha}
    cost_params = {"delta": delta, "return_housing": return_housing, "tau": tau}
    bequest_params = {
        "return_liquid": return_liquid,
        "theta": theta,
        "bequest_shift": bequest_shift,
        "gamma_c": gamma_c,
    }
    if variant == "brute":
        alive = {
            "utility": utility_params,
            "housing_cost": cost_params,
            "next_liquid": {"return_liquid": return_liquid},
            "borrowing_constraint": {"return_liquid": return_liquid},
        }
    else:
        alive = {
            "utility": utility_params,
            "housing_cost": cost_params,
            "resources": {"return_liquid": return_liquid},
            "next_liquid": {},
            "inverse_marginal_utility": {"gamma_c": gamma_c},
        }
    return {
        "discount_factor": discount_factor,
        "alive": alive,
        "dead": {"utility": bequest_params},
    }
