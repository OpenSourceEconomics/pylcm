"""Example implementation of Mahler & Yum (2024).

This model replicates the lifecycle model from the paper "Lifestyle Behaviors and
Wealth-Health Gaps in Germany" by Lukas Mahler and Minchul Yum (Econometrica, 2024)
"""

from dataclasses import make_dataclass
from functools import partial
from typing import Any

import jax

jax.config.update("jax_enable_x64", val=False)

import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.interpolate import interp1d

import lcm
from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.dispatchers import _base_productmap
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    Float1D,
    FloatND,
    Int1D,
    Period,
    RegimeName,
)

# ======================================================================================
# Parameters
# ======================================================================================

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn_not_normalized: float = 57706.57
theta_val: Float1D = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
ages = AgeGrid(start=25, stop=101, step="2Y")
n: int = ages.n_periods
retirement_age: int = 19
taul: float = 0.128
lamda: float = 1.0 - 0.321
rho: float = 0.975
r: float = 1.04**2.0
tt0: float = 0.115
winit: Float1D = jnp.array([43978, 48201])
avrgearn = avrgearn_not_normalized / winit[1]
mincon0: float = 0.10
mincon = mincon0 * avrgearn


def calc_savingsgrid(x: Float1D) -> Float1D:
    x = ((jnp.log(10.0**2) - jnp.log(10.0**0)) / 49) * x
    x = jnp.exp(x)
    xgrid = x - 10.0 ** (0.0)
    xgrid = xgrid / (10.0**2 - 10.0**0.0)
    return xgrid * (30 - 0) + 0


# ======================================================================================
# Discrete Variables
# ======================================================================================


@categorical
class WorkingStatus:
    retired: int
    part: int
    full: int


@categorical
class EducationStatus:
    low: int
    high: int


Effort = make_dataclass(
    "HealthEffort", [("class" + str(i), int, int(i)) for i in range(40)]
)


@categorical
class HealthStatus:
    bad: int
    good: int


@categorical
class ProductivityType:
    low: int
    high: int


@categorical
class HealthType:
    low: int
    high: int


@categorical
class ProductivityShock:
    val0: int
    val1: int
    val2: int
    val3: int
    val4: int


@categorical
class RegimeId:
    alive: int
    dead: int


# ======================================================================================
# Model functions
# ======================================================================================


# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    wealth: ContinuousState,  # noqa: ARG001
    health_type: DiscreteState,  # noqa: ARG001
    education: DiscreteState,  # noqa: ARG001
    scaled_adjustment_cost: FloatND,
    fcost: FloatND,
    disutil: FloatND,
    cons_util: FloatND,
) -> FloatND:
    return cons_util - disutil - fcost - scaled_adjustment_cost


def disutil(
    working: DiscreteAction,
    health: DiscreteState,
    education: DiscreteState,
    period: Period,
    phigrid: FloatND,
) -> FloatND:
    return phigrid[period, education, health] * ((working / 2) ** (2)) / 2


def scaled_adjustment_cost(
    period: Period,
    adjustment_cost: ContinuousState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    chimaxgrid: Float1D,
) -> FloatND:
    return jnp.where(
        jnp.logical_not(effort == effort_t_1),
        adjustment_cost * chimaxgrid[period],
        0,
    )


def cnow(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> FloatND:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return jnp.maximum(net_income + (wealth) * r - (saving), mincon)


def cons_util(
    health: DiscreteState, cnow: FloatND, kappa: float, sigma: float, bb: float
) -> FloatND:
    mucon = jnp.where(health, 1, kappa)
    return mucon * (((cnow) ** (1.0 - sigma)) / (1.0 - sigma)) + mucon * bb


def fcost(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    effort: DiscreteAction,
    psi: float,
    xigrid: FloatND,
) -> FloatND:
    return (
        xigrid[period, education, health]
        * (eff_grid[effort] ** (1 + (1 / psi)))
        / (1 + (1 / psi))
    )


# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(benefits: FloatND, taxed_income: FloatND, pension: FloatND) -> FloatND:
    return taxed_income + pension + benefits


def scaled_productivity_shock(
    productivity_shock: ContinuousState, sigx: float
) -> FloatND:
    return productivity_shock * sigx


def income(
    working: DiscreteAction,
    period: Period,
    health: DiscreteState,
    education: DiscreteState,
    productivity: DiscreteState,
    scaled_productivity_shock: FloatND,
    income_grid: FloatND,
) -> FloatND:
    return (
        income_grid[period, health, education]
        * (working / 2)
        * theta_val[productivity]
        * jnp.exp(scaled_productivity_shock)
    )


def taxed_income(income: FloatND) -> FloatND:
    return lamda * (income ** (1.0 - taul)) * (avrgearn**taul)


def benefits(period: Period, health: DiscreteState, working: DiscreteAction) -> FloatND:
    eligible = jnp.logical_and(health == 0, working == 0)
    return jnp.where(
        jnp.logical_and(eligible, period <= retirement_age), tt0 * avrgearn, 0
    )


def pension(
    period: Period,
    education: DiscreteState,
    productivity: DiscreteState,
    income_grid: FloatND,
    penre: float,
) -> FloatND:
    return jnp.where(
        period > retirement_age,
        income_grid[19, 1, education] * theta_val[productivity] * penre,
        0,
    )


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(saving: ContinuousAction) -> ContinuousState:
    return saving


@lcm.mark.stochastic
def next_health(
    period: Period,
    health: DiscreteState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    education: DiscreteState,
    health_type: DiscreteState,
    health_transition: FloatND,
) -> FloatND:
    return health_transition[period, health, effort, effort_t_1, education, health_type]


def next_effort_t_1(effort: DiscreteAction) -> DiscreteState:
    return effort


# --------------------------------------------------------------------------------------
# Regime Transitions
# --------------------------------------------------------------------------------------
@lcm.mark.stochastic
def next_regime(
    period: Period,
    education: DiscreteState,
    health: DiscreteState,
    regime_transition: FloatND,
) -> FloatND:
    """Return probability array [P(alive), P(dead)] indexed by RegimeId."""
    survival_prob = regime_transition[period, education, health]
    return jnp.array([survival_prob, 1 - survival_prob])


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def retirement_constraint(period: Period, working: DiscreteAction) -> BoolND:
    return jnp.logical_not(jnp.logical_and(period > retirement_age, working > 0))


def savings_constraint(
    net_income: FloatND, wealth: ContinuousState, saving: ContinuousAction
) -> BoolND:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return net_income + (wealth) * r >= (saving)


# ======================================================================================
# Model specification
# ======================================================================================


def alive_is_active(age: float, final_age_alive: float) -> bool:
    return age <= final_age_alive


def dead_is_active(age: float, initial_age: float) -> bool:
    return age > initial_age


prod_shock_grid = lcm.shocks.ar1.Rouwenhorst(n_points=5, rho=rho, mu=0, sigma=1)

ALIVE_REGIME = Regime(
    transition=next_regime,
    active=partial(alive_is_active, final_age_alive=ages.values[-2]),
    states={
        "wealth": LinSpacedGrid(start=0, stop=49, n_points=50, transition=next_wealth),
        "health": DiscreteGrid(HealthStatus, transition=next_health),
        "productivity_shock": prod_shock_grid,
        "effort_t_1": DiscreteGrid(Effort, transition=next_effort_t_1),
        "adjustment_cost": lcm.shocks.iid.Uniform(n_points=5, start=0, stop=1),
        "education": DiscreteGrid(EducationStatus, transition=None),
        "productivity": DiscreteGrid(ProductivityType, transition=None),
        "health_type": DiscreteGrid(HealthType, transition=None),
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "saving": LinSpacedGrid(start=0, stop=49, n_points=50),
        "effort": DiscreteGrid(Effort),
    },
    functions={
        "utility": utility,
        "disutil": disutil,
        "fcost": fcost,
        "cons_util": cons_util,
        "cnow": cnow,
        "income": income,
        "benefits": benefits,
        "scaled_adjustment_cost": scaled_adjustment_cost,
        "net_income": net_income,
        "taxed_income": taxed_income,
        "pension": pension,
        "scaled_productivity_shock": scaled_productivity_shock,
    },
    constraints={
        "retirement_constraint": retirement_constraint,
        "savings_constraint": savings_constraint,
    },
)

DEAD_REGIME = Regime(
    transition=None,
    active=partial(dead_is_active, initial_age=ages.values[0]),
    functions={"utility": lambda: 0.0},
)

MAHLER_YUM_MODEL = Model(
    regimes={"alive": ALIVE_REGIME, "dead": DEAD_REGIME},
    ages=ages,
    regime_id_class=RegimeId,
)


# ======================================================================================
# Mahler & Yum starting params
# ======================================================================================


START_PARAMS = {
    # Disutility of work
    "nu": {
        "h": [2.63390750888379, 1.66602983591164, 1.27839561280412, 1.71439043350863],
        "u": [2.41177758126754, 1.8133670880598, 1.39103558901915, 2.41466980231321],
        "ad": 0.807247922589072,
    },
    # Disutility of effort
    "xi": {
        "hs": {
            "h": [
                0.146075197675677,
                0.55992411008533,
                1.04795036000287,
                1.60294886005945,
            ],
            "u": [
                0.628031290227532,
                1.36593242946612,
                1.64963812690034,
                0.734873142494319,
            ],
        },
        "cl": {
            "h": [
                0.091312997289004,
                0.302477689083851,
                0.739843441095022,
                1.36582077051777,
            ],
            "u": [
                0.46921037985024,
                0.996665589702672,
                1.65388250352532,
                1.08866246911941,
            ],
        },
    },
    # Income process
    "income_process": {
        "hs": {
            "y1": 0.899399488241831,
            "yt_s": 0.0615804210614531,
            "yt_sq": -0.00250769285750586,
            "wagep": 0.17769766414897,
        },
        "cl": {
            "y1": 1.1654726432446,
            "yt_s": 0.0874283672769353,
            "yt_sq": -0.00293713499239749,
            "wagep": 0.144836058314823,
        },
        "sigx": 0.0289408524185787,
    },
    # Effort habit adjustment cost max
    "chi": [0.000120437772838191, 0.14468204213946],
    # Discount ratio
    "beta": {"mean": 0.942749393405227, "std": 0.0283688760224992},
    # Elasticity of disutility of effort
    "psi": 1.11497911620865,
    # Utility constant
    "bb": 13.1079320277342,
    # Consumption utility penalty for unhealthy
    "conp": 0.871503495423925,
    # Pension replacement ratio
    "penre": 0.358766004066242,
    # Coefficient of relative risk-aversion
    "sigma": 2,
}

# ======================================================================================
# Model Input Construction
# ======================================================================================


def create_phigrid(nu: dict[str, list[float]]) -> FloatND:
    phi_interp_values = jnp.array([1, 8, 13, 20])
    phigrid = jnp.zeros((retirement_age + 1, 2, 2))
    health = ["u", "h"]
    for i in range(2):
        for j in range(2):
            interp_points = jnp.arange(1, retirement_age + 2)
            spline = interp1d(
                np.asarray(phi_interp_values),
                np.asarray(nu[health[j]]),
                kind="cubic",
            )
            temp_grid = jnp.asarray(spline(interp_points))
            temp_grid = jnp.where(
                i == 0, temp_grid * jnp.exp(jnp.array(nu["ad"])), temp_grid
            )
            phigrid = phigrid.at[:, i, j].set(temp_grid)
    return phigrid


def create_xigrid(xi: dict[str, dict[str, list[float]]]) -> FloatND:
    xi_interp_values = jnp.array([1, 12, 20, 31])
    xigrid = jnp.zeros((n, 2, 2))
    edu = ["hs", "cl"]
    health = ["u", "h"]
    for i in range(2):
        for j in range(2):
            interp_points = np.arange(1, 31)
            spline = interp1d(
                np.asarray(xi_interp_values),
                np.asarray(xi[edu[i]][health[j]]),
                kind="cubic",
            )
            temp_grid = jnp.asarray(spline(interp_points))
            xigrid = xigrid.at[0:30, i, j].set(temp_grid)
            xigrid = xigrid.at[30:n, i, j].set(xi[edu[i]][health[j]][3])
    return xigrid


def create_chimaxgrid(chi: list[float]) -> Float1D:
    t = jnp.arange(38)
    return jnp.maximum(chi[0] * jnp.exp(chi[1] * t), 0)


def create_income_grid(income_process: dict[str, dict[str, float]]) -> FloatND:
    sdztemp = ((income_process["sigx"] ** 2.0) / (1.0 - rho**2.0)) ** 0.5  # ty: ignore[unsupported-operator]
    j = jnp.arange(20)
    health = jnp.arange(2)
    education = jnp.arange(2)

    def calc_base(period: Period, health: Int1D, education: Int1D) -> Float1D:
        yt = jnp.where(
            education == 1,
            (
                income_process["cl"]["y1"]
                * jnp.exp(
                    income_process["cl"]["yt_s"] * (period)
                    + income_process["cl"]["yt_sq"] * (period) ** 2.0
                )
            )
            * (1.0 - income_process["cl"]["wagep"] * (1 - health)),
            (
                income_process["hs"]["y1"]
                * jnp.exp(
                    income_process["hs"]["yt_s"] * (period)
                    + income_process["hs"]["yt_sq"] * (period) ** 2.0
                )
            )
            * (1.0 - income_process["hs"]["wagep"] * (1 - health)),
        )
        return yt / (
            jnp.exp(((jnp.log(theta_val[1]) ** 2.0) ** 2.0) / 2.0)
            * jnp.exp(((sdztemp**2.0) ** 2.0) / 2.0)
        )

    mapped = _base_productmap(calc_base, ("_period", "health", "education"))
    return mapped(j, health, education)


# --------------------------------------------------------------------------------------
# Static Grids
# --------------------------------------------------------------------------------------
eff_grid: Float1D = jnp.linspace(0, 1, 40)
tr2yp_grid: FloatND = jnp.zeros((38, 2, 40, 40, 2, 2, 2))
j: Float1D = jnp.floor_divide(jnp.arange(38), 5)

# --------------------------------------------------------------------------------------
# Health Transition Probability Grid
# --------------------------------------------------------------------------------------

const_healthtr: float = -0.906
age_const: Float1D = jnp.asarray(
    [0.0, -0.289, -0.644, -0.881, -1.138, -1.586, -1.586, -1.586]
)
eff_param: Float1D = jnp.asarray([0.693, 0.734])
healthy_dummy: float = 2.311
htype_dummy: float = 0.632
college_dummy: float = 0.238


def health_trans(
    period: Period, health: Int1D, eff: Int1D, eff_1: Int1D, edu: Int1D, ht: Int1D
) -> Float1D:
    y = (
        const_healthtr
        + age_const[period]
        + edu * college_dummy
        + health * healthy_dummy
        + ht * htype_dummy
        + eff_grid[eff] * eff_param[0]
        + eff_grid[eff_1] * eff_param[1]
    )
    return jnp.exp(y) / (1.0 + jnp.exp(y))


mapped_health_trans = _base_productmap(
    health_trans, ("period", "health", "eff", "eff_1", "edu", "ht")
)

tr2yp_grid = tr2yp_grid.at[:, :, :, :, :, :, 1].set(
    mapped_health_trans(
        j, jnp.arange(2), jnp.arange(40), jnp.arange(40), jnp.arange(2), jnp.arange(2)
    )
)
tr2yp_grid = tr2yp_grid.at[:, :, :, :, :, :, 0].set(
    1.0 - tr2yp_grid[:, :, :, :, :, :, 1]
)


def create_regime_transition_grid() -> FloatND:
    surv_hs: FloatND = jnp.array(np.loadtxt("data/surv_hs.txt", skiprows=1))
    surv_cl: FloatND = jnp.array(np.loadtxt("data/surv_cl.txt", skiprows=1))
    spgrid: FloatND = jnp.zeros((38, 2, 2))
    spgrid = spgrid.at[:, 0, 0].set(surv_hs[:, 1])
    spgrid = spgrid.at[:, 1, 0].set(surv_cl[:, 1])
    spgrid = spgrid.at[:, 0, 1].set(surv_hs[:, 0])
    return spgrid.at[:, 1, 1].set(surv_cl[:, 0])


def create_inputs(
    seed: int,
    n_simulation_subjects: int,
    nu: dict[str, list[float]],
    xi: dict[str, dict[str, list[float]]],
    income_process: dict[str, dict[str, float] | float],
    chi: list[float],
    psi: float,
    bb: float,
    conp: float,
    penre: float,
    sigma: int,
) -> tuple[dict[RegimeName, Any], dict[RegimeName, Any], list[RegimeName]]:
    # Create variable grids from supplied parameters
    income_grid = create_income_grid(income_process)  # ty: ignore[invalid-argument-type]
    chimax_grid = create_chimaxgrid(chi)
    xvalues = prod_shock_grid.get_gridpoints()
    xtrans = prod_shock_grid.get_transition_probs()
    xi_grid = create_xigrid(xi)
    phi_grid = create_phigrid(nu)

    regime_transition = create_regime_transition_grid()

    params = {
        "disutil": {"phigrid": phi_grid},
        "fcost": {"psi": psi, "xigrid": xi_grid},
        "cons_util": {"sigma": sigma, "bb": bb, "kappa": conp},
        "income": {"income_grid": income_grid},
        "pension": {"income_grid": income_grid, "penre": penre},
        "scaled_adjustment_cost": {"chimaxgrid": chimax_grid},
        "scaled_productivity_shock": {"sigx": jnp.sqrt(income_process["sigx"])},  # ty: ignore[invalid-argument-type]
        "next_health": {"health_transition": tr2yp_grid},
        "next_regime": {"regime_transition": regime_transition},
    }

    # Create initial states for the simulation

    prod = jnp.zeros((16), dtype=jnp.int8)
    ht = jnp.zeros((16), dtype=jnp.int8)
    ed = jnp.zeros((16), dtype=jnp.int8)
    for i in range(1, 3):
        for j in range(1, 3):
            for k in range(1, 3):
                index = (i - 1) * 2 * 2 + (j - 1) * 2 + k - 1
                prod = prod.at[index].set(j - 1)
                ht = ht.at[index].set(1 - (k - 1))
                prod = prod.at[index + 8].set(j - 1)
                ht = ht.at[index + 8].set(1 - (k - 1))
                ed = ed.at[index + 8].set(1)
    init_distr_2b2t2h = jnp.array(np.loadtxt("data/init_distr_2b2t2h.txt", skiprows=1))
    initial_dists = jnp.diff(init_distr_2b2t2h[:, 0], prepend=0)
    eff_grid = jnp.linspace(0, 1, 40)
    key = random.key(seed)
    initial_wealth = jnp.full((n_simulation_subjects), 0, dtype=jnp.int8)
    types = random.choice(
        key, jnp.arange(16), (n_simulation_subjects,), p=initial_dists
    )
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0], (n_simulation_subjects,))
    health_thresholds = init_distr_2b2t2h[:, 1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = 1 - ht[types]
    initial_education = ed[types]
    initial_productivity = prod[types]
    initial_effort = jnp.searchsorted(eff_grid, init_distr_2b2t2h[:, 2][types])
    initial_adjustment_cost = random.uniform(new_keys[1], (n_simulation_subjects,))
    prod_dist = jax.lax.fori_loop(
        0,
        200,
        lambda i, a: a @ xtrans.T,  # noqa: ARG005
        jnp.full(5, 1 / 5),
    )
    initial_productivity_shock = xvalues[
        random.choice(new_keys[2], jnp.arange(5), (n_simulation_subjects,), p=prod_dist)
    ]
    initial_states = {
        "wealth": initial_wealth,
        "health": initial_health,
        "health_type": initial_health_type,
        "effort_t_1": initial_effort,
        "productivity_shock": initial_productivity_shock,
        "adjustment_cost": initial_adjustment_cost,
        "education": initial_education,
        "productivity": initial_productivity,
    }
    initial_regimes = ["alive"] * n_simulation_subjects
    return params, initial_states, initial_regimes


# ======================================================================================
# Solve and simulate the model
# ======================================================================================

if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("lcm")

    # Discount-factor heterogeneity: solve separate models per type.
    # Each type gets beta = beta_mean +/- beta_std.
    beta_mean = START_PARAMS["beta"]["mean"]
    beta_std = START_PARAMS["beta"]["std"]
    discount_factors = {
        "low": beta_mean - beta_std,
        "high": beta_mean + beta_std,
    }

    # Build common inputs (everything except discount_factor).
    start_params_without_beta = {k: v for k, v in START_PARAMS.items() if k != "beta"}
    common_params, initial_states, initial_regimes = create_inputs(
        seed=7235,
        n_simulation_subjects=1_000,
        **start_params_without_beta,  # ty: ignore[invalid-argument-type]
    )

    timings: list[float] = []
    for i in range(3):
        t0 = time.perf_counter()
        for beta_value in discount_factors.values():
            simulation_result = MAHLER_YUM_MODEL.solve_and_simulate(
                params={
                    "alive": {"discount_factor": beta_value, **common_params},
                },
                initial_states=initial_states,
                initial_regimes=initial_regimes,
                seed=8295,
            )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        logger.info("Run %d: %.3fs", i + 1, elapsed)

    logger.info("Timing summary:")
    for i, t in enumerate(timings):
        logger.info("  Run %d: %.3fs", i + 1, t)
