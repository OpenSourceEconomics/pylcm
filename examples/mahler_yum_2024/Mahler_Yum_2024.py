"""Example implementation of Mahler & Yum (2024).

This model implements the life cycle model from the paper 'Lifestyle Behaviors
and Wealth-Health Gaps in Germany' by Lukas Mahler and Minchul Yum (2024,
https://doi.org/10.3982/ECTA20603).
"""

from dataclasses import dataclass, make_dataclass

import jax.numpy as jnp
import numpy as np

import lcm
from lcm import DiscreteGrid, LinspaceGrid
from lcm.model import Model
from lcm.regime import Regime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    Float1D,
    FloatND,
    Int1D,
    Period,
    RegimeName,
)

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn: float = 57706.57
theta_val: Float1D = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
n: int = 38
retirement_age: int = 19
taul: float = 0.128
lamda: float = 1.0 - 0.321
rho: float = 0.975
r: float = 1.04**2.0
tt0: float = 0.115
winit: Float1D = jnp.array([43978, 48201])
avrgearn = avrgearn / winit[1]
mincon0: float = 0.10
mincon = mincon0 * avrgearn


def calc_savingsgrid(x: float):
    x = ((jnp.log(10.0**2) - jnp.log(10.0**0)) / 49) * x
    x = jnp.exp(x)
    xgrid = x - 10.0 ** (0.0)
    xgrid = xgrid / (10.0**2 - 10.0**0.0)
    return xgrid * (30 - 0) + 0


# --------------------------------------------------------------------------------------
# Categorical variables
# --------------------------------------------------------------------------------------
@dataclass
class WorkingStatus:
    retired: int = 0
    part: int = 1
    full: int = 2


@dataclass
class EducationStatus:
    low: int = 0
    high: int = 1


AdjustmentCost = make_dataclass(
    "AdjustmentCost", [("class" + str(i), int, int(i)) for i in range(5)]
)
Effort = make_dataclass(
    "HealthEffort", [("class" + str(i), int, int(i)) for i in range(40)]
)


@dataclass
class DiscountFactor:
    low: int = 0
    high: int = 1


@dataclass
class HealthStatus:
    bad: int = 0
    good: int = 1


@dataclass
class ProductivityType:
    low: int = 0
    high: int = 1


@dataclass
class HealthType:
    low: int = 0
    high: int = 1


@dataclass
class ProductivityShock:
    val0: int = 0
    val1: int = 1
    val2: int = 2
    val3: int = 3
    val4: int = 4


@dataclass
class Dead:
    dead: int = 0
    # ----------------------------------------------------------------------------------
    # Grid Creation
    # ----------------------------------------------------------------------------------


eff_grid: Int1D = jnp.linspace(0, 1, 40)


# ======================================================================================
# Model functions
# ======================================================================================
# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    period: Period,
    wealth: ContinuousState,  # noqa: ARG001
    health_type: HealthType,  # noqa: ARG001
    education: EducationStatus,  # noqa: ARG001
    adj_cost: int,
    fcost: float,
    disutil: float,
    cons_util: float,
    discount_factor: DiscountFactor,
    beta_mean: float,
    beta_std: float,
) -> float:
    beta = beta_mean + jnp.where(discount_factor, beta_std, -beta_std)
    f = cons_util - disutil - fcost - adj_cost
    return f * (beta**period)


def disutil(
    working: WorkingStatus,
    health: HealthStatus,
    education: EducationStatus,
    period: Period,
    phigrid: FloatND,
) -> float:
    return phigrid[period, education, health] * ((working / 2) ** (2)) / 2


def adj_cost(
    period: Period,
    adjustment_cost: int,
    effort: float,
    effort_t_1: float,
    chimaxgrid: Float1D,
) -> float:
    return jnp.where(
        jnp.logical_not(effort == effort_t_1),
        adjustment_cost * (chimaxgrid[period] / 4),
        0,
    )


def cnow(net_income: float, wealth: ContinuousState, saving: ContinuousAction) -> float:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return jnp.maximum(net_income + (wealth) * r - (saving), mincon)


def cons_util(
    health: HealthStatus, cnow: float, kappa: float, sigma: float, bb: float
) -> float:
    mucon = jnp.where(health, 1, kappa)
    return mucon * (((cnow) ** (1.0 - sigma)) / (1.0 - sigma)) + mucon * bb


def fcost(
    period: Period,
    education: EducationStatus,
    health: HealthStatus,
    effort: int,
    psi: float,
    xigrid: FloatND,
) -> float:
    return (
        xigrid[period, education, health]
        * (eff_grid[effort] ** (1 + (1 / psi)))
        / (1 + (1 / psi))
    )


# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(benefits: float, taxed_income: float, pension: float) -> float:
    return taxed_income + pension + benefits


def income(
    working: WorkingStatus,
    period: Period,
    health: HealthStatus,
    education: EducationStatus,
    productivity: ProductivityType,
    productivity_shock: ProductivityShock,
    xvalues: Float1D,
    income_grid: FloatND,
) -> float:
    return (
        income_grid[period, health, education]
        * (working / 2)
        * theta_val[productivity]
        * jnp.exp(xvalues[productivity_shock])
    )


def taxed_income(income: float) -> float:
    return lamda * (income ** (1.0 - taul)) * (avrgearn**taul)


def benefits(period: Period, health: HealthStatus, working: WorkingStatus) -> float:
    eligible = jnp.logical_and(health == 0, working == 0)
    return jnp.where(
        jnp.logical_and(eligible, period <= retirement_age), tt0 * avrgearn, 0
    )


def pension(
    period: Period,
    education: EducationStatus,
    productivity: ProductivityType,
    income_grid: FloatND,
    penre: float,
) -> float:
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


def next_discount_factor(discount_factor: DiscountFactor) -> DiscountFactor:
    return discount_factor


@lcm.mark.stochastic
def next_health(
    period: Period,
    health: HealthStatus,
    effort: int,
    effort_t_1: int,
    education: EducationStatus,
    health_type: HealthType,
):
    pass


def next_productivity(productivity: ProductivityType) -> ProductivityType:
    return productivity


def next_health_type(health_type: HealthType) -> HealthType:
    return health_type


def next_effort_t_1(effort: int) -> int:
    return effort


def next_education(education: EducationStatus) -> EducationStatus:
    return education


@lcm.mark.stochastic
def next_adjustment_cost(adjustment_cost: int):
    pass


@lcm.mark.stochastic
def next_productivity_shock(productivity_shock: ProductivityShock):
    pass


# --------------------------------------------------------------------------------------
# Regime Transitions
# --------------------------------------------------------------------------------------

surv_hs: FloatND = jnp.array(np.loadtxt("data/surv_hs.txt"))
surv_cl: FloatND = jnp.array(np.loadtxt("data/surv_cl.txt"))
spgrid: FloatND = jnp.zeros((38, 2, 2))
spgrid = spgrid.at[:, 0, 0].set(surv_hs[:, 1])
spgrid = spgrid.at[:, 1, 0].set(surv_cl[:, 1])
spgrid = spgrid.at[:, 0, 1].set(surv_hs[:, 0])
spgrid = spgrid.at[:, 1, 1].set(surv_cl[:, 0])


def alive_to_dead(
    period: Period, education: EducationStatus, health: HealthStatus
) -> dict[RegimeName, float]:
    return {
        "alive": spgrid[period, education, health],
        "dead": 1 - spgrid[period, education, health],
    }


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def retirement_constraint(period: Period, working: WorkingStatus) -> bool:
    return jnp.logical_not(jnp.logical_and(period > retirement_age, working > 0))


def savings_constraint(
    net_income: float, wealth: ContinuousState, saving: ContinuousAction
) -> bool:
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return net_income + (wealth) * r >= (saving)


# ======================================================================================
# Model specification and parameters
# ======================================================================================


ALIVE_REGIME = Regime(
    name="alive",
    utility=utility,
    functions={
        "disutil": disutil,
        "fcost": fcost,
        "cons_util": cons_util,
        "cnow": cnow,
        "income": income,
        "benefits": benefits,
        "adj_cost": adj_cost,
        "net_income": net_income,
        "taxed_income": taxed_income,
        "pension": pension,
        "retirement_constraint": retirement_constraint,
        "savings_constraint": savings_constraint,
    },
    actions={
        "working": DiscreteGrid(WorkingStatus),
        "saving": LinspaceGrid(start=0, stop=49, n_points=50),
        "effort": DiscreteGrid(Effort),
    },
    states={
        "wealth": LinspaceGrid(start=0, stop=49, n_points=50),
        "health": DiscreteGrid(HealthStatus),
        "productivity_shock": DiscreteGrid(ProductivityShock),
        "effort_t_1": DiscreteGrid(Effort),
        "adjustment_cost": DiscreteGrid(AdjustmentCost),
        "education": DiscreteGrid(EducationStatus),
        "discount_factor": DiscreteGrid(DiscountFactor),
        "productivity": DiscreteGrid(ProductivityType),
        "health_type": DiscreteGrid(HealthType),
    },
    constraints={
        "retirement_constraint": retirement_constraint,
        "savings_constraint": savings_constraint,
    },
    transitions={
        "alive": {
            "next_wealth": next_wealth,
            "next_health": next_health,
            "next_productivity_shock": next_productivity_shock,
            "next_discount_factor": next_discount_factor,
            "next_adjustment_cost": next_adjustment_cost,
            "next_effort_t_1": next_effort_t_1,
            "next_health_type": next_health_type,
            "next_education": next_education,
            "next_productivity": next_productivity,
        },
        "dead": {"next_dead": lambda: Dead.dead},
    },
    regime_transition_probs=alive_to_dead,
)

DEAD_REGIME = Regime(
    name="dead",
    utility=lambda dead: jnp.asarray([0.0]),  # noqa: ARG005
    states={"dead": DiscreteGrid(Dead)},
    actions={},
    transitions={"dead": {"next_dead": lambda dead: Dead.dead}},  # noqa: ARG005
    regime_transition_probs=lambda: {"alive": 0.0, "dead": 1.0},
)

MAHLER_YUM_MODEL = Model([ALIVE_REGIME, DEAD_REGIME], n_periods=n)


########################
# Mahler & Yum Params  #
########################
winit: Float1D = jnp.array([43978, 48201])

nuh_1: float = 2.63390750888379
nuh_2: float = 1.66602983591164
nuh_3: float = 1.27839561280412
nuh_4: float = 1.71439043350863

# unhealthy
nuu_1: float = 2.41177758126754
nuu_2: float = 1.8133670880598
nuu_3: float = 1.39103558901915
nuu_4: float = 2.41466980231321

nuad: float = 0.807247922589072
nuh: Float1D = jnp.array([nuh_1, nuh_2, nuh_3, nuh_4])
nuu: Float1D = jnp.array([nuu_1, nuu_2, nuu_3, nuu_4])
nu: list[Float1D] = [nuu, nuh]
# direct utility cost of effort
# hs-Healthy
xihsh_1: float = 0.146075197675677
xihsh_2: float = 0.55992411008533
xihsh_3: float = 1.04795036000287
xihsh_4: float = 1.60294886005945


# hs-Unhealthy
xihsu_1: float = 0.628031290227532
xihsu_2: float = 1.36593242946612
xihsu_3: float = 1.64963812690034
xihsu_4: float = 0.734873142494319


# cl-Healthy
xiclh_1: float = 0.091312997289004
xiclh_2: float = 0.302477689083851
xiclh_3: float = 0.739843441095022
xiclh_4: float = 1.36582077051777


# cl-Unhealthy
xiclu_1: float = 0.46921037985024
xiclu_2: float = 0.996665589702672
xiclu_3: float = 1.65388250352532
xiclu_4: float = 1.08866246911941

xi_hsh: Float1D = jnp.array([xihsh_1, xihsh_2, xihsh_3, xihsh_4])
xi_hsu: Float1D = jnp.array([xihsu_1, xihsu_2, xihsu_3, xihsu_4])
xi_clu: Float1D = jnp.array([xiclu_1, xiclu_2, xiclu_3, xiclu_4])
xi_clh: Float1D = jnp.array([xiclh_1, xiclh_2, xiclh_3, xiclh_4])

xi: list[Float1D] = [[xi_hsu, xi_hsh], [xi_clu, xi_clh]]

beta_mean: float = 0.942749393405227
beta_std: float = 0.0283688760224992

# effort habit adjustment cost max
chi_1: float = 0.000120437772838191
chi_2: float = 0.14468204213946

sigx: float = 0.0289408524185787

penre: float = 0.358766004066242


bb: float = 13.1079320277342

conp: float = 0.871503495423925
psi: float = 1.11497911620865

# Wage profile for hs + healthy
yths_s: float = 0.0615804210614531
yths_sq: float = -0.00250769285750586

# Wage profile for cl + healthy
ytcl_s: float = 0.0874283672769353
ytcl_sq: float = -0.00293713499239749

# wage penalty: depends on education and age
wagep_hs: float = 0.17769766414897
wagep_cl: float = 0.144836058314823


# Initial yt(1) for hs relative to cl
y1_hs: float = 0.899399488241831
y1_cl: float = 1.1654726432446

sigma: float = 2.0

START_PARAMS = {
    "nuh_1": nuh_1,
    "nuh_2": nuh_2,
    "nuh_3": nuh_3,
    "nuh_4": nuh_4,
    "nuu_1": nuu_1,
    "nuu_2": nuu_2,
    "nuu_3": nuu_3,
    "nuu_4": nuu_4,
    "nuad": nuad,
    "xihsh_1": xihsh_1,
    "xihsh_2": xihsh_2,
    "xihsh_3": xihsh_3,
    "xihsh_4": xihsh_4,
    "xihsu_1": xihsu_1,
    "xihsu_2": xihsu_2,
    "xihsu_3": xihsu_3,
    "xihsu_4": xihsu_4,
    "xiclu_1": xiclu_1,
    "xiclu_2": xiclu_2,
    "xiclu_3": xiclu_3,
    "xiclu_4": xiclu_4,
    "xiclh_1": xiclh_1,
    "xiclh_2": xiclh_2,
    "xiclh_3": xiclh_3,
    "xiclh_4": xiclh_4,
    "y1_hs": y1_hs,
    "yths_s": yths_s,
    "yths_sq": yths_sq,
    "wagep_hs": wagep_hs,
    "y1_cl": y1_cl,
    "ytcl_s": ytcl_s,
    "ytcl_sq": ytcl_sq,
    "wagep_cl": wagep_cl,
    "sigx": sigx,
    "chi_1": chi_1,
    "chi_2": chi_2,
    "psi": psi,
    "bb": 11,
    "conp": conp,
    "penre": penre,
    "beta_mean": beta_mean,
    "beta_std": beta_std,
}
