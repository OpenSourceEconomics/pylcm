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

# --------------------------------------------------------------------------------------
# Fixed Parameters
# --------------------------------------------------------------------------------------
avrgearn = 57706.57
theta_val = jnp.array([jnp.exp(-0.2898), jnp.exp(0.2898)])
n = 38
retirement_age = 19
taul = 0.128
lamda = 1.0 - 0.321
rho = 0.975
r = 1.04**2.0
tt0 = 0.115
winit = jnp.array([43978, 48201])
avrgearn = avrgearn / winit[1]
mincon0 = 0.10
mincon = mincon0 * avrgearn


def calc_savingsgrid(x):
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
class Health:
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


eff_grid = jnp.linspace(0, 1, 40)


# ======================================================================================
# Model functions
# ======================================================================================
# --------------------------------------------------------------------------------------
# Utility function
# --------------------------------------------------------------------------------------
def utility(
    period,
    wealth,  # noqa: ARG001
    health_type,  # noqa: ARG001
    education,  # noqa: ARG001
    adj_cost,
    fcost,
    disutil,
    cons_util,
    discount_factor,
    beta_mean,
    beta_std,
):
    beta = beta_mean + jnp.where(discount_factor, beta_std, -beta_std)
    f = cons_util - disutil - fcost - adj_cost
    return f * (beta**period)


def disutil(working, health, education, period, phigrid):
    return phigrid[period, education, health] * ((working / 2) ** (2)) / 2


def adj_cost(period, adjustment_cost, effort, effort_t_1, chimaxgrid):
    return jnp.where(
        jnp.logical_not(effort == effort_t_1),
        adjustment_cost * (chimaxgrid[period] / 4),
        0,
    )


def cnow(net_income, wealth, saving):
    wealth = calc_savingsgrid(wealth)
    saving = calc_savingsgrid(saving)
    return jnp.maximum(net_income + (wealth) * r - (saving), mincon)


def cons_util(health, cnow, kappa, sigma, bb):
    mucon = jnp.where(health, 1, kappa)
    return mucon * (((cnow) ** (1.0 - sigma)) / (1.0 - sigma)) + mucon * bb


def fcost(period, education, health, effort, psi, xigrid):
    return (
        xigrid[period, education, health]
        * (eff_grid[effort] ** (1 + (1 / psi)))
        / (1 + (1 / psi))
    )


# --------------------------------------------------------------------------------------
# Income Calculation
# --------------------------------------------------------------------------------------
def net_income(benefits, taxed_income, pension):
    return taxed_income + pension + benefits


def income(
    working,
    period,
    health,
    education,
    productivity,
    productivity_shock,
    xvalues,
    income_grid,
):
    return (
        income_grid[period, health, education]
        * (working / 2)
        * theta_val[productivity]
        * jnp.exp(xvalues[productivity_shock])
    )


def taxed_income(income):
    return lamda * (income ** (1.0 - taul)) * (avrgearn**taul)


def benefits(period, health, working):
    eligible = jnp.logical_and(health == 0, working == 0)
    return jnp.where(
        jnp.logical_and(eligible, period <= retirement_age), tt0 * avrgearn, 0
    )


def pension(period, education, productivity, income_grid, penre):
    return jnp.where(
        period > retirement_age,
        income_grid[19, 1, education] * theta_val[productivity] * penre,
        0,
    )


# --------------------------------------------------------------------------------------
# State transitions
# --------------------------------------------------------------------------------------
def next_wealth(saving):
    return saving


def next_discount_factor(discount_factor):
    return discount_factor


@lcm.mark.stochastic
def next_alive(alive, period, education, health):
    pass


@lcm.mark.stochastic
def next_health(period, health, effort, effort_t_1, education, health_type):
    pass


def next_productivity(productivity):
    return productivity


def next_health_type(health_type):
    return health_type


def next_effort_t_1(effort):
    return effort


def next_education(education):
    return education


@lcm.mark.stochastic
def next_adjustment_cost(adjustment_cost):
    pass


@lcm.mark.stochastic
def next_productivity_shock(productivity_shock):
    pass


# --------------------------------------------------------------------------------------
# Regime Transitions
# --------------------------------------------------------------------------------------

surv_hs = jnp.array(np.loadtxt("data/surv_hs.txt"))
surv_cl = jnp.array(np.loadtxt("data/surv_cl.txt"))
spgrid = jnp.zeros((38, 2, 2))
spgrid = spgrid.at[:, 0, 0].set(surv_hs[:, 1])
spgrid = spgrid.at[:, 1, 0].set(surv_cl[:, 1])
spgrid = spgrid.at[:, 0, 1].set(surv_hs[:, 0])
spgrid = spgrid.at[:, 1, 1].set(surv_cl[:, 0])


def alive_to_dead(period, education, health):
    return {
        "alive": spgrid[period, education, health],
        "dead": 1 - spgrid[period, education, health],
    }


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def retirement_constraint(period, working):
    return jnp.logical_not(jnp.logical_and(period > retirement_age, working > 0))


def savings_constraint(net_income, wealth, saving):
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
        "health": DiscreteGrid(Health),
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
winit = jnp.array([43978, 48201])

nuh_1 = 2.63390750888379
nuh_2 = 1.66602983591164
nuh_3 = 1.27839561280412
nuh_4 = 1.71439043350863

# unhealthy
nuu_1 = 2.41177758126754
nuu_2 = 1.8133670880598
nuu_3 = 1.39103558901915
nuu_4 = 2.41466980231321

nuad = 0.807247922589072
nuh = jnp.array([nuh_1, nuh_2, nuh_3, nuh_4])
nuu = jnp.array([nuu_1, nuu_2, nuu_3, nuu_4])
nu = [nuu, nuh]
# direct utility cost of effort
# hs-Healthy
xihsh_1 = 0.146075197675677
xihsh_2 = 0.55992411008533
xihsh_3 = 1.04795036000287
xihsh_4 = 1.60294886005945


# hs-Unhealthy
xihsu_1 = 0.628031290227532
xihsu_2 = 1.36593242946612
xihsu_3 = 1.64963812690034
xihsu_4 = 0.734873142494319


# cl-Healthy
xiclh_1 = 0.091312997289004
xiclh_2 = 0.302477689083851
xiclh_3 = 0.739843441095022
xiclh_4 = 1.36582077051777


# cl-Unhealthy
xiclu_1 = 0.46921037985024
xiclu_2 = 0.996665589702672
xiclu_3 = 1.65388250352532
xiclu_4 = 1.08866246911941

xi_hsh = jnp.array([xihsh_1, xihsh_2, xihsh_3, xihsh_4])
xi_hsu = jnp.array([xihsu_1, xihsu_2, xihsu_3, xihsu_4])
xi_clu = jnp.array([xiclu_1, xiclu_2, xiclu_3, xiclu_4])
xi_clh = jnp.array([xiclh_1, xiclh_2, xiclh_3, xiclh_4])

xi = [[xi_hsu, xi_hsh], [xi_clu, xi_clh]]

beta_mean = 0.942749393405227
beta_std = 0.0283688760224992

# effort habit adjustment cost max
chi_1 = 0.000120437772838191
chi_2 = 0.14468204213946

sigx = 0.0289408524185787

penre = 0.358766004066242


bb = 13.1079320277342

conp = 0.871503495423925
psi = 1.11497911620865

# Wage profile for hs + healthy
yths_s = 0.0615804210614531
yths_sq = -0.00250769285750586

# Wage profile for cl + healthy
ytcl_s = 0.0874283672769353
ytcl_sq = -0.00293713499239749

# wage penalty: depends on education and age
wagep_hs = 0.17769766414897
wagep_cl = 0.144836058314823


# Initial yt(1) for hs relative to cl
y1_hs = 0.899399488241831
y1_cl = 1.1654726432446

sigma = 2.0

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
