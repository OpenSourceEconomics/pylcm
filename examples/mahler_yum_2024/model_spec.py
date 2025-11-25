"""Example implementation of Mahler & Yum (2024).

This model replicates the lifecycle model from the paper "Lifestyle Behaviors and
Wealth-Health Gaps in Germany" by Lukas Mahler and Minchul Yum (Econometrica, 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.interpolate import interp1d

import lcm
from lcm import DiscreteGrid, LinspaceGrid, Model, Regime
from lcm.dispatchers import _base_productmap

if TYPE_CHECKING:
    from lcm.typing import (
        Any,
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
n: int = 38
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

# --------------------------------------------------------------------------------------
# Health Techonology Parameters
# --------------------------------------------------------------------------------------

const_healthtr: float = -0.906
age_const: Float1D = jnp.asarray(
    [0.0, -0.289, -0.644, -0.881, -1.138, -1.586, -1.586, -1.586]
)
eff_param: Float1D = jnp.asarray([0.693, 0.734])
eff_sq: float = 0
healthy_dummy: float = 2.311
htype_dummy: float = 0.632
college_dummy: float = 0.238


def calc_savingsgrid(x: Float1D) -> Float1D:
    x = ((jnp.log(10.0**2) - jnp.log(10.0**0)) / 49) * x
    x = jnp.exp(x)
    xgrid = x - 10.0 ** (0.0)
    xgrid = xgrid / (10.0**2 - 10.0**0.0)
    return xgrid * (30 - 0) + 0


# ======================================================================================
# Discrete Variables
# ======================================================================================


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


# --------------------------------------------------------------------------------------
# Grid Creation
# --------------------------------------------------------------------------------------


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
    health_type: DiscreteState,  # noqa: ARG001
    education: DiscreteState,  # noqa: ARG001
    adj_cost: FloatND,
    fcost: FloatND,
    disutil: FloatND,
    cons_util: FloatND,
    discount_factor: DiscreteState,
    beta_mean: float,
    beta_std: float,
) -> FloatND:
    beta = beta_mean + jnp.where(discount_factor, beta_std, -beta_std)
    f = cons_util - disutil - fcost - adj_cost
    return f * (beta**period)


def disutil(
    working: DiscreteAction,
    health: DiscreteState,
    education: DiscreteState,
    period: Period,
    phigrid: FloatND,
) -> FloatND:
    return phigrid[period, education, health] * ((working / 2) ** (2)) / 2


def adj_cost(
    period: Period,
    adjustment_cost: DiscreteState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    chimaxgrid: Float1D,
) -> FloatND:
    return jnp.where(
        jnp.logical_not(effort == effort_t_1),
        adjustment_cost * (chimaxgrid[period] / 4),
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


def income(
    working: DiscreteAction,
    period: Period,
    health: DiscreteState,
    education: DiscreteState,
    productivity: DiscreteState,
    productivity_shock: DiscreteState,
    xvalues: Float1D,
    income_grid: FloatND,
) -> FloatND:
    return (
        income_grid[period, health, education]
        * (working / 2)
        * theta_val[productivity]
        * jnp.exp(xvalues[productivity_shock])
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


def next_discount_factor(discount_factor: DiscreteState) -> DiscreteState:
    return discount_factor


@lcm.mark.stochastic
def next_health(  # type: ignore[empty-body]
    period: Period,
    health: DiscreteState,
    effort: DiscreteAction,
    effort_t_1: DiscreteState,
    education: DiscreteState,
    health_type: DiscreteState,
) -> DiscreteState:
    pass


def next_productivity(productivity: DiscreteState) -> DiscreteState:
    return productivity


def next_health_type(health_type: DiscreteState) -> DiscreteState:
    return health_type


def next_effort_t_1(effort: DiscreteAction) -> DiscreteState:
    return effort


def next_education(education: DiscreteState) -> DiscreteState:
    return education


@lcm.mark.stochastic
def next_adjustment_cost(adjustment_cost: DiscreteState) -> DiscreteState:  # type: ignore[empty-body]
    pass


@lcm.mark.stochastic
def next_productivity_shock(productivity_shock: DiscreteState) -> DiscreteState:  # type: ignore[empty-body]
    pass


# --------------------------------------------------------------------------------------
# Regime Transitions
# --------------------------------------------------------------------------------------
surv_hs: FloatND = jnp.array(np.loadtxt("data/surv_hs.txt", skiprows=1))
surv_cl: FloatND = jnp.array(np.loadtxt("data/surv_cl.txt", skiprows=1))
spgrid: FloatND = jnp.zeros((38, 2, 2))
spgrid = spgrid.at[:, 0, 0].set(surv_hs[:, 1])
spgrid = spgrid.at[:, 1, 0].set(surv_cl[:, 1])
spgrid = spgrid.at[:, 0, 1].set(surv_hs[:, 0])
spgrid = spgrid.at[:, 1, 1].set(surv_cl[:, 0])


def alive_to_dead(
    period: Period, education: DiscreteState, health: DiscreteState
) -> dict[str, Float1D]:
    return {
        "alive": spgrid[period, education, health],
        "dead": 1 - spgrid[period, education, health],
    }


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
    regime_transition_probs=alive_to_dead,  # type: ignore[arg-type]
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


# ======================================================================================
# Mahler & Yum starting params
# ======================================================================================

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

xi: list[list[Float1D]] = [[xi_hsu, xi_hsh], [xi_clu, xi_clh]]

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

# ======================================================================================
# Model Input Construction
# ======================================================================================


def create_phigrid(nu: list[Float1D], nu_e: float) -> FloatND:
    phi_interp_values = jnp.array([1, 8, 13, 20])
    phigrid = jnp.zeros((retirement_age + 1, 2, 2))
    for i in range(2):
        for j in range(2):
            interp_points = jnp.arange(1, retirement_age + 2)
            spline = interp1d(
                np.asarray(phi_interp_values), np.asarray(nu[j]), kind="cubic"
            )
            temp_grid = jnp.asarray(spline(interp_points))
            temp_grid = jnp.where(i == 0, temp_grid * jnp.exp(nu_e), temp_grid)
            phigrid = phigrid.at[:, i, j].set(temp_grid)
    return phigrid


def create_xigrid(xi: list[list[Float1D]]) -> FloatND:
    xi_interp_values = jnp.array([1, 12, 20, 31])
    xigrid = jnp.zeros((n, 2, 2))
    for i in range(2):
        for j in range(2):
            interp_points = np.arange(1, 31)
            spline = interp1d(
                np.asarray(xi_interp_values), np.asarray(xi[i][j]), kind="cubic"
            )
            temp_grid = jnp.asarray(spline(interp_points))
            xigrid = xigrid.at[0:30, i, j].set(temp_grid)
            xigrid = xigrid.at[30:n, i, j].set(xi[i][j][3])
    return xigrid


def create_chimaxgrid(chi_1: float, chi_2: float) -> Float1D:
    t = jnp.arange(38)
    return jnp.maximum(chi_1 * jnp.exp(chi_2 * t), 0)


def create_income_grid(
    y1_hs: float,
    y1_cl: float,
    yths_s: float,
    yths_sq: float,
    wagep_hs: float,
    wagep_cl: float,
    ytcl_s: float,
    ytcl_sq: float,
    sigx: float,
) -> FloatND:
    sdztemp = ((sigx**2.0) / (1.0 - rho**2.0)) ** 0.5
    j = jnp.arange(20)
    health = jnp.arange(2)
    education = jnp.arange(2)

    def calc_base(period: Period, health: Int1D, education: Int1D) -> Float1D:
        yt = jnp.where(
            education == 1,
            (y1_cl * jnp.exp(ytcl_s * (period) + ytcl_sq * (period) ** 2.0))
            * (1.0 - wagep_cl * (1 - health)),
            (y1_hs * jnp.exp(yths_s * (period) + yths_sq * (period) ** 2.0))
            * (1.0 - wagep_hs * (1 - health)),
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


def rouwenhorst(rho: float, sigma_eps: Float1D, n: int) -> tuple[FloatND, FloatND]:
    mu_eps = 0

    q = (rho + 1) / 2
    nu = jnp.sqrt((n - 1) / (1 - rho**2)) * sigma_eps
    P = jnp.zeros((n, n))

    P = P.at[0, 0].set(q)
    P = P.at[0, 1].set(1 - q)
    P = P.at[1, 0].set(1 - q)
    P = P.at[1, 1].set(q)

    for i in range(2, n):
        P11 = jnp.zeros((i + 1, i + 1))
        P12 = jnp.zeros((i + 1, i + 1))
        P21 = jnp.zeros((i + 1, i + 1))
        P22 = jnp.zeros((i + 1, i + 1))

        P11 = P11.at[0:i, 0:i].set(P[0:i, 0:i])
        P12 = P12.at[0:i, 1 : i + 1].set(P[0:i, 0:i])
        P21 = P21.at[1 : i + 1, 0:i].set(P[0:i, 0:i])
        P22 = P22.at[1 : i + 1, 1 : i + 1].set(P[0:i, 0:i])

        P = P.at[0 : i + 1, 0 : i + 1].set(
            q * P11 + (1 - q) * P12 + (1 - q) * P21 + q * P22
        )
        P = P.at[1:i, :].set(P[1:i, :] / 2)
    return jnp.linspace(mu_eps / (1.0 - rho) - nu, mu_eps / (1.0 - rho) + nu, n), P.T


def create_inputs(
    seed: int,
    n_simulation_subjects: int,
    nuh_1: float,
    nuh_2: float,
    nuh_3: float,
    nuh_4: float,
    nuu_1: float,
    nuu_2: float,
    nuu_3: float,
    nuu_4: float,
    xihsh_1: float,
    xihsh_2: float,
    xihsh_3: float,
    xihsh_4: float,
    xihsu_1: float,
    xihsu_2: float,
    xihsu_3: float,
    xihsu_4: float,
    xiclu_1: float,
    xiclu_2: float,
    xiclu_3: float,
    xiclu_4: float,
    xiclh_1: float,
    xiclh_2: float,
    xiclh_3: float,
    xiclh_4: float,
    y1_hs: float,
    y1_cl: float,
    yths_s: float,
    yths_sq: float,
    wagep_hs: float,
    wagep_cl: float,
    ytcl_s: float,
    ytcl_sq: float,
    sigx: float,
    chi_1: float,
    chi_2: float,
    psi: float,
    nuad: float,
    bb: float,
    conp: float,
    penre: float,
    beta_mean: float,
    beta_std: float,
) -> tuple[dict[RegimeName, Any], dict[RegimeName, Any], list[RegimeName]]:
    # Gather parameters
    nuh: Float1D = jnp.array([nuh_1, nuh_2, nuh_3, nuh_4])
    nuu: Float1D = jnp.array([nuu_1, nuu_2, nuu_3, nuu_4])
    nu: list[Float1D] = [nuu, nuh]
    xi_hsh: Float1D = jnp.array([xihsh_1, xihsh_2, xihsh_3, xihsh_4])
    xi_hsu: Float1D = jnp.array([xihsu_1, xihsu_2, xihsu_3, xihsu_4])
    xi_clu: Float1D = jnp.array([xiclu_1, xiclu_2, xiclu_3, xiclu_4])
    xi_clh: Float1D = jnp.array([xiclh_1, xiclh_2, xiclh_3, xiclh_4])
    xi: list[list[Float1D]] = [[xi_hsu, xi_hsh], [xi_clu, xi_clh]]

    # Create variable grids from supplied parameters
    income_grid = create_income_grid(
        y1_hs, y1_cl, yths_s, yths_sq, wagep_hs, wagep_cl, ytcl_s, ytcl_sq, sigx
    )
    chimax_grid = create_chimaxgrid(chi_1, chi_2)
    xvalues, xtrans = rouwenhorst(rho, jnp.sqrt(sigx), 5)
    xi_grid = create_xigrid(xi)
    phi_grid = create_phigrid(nu, nuad)

    # Create parameters
    params = {
        "beta": 1,
        "disutil": {"phigrid": phi_grid},
        "fcost": {"psi": psi, "xigrid": xi_grid},
        "cons_util": {"sigma": sigma, "bb": bb, "kappa": conp},
        "utility": {"beta_mean": beta_mean, "beta_std": beta_std},
        "income": {"income_grid": income_grid, "xvalues": xvalues},
        "pension": {"income_grid": income_grid, "penre": penre},
        "adj_cost": {"chimaxgrid": chimax_grid},
        "shocks": {
            "alive__next_productivity_shock": xtrans.T,
            "alive__next_health": tr2yp_grid,
            "alive__next_adjustment_cost": jnp.full((5, 5), 1 / 5),
        },
    }

    # Create initial states for the simulation

    discount = jnp.zeros((16), dtype=jnp.int8)
    prod = jnp.zeros((16), dtype=jnp.int8)
    ht = jnp.zeros((16), dtype=jnp.int8)
    ed = jnp.zeros((16), dtype=jnp.int8)
    for i in range(1, 3):
        for j in range(1, 3):
            for k in range(1, 3):
                index = (i - 1) * 2 * 2 + (j - 1) * 2 + k - 1
                discount = discount.at[index].set(i - 1)
                prod = prod.at[index].set(j - 1)
                ht = ht.at[index].set(1 - (k - 1))
                discount = discount.at[index + 8].set(i - 1)
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
    initial_discount = discount[types]
    initial_effort = jnp.searchsorted(eff_grid, init_distr_2b2t2h[:, 2][types])
    initial_adjustment_cost = random.choice(
        new_keys[1], jnp.arange(5), (n_simulation_subjects,)
    )
    prod_dist = jax.lax.fori_loop(
        0,
        1000000,
        lambda i, a: a @ xtrans.T,  # noqa: ARG005
        jnp.full(5, 1 / 5),
    )
    initial_productivity_shock = random.choice(
        new_keys[2], jnp.arange(5), (n_simulation_subjects,), p=prod_dist
    )
    initial_states = {
        "alive": {
            "wealth": initial_wealth,
            "health": initial_health,
            "health_type": initial_health_type,
            "effort_t_1": initial_effort,
            "productivity_shock": initial_productivity_shock,
            "adjustment_cost": initial_adjustment_cost,
            "education": initial_education,
            "productivity": initial_productivity,
            "discount_factor": initial_discount,
        },
        "dead": {"dead": jnp.full(n_simulation_subjects, 0)},
    }
    initial_regimes = ["alive"] * n_simulation_subjects
    return params, initial_states, initial_regimes
