import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from Mahler_Yum_2024 import MAHLER_YUM_MODEL
from scipy.interpolate import interp1d

from lcm.dispatchers import _base_productmap
from lcm.typing import Any, Float1D, FloatND, Int1D, RegimeName

model = MAHLER_YUM_MODEL

# ======================================================================================
# Fixed Parameters
# ======================================================================================
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
sigma: float = 2

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

# ======================================================================================
# Grid Creation
# ======================================================================================

# --------------------------------------------------------------------------------------
# Dynamic Grid Creation Functions
# --------------------------------------------------------------------------------------


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

    def calc_base(_period: Int1D, health: Int1D, education: Int1D) -> Float1D:
        yt = jnp.where(
            education == 1,
            (y1_cl * jnp.exp(ytcl_s * (_period) + ytcl_sq * (_period) ** 2.0))
            * (1.0 - wagep_cl * (1 - health)),
            (y1_hs * jnp.exp(yths_s * (_period) + yths_sq * (_period) ** 2.0))
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
    period: Int1D, health: Int1D, eff: Int1D, eff_1: Int1D, edu: Int1D, ht: Int1D
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
    n: int,
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
    init_distr_2b2t2h = jnp.array(np.loadtxt("data/init_distr_2b2t2h.txt"))
    initial_dists = jnp.diff(init_distr_2b2t2h[:, 0], prepend=0)
    eff_grid = jnp.linspace(0, 1, 40)
    key = random.key(seed)
    initial_wealth = jnp.full((n), 0, dtype=jnp.int8)
    types = random.choice(key, jnp.arange(16), (n,), p=initial_dists)
    new_keys = random.split(key=key, num=3)
    health_draw = random.uniform(new_keys[0], (n,))
    health_thresholds = init_distr_2b2t2h[:, 1][types]
    initial_health = jnp.where(health_draw > health_thresholds, 0, 1)
    initial_health_type = 1 - ht[types]
    initial_education = ed[types]
    initial_productivity = prod[types]
    initial_discount = discount[types]
    initial_effort = jnp.searchsorted(eff_grid, init_distr_2b2t2h[:, 2][types])
    initial_adjustment_cost = random.choice(new_keys[1], jnp.arange(5), (n,))
    prod_dist = jax.lax.fori_loop(
        0,
        1000000,
        lambda i, a: a @ xtrans.T,  # noqa: ARG005
        jnp.full(5, 1 / 5),
    )
    initial_productivity_shock = random.choice(
        new_keys[2], jnp.arange(5), (n,), p=prod_dist
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
        "dead": {"dead": jnp.full(n, 0)},
    }
    initial_regimes = ["alive"] * n
    return params, initial_states, initial_regimes
