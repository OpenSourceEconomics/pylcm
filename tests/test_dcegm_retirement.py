"""Spec for full DC-EGM on the Iskhakov et al. (2017) retirement model.

Discrete retirement choice + continuous consumption, solved with DC-EGM (FUES
envelope) and compared against:

- the analytical worker/retired value functions shipped in
  `tests/data/analytical_solution/` (kinked case, no taste shocks), and
- the brute-force solver with regime-level taste shocks on the equivalent spec
  (smoothed case): both solvers approximate the same smoothed model, so their V
  arrays agree up to the consumption-grid resolution of the brute solution.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.config import TEST_DATA
from _lcm.typing import PeriodToRegimeToVArr
from lcm import AgeGrid, MarkovTransition, Model
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import FloatND
from tests.test_models.deterministic import base, dcegm_variants
from tests.test_models.deterministic.dcegm_variants import (
    get_full_model,
    get_full_params,
)

ANALYTICAL_CASES = {
    "iskhakov_2017_five_periods": {"n_periods": 6, "disutility_of_work": 1.0},
    "iskhakov_2017_low_delta": {"n_periods": 4, "disutility_of_work": 0.1},
}


def _load_analytical(case: str, kind: str) -> np.ndarray:
    return np.genfromtxt(
        TEST_DATA.joinpath("analytical_solution", f"{case}__values_{kind}.csv"),
        delimiter=",",
    )


def _stack_regime_V(
    period_to_regime_to_V_arr: PeriodToRegimeToVArr, regime: str
) -> np.ndarray:
    periods = sorted(period_to_regime_to_V_arr)[:-1]
    return np.stack([np.asarray(period_to_regime_to_V_arr[p][regime]) for p in periods])


@pytest.mark.parametrize(("case", "spec"), ANALYTICAL_CASES.items())
def test_dcegm_matches_analytical_solution(case, spec):
    """DC-EGM reproduces the analytical worker and retired value functions.

    Tighter than the brute-force tolerance: the secondary kinks from the
    retirement choice are exactly where the envelope step pays off.
    """
    model = get_full_model("dcegm", spec["n_periods"])
    params = get_full_params(
        spec["n_periods"],
        discount_factor=0.98,
        disutility_of_work=spec["disutility_of_work"],
        interest_rate=0.0,
        wage=20.0,
    )

    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug")

    for kind, regime in [("worker", "working_life"), ("retired", "retirement")]:
        numerical = _stack_regime_V(period_to_regime_to_V_arr, regime)
        analytical = _load_analytical(case, kind)
        # Elementwise — every (period, node) value must hit the analytical
        # solution; aggregating over periods could hide a localized error.
        np.testing.assert_allclose(numerical, analytical, atol=0.03, err_msg=f"{kind}")


def _retirement_stay_prob(age: float, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, 1.0)


def _retirement_death_prob(age: float, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 0.0)


def test_brute_force_regime_targeting_dcegm_regime_agrees_with_all_brute():
    """A brute-force worker targeting a DC-EGM retiree solves to brute values.

    Brute-force regimes may target DC-EGM regimes (they only read the
    target's V array); the DC-EGM retiree declares its reachable targets
    granularly so the brute worker is structurally unreachable from it. The
    mixed model's V differs from the all-brute model's only through the
    retiree continuation, on which the two solvers agree wherever the brute
    solver is reliable. At the lowest wealth nodes the brute leg leans on
    consumption below its grid start and coarse interpolation, so its values
    are the biased side there and the comparison starts above them.
    """
    n_periods = 4
    n_brute_unstable_nodes = 12
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])

    def active(age: float, la: float = last_age) -> bool:
        return age < la

    mixed = Model(
        regimes={
            "working_life": base.working_life.replace(active=active),
            "retirement": dcegm_variants.dcegm_retirement_full.replace(
                active=active,
                transition={
                    "retirement": MarkovTransition(_retirement_stay_prob),
                    "dead": MarkovTransition(_retirement_death_prob),
                },
            ),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )
    params = get_full_params(n_periods, discount_factor=0.98, wage=20.0)

    mixed_solution = mixed.solve(params=params, log_level="debug")
    brute_solution = get_full_model("brute_force", n_periods).solve(
        params=params, log_level="debug"
    )

    for period in sorted(brute_solution)[:-1]:
        for regime in ["working_life", "retirement"]:
            np.testing.assert_allclose(
                np.asarray(mixed_solution[period][regime])[n_brute_unstable_nodes:],
                np.asarray(brute_solution[period][regime])[n_brute_unstable_nodes:],
                atol=1e-2,
                rtol=1e-3,
                err_msg=f"period={period}, regime={regime}",
            )


def _smoothed_model_pair(n_periods: int, shocks) -> dict[str, Model]:
    """Equivalent-spec pair with EV1 taste shocks on the working regime."""
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])

    def active(age: float, la: float = last_age) -> bool:
        return age < la

    brute = Model(
        regimes={
            "working_life": base.working_life.replace(
                active=active, taste_shocks=shocks
            ),
            "retirement": base.retirement.replace(active=active),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )
    dcegm = Model(
        regimes={
            "working_life": dcegm_variants.dcegm_working_life.replace(
                active=active, taste_shocks=shocks
            ),
            "retirement": dcegm_variants.dcegm_retirement_full.replace(active=active),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )
    return {"brute_force": brute, "dcegm": dcegm}


def test_smoothed_model_brute_and_dcegm_agree():
    """With taste shocks, brute force and DC-EGM solve the same smoothed model.

    Brute computes `λ·logsumexp(Qc/λ)` on the consumption grid; DC-EGM computes
    the same object from exact Euler policies. Agreement is up to the brute
    solver's resolution, which fails at the lowest wealth nodes: there the
    brute solution leans on consumption choices below its grid start and on
    coarse interpolation where log utility curves hardest (the lowest
    retirement wealth nodes in the earliest periods, where the brute value
    falls well below the closed form while DC-EGM stays close to it).
    DC-EGM's exact-slope carry read does not mirror that low-wealth error, so
    the comparison covers the wealth nodes where the brute solver is reliable.
    """
    n_periods = 4
    scale = 0.2
    n_brute_unstable_nodes = 12
    params = get_full_params(n_periods, discount_factor=0.98, wage=20.0)
    params["working_life"]["taste_shocks"] = {"scale": scale}

    models = _smoothed_model_pair(n_periods, ExtremeValueTasteShocks())
    solutions = {
        solver: model.solve(params=params, log_level="debug")
        for solver, model in models.items()
    }

    for period in sorted(solutions["brute_force"])[:-1]:
        for regime in ["working_life", "retirement"]:
            brute_V = np.asarray(solutions["brute_force"][period][regime])
            dcegm_V = np.asarray(solutions["dcegm"][period][regime])
            np.testing.assert_allclose(
                dcegm_V[n_brute_unstable_nodes:],
                brute_V[n_brute_unstable_nodes:],
                atol=1e-2,
                rtol=1e-3,
            )
