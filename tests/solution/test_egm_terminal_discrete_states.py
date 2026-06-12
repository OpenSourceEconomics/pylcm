"""DC-EGM with a terminal target that shares a fixed discrete state.

A terminal regime's bequest utility may be heterogeneous in a discrete state
that the DC-EGM parent also carries. The motivating case is a fixed preference
type `pref_type`: the parent carries it as one of its own discrete combo axes,
the terminal `dead` regime reads it to scale the bequest, and the transition
into `dead` is the identity (`next_pref_type = pref_type`). At a parent combo
with `pref_type = k`, the terminal carry the parent needs is the terminal
utility evaluated at `pref_type = k` on the wealth grid — one row per shared
discrete combo.

The DC-EGM solution must match a mathematically identical dense brute-force
spec per type: a wrong per-type carry alignment (reading type 0's bequest for
type 1) breaks the comparison because the two types' bequests differ
materially.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

N_PERIODS = 4
# The lowest wealth nodes are where grid search is least reliable (the value
# function curves hardest as the borrowing constraint starts to bind) and the
# DC-EGM constrained segment is exact; they are excluded from the head-to-head
# so the comparison lives on territory where both solvers are well-defined.
N_BRUTE_UNSTABLE_NODES = 20

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=400.0, n_points=120)
CONSUMPTION_GRID = LinSpacedGrid(start=1.0, stop=400.0, n_points=500)
# Dense terminal wealth grid: the bequest continuation is read by linear
# interpolation on this grid, so it must resolve the curvature of `log`.
BEQUEST_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=400.0, n_points=600)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(400.0 * (i / 199) ** 3 for i in range(200)))


@categorical(ordered=False)
class PrefType:
    impatient_bequest: ScalarInt
    strong_bequest: ScalarInt


@categorical(ordered=False)
class RegimeId:
    retirement: ScalarInt
    dead: ScalarInt


def bequest_weight(
    pref_type: DiscreteState, weight_low: float, weight_high: float
) -> FloatND:
    """Per-type bequest weight; the two types differ enough to catch misalignment."""
    return jnp.where(pref_type == PrefType.strong_bequest, weight_high, weight_low)


def bequest_utility(
    wealth: ContinuousState,
    pref_type: DiscreteState,
    weight_low: float,
    weight_high: float,
    bequest_shifter: float,
) -> FloatND:
    weight = bequest_weight(
        pref_type=pref_type, weight_low=weight_low, weight_high=weight_high
    )
    return weight * jnp.log(wealth + bequest_shifter)


def utility_retirement(
    consumption: ContinuousAction, pref_type: DiscreteState, flow_taste: float
) -> FloatND:
    """Log consumption plus a per-type additive flow taste.

    The additive `pref_type` term makes the preference type a genuine combo
    axis of the parent's own utility (a model state must be used by the
    regime that carries it) without touching marginal utility, so the Euler
    inversion and the brute-force oracle stay identical.
    """
    taste = jnp.where(pref_type == PrefType.strong_bequest, flow_taste, 0.0)
    return jnp.log(consumption) + taste


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings_post(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def next_wealth_from_savings(
    savings_post: FloatND, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * savings_post


def next_wealth_brute(
    wealth: ContinuousState, consumption: ContinuousAction, interest_rate: float
) -> ContinuousState:
    return (1.0 + interest_rate) * (wealth - consumption)


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_regime_from_retirement(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.retirement,
    )


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _make_dead_regime() -> UserRegime:
    return UserRegime(
        transition=None,
        states={
            "wealth": BEQUEST_WEALTH_GRID,
            "pref_type": DiscreteGrid(PrefType),
        },
        functions={
            "utility": bequest_utility,
            "bequest_weight": bequest_weight,
        },
    )


@functools.cache
def _get_dcegm_model() -> Model:
    """Retirement DC-EGM model with a fixed `pref_type` shared with `dead`."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    solver = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings_post",
        savings_grid=SAVINGS_GRID,
        n_constrained_points=64,
    )
    retirement = UserRegime(
        transition=next_regime_from_retirement,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "pref_type": DiscreteGrid(PrefType)},
        state_transitions={
            "wealth": next_wealth_from_savings,
            "pref_type": fixed_transition("pref_type"),
        },
        functions={
            "utility": utility_retirement,
            "resources": resources,
            "savings_post": savings_post,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=solver,
        active=lambda age, la=last_age: age < la,
    )
    return Model(
        regimes={"retirement": retirement, "dead": _make_dead_regime()},
        ages=ages,
        regime_id_class=RegimeId,
    )


@functools.cache
def _get_brute_model() -> Model:
    """Mathematically identical brute-force spec sharing `pref_type` with `dead`."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    retirement = UserRegime(
        transition=next_regime_from_retirement,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "pref_type": DiscreteGrid(PrefType)},
        state_transitions={
            "wealth": next_wealth_brute,
            "pref_type": fixed_transition("pref_type"),
        },
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={"utility": utility_retirement},
        active=lambda age, la=last_age: age < la,
    )
    return Model(
        regimes={"retirement": retirement, "dead": _make_dead_regime()},
        ages=ages,
        regime_id_class=RegimeId,
    )


def _get_params() -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    return {
        "discount_factor": 0.98,
        "interest_rate": 0.0,
        "final_age_alive": final_age_alive,
        "weight_low": 0.3,
        "weight_high": 3.0,
        "bequest_shifter": 5.0,
        "flow_taste": 0.5,
    }


def test_dcegm_terminal_bequest_by_pref_type_matches_brute_force():
    """DC-EGM matches a dense brute-force spec at every working-life period.

    Each `pref_type` slice of the published value-function array agrees with
    the brute-force solution on the wealth nodes where grid search is stable,
    so the parent reads the terminal bequest carry at its own type.
    """
    params = _get_params()
    dcegm_solution = _get_dcegm_model().solve(params=params, log_level="debug")
    brute_solution = _get_brute_model().solve(params=params, log_level="debug")

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["retirement"])
        dcegm_V = np.asarray(dcegm_solution[period]["retirement"])
        assert brute_V.shape == dcegm_V.shape
        np.testing.assert_allclose(
            dcegm_V[..., N_BRUTE_UNSTABLE_NODES:],
            brute_V[..., N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def test_dcegm_terminal_bequest_differs_by_pref_type():
    """The two `pref_type` slices carry materially different value functions.

    Guards the comparison: if the per-type bequests were identical, a swapped
    carry alignment would pass silently. The strong-bequest type values wealth
    in the terminal period far above the impatient-bequest type.
    """
    params = _get_params()
    dcegm_solution = _get_dcegm_model().solve(params=params, log_level="debug")

    decision_period = sorted(dcegm_solution)[0]
    V = np.asarray(dcegm_solution[decision_period]["retirement"])
    impatient = V[PrefType.impatient_bequest]
    strong = V[PrefType.strong_bequest]
    assert np.all(strong[N_BRUTE_UNSTABLE_NODES:] > impatient[N_BRUTE_UNSTABLE_NODES:])


def test_terminal_discrete_state_not_carried_by_parent_is_rejected():
    """A terminal discrete state absent from the parent's combo axes is rejected.

    The parent has no axis to align the terminal carry's discrete dimension to,
    so the DC-EGM solve reports the unsupported configuration rather than
    silently mis-indexing.
    """
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    solver = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings_post",
        savings_grid=SAVINGS_GRID,
        n_constrained_points=64,
    )

    def plain_utility(consumption: ContinuousAction) -> FloatND:
        return jnp.log(consumption)

    # The parent does NOT carry `pref_type`; only `dead` does.
    retirement = UserRegime(
        transition=next_regime_from_retirement,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={"wealth": next_wealth_from_savings},
        functions={
            "utility": plain_utility,
            "resources": resources,
            "savings_post": savings_post,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=solver,
        active=lambda age, la=last_age: age < la,
    )
    model = Model(
        regimes={"retirement": retirement, "dead": _make_dead_regime()},
        ages=ages,
        regime_id_class=RegimeId,
    )
    params = {k: v for k, v in _get_params().items() if k != "flow_taste"}
    with pytest.raises(NotImplementedError, match="pref_type"):
        model.solve(params=params, log_level="debug")
