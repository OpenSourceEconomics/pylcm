"""End-to-end: a minimal 2-regime-uncertainty model with state-conditioned income sigma.

The acceptance test for the DAG wiring: a precautionary-savings model whose IID income
shock has a `sigma` conditioned on a discrete `uncertainty` state must build and solve,
its value must depend on `uncertainty` iff the two regime sigmas differ, and the
unsupported families (Gauss-Hermite IID, Rouwenhorst) must be rejected at construction.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.processes.base import StateConditioned
from _lcm.regime_building.processing import (
    _validate_conditioning_codes_agree_across_regimes,
)
from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Uncertainty:
    low: ScalarInt
    high: ScalarInt


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
    interest_rate: float,
) -> FloatND:
    return (1 + interest_rate) * (wealth - consumption) + jnp.exp(income)


def next_uncertainty(uncertainty: DiscreteState) -> FloatND:
    """Absorbing: each uncertainty regime stays put."""
    return jnp.where(
        uncertainty == Uncertainty.low,
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
    )


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, RegimeId.dead, RegimeId.alive)


def wealth_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _income_process(sigma_low: float, sigma_high: float, n_points: int = 5):
    grid_sigma = max(sigma_low, sigma_high)  # fixed common grid = widest regime
    return NormalIIDProcess(
        n_points=n_points,
        gauss_hermite=False,
        mu=0.0,
        sigma=grid_sigma,
        n_std=3.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": sigma_low, "high": sigma_high}
        ),
    )


@functools.cache
def _get_model(sigma_low: float, sigma_high: float, n_periods: int = 5) -> Model:
    final_age_alive = 20 + (n_periods - 2) * 10
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": _income_process(sigma_low, sigma_high),
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=final_age_alive: age > n,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=20 + (n_periods - 1) * 10, step="10Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def _params():
    return {
        "discount_factor": 0.95,
        "alive": {"next_wealth": {"interest_rate": 0.03}},
    }


def _solve(sigma_low, sigma_high):
    return _get_model(sigma_low, sigma_high).solve(
        params=_params(), log_level="warning"
    )


def _uncertainty_axis_maxdiff(V) -> float:
    """Max |V(...,uncertainty=low) - V(...,uncertainty=high)| over the alive-regime
    value leaves (the only state axis of size 2 is `uncertainty`)."""
    m = 0.0
    for leaf in jax.tree_util.tree_leaves(V):
        a = np.asarray(leaf)
        if a.ndim >= 1 and 2 in a.shape:
            ax = list(a.shape).index(2)
            m = max(m, float(np.abs(np.take(a, 0, ax) - np.take(a, 1, ax)).max()))
    return m


def test_conditioned_model_solves():
    """The milestone: the state-conditioned model builds and solves, V finite."""
    V = _solve(0.05, 0.30)
    assert V is not None
    for leaf in jax.tree_util.tree_leaves(V):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_equal_sigma_makes_uncertainty_irrelevant():
    """Degeneracy: if both regimes share sigma, the conditioning collapses and the value
    is *exactly* independent of the uncertainty state."""
    assert _uncertainty_axis_maxdiff(_solve(0.30, 0.30)) == 0.0


def test_higher_uncertainty_changes_value():
    """Conditioning is live: distinct per-regime sigmas make the value depend on the
    uncertainty state (the precautionary response flows through the transition CDF)."""
    assert _uncertainty_axis_maxdiff(_solve(0.02, 0.60)) > 1e-3


def _simulate_income_by_uncertainty(sigma_low: float, sigma_high: float, n: int = 4000):
    """Simulate `n` agents, half starting `low`, half `high`; return income by regime.

    `uncertainty` is absorbing, so each half stays in its regime for the whole path and
    the simulated `income` column is a clean draw from that regime's law.
    """
    model = _get_model(sigma_low, sigma_high)
    half = n // 2
    result = model.simulate(
        params=_params(),
        log_level="warning",
        initial_conditions={
            "wealth": jnp.full(n, 10.0),
            "income": jnp.zeros(n),
            "uncertainty": jnp.array(
                [Uncertainty.low] * half + [Uncertainty.high] * half
            ),
            "age": jnp.full(n, 20.0),
            "regime_id": jnp.array([RegimeId.alive] * n),
        },
        period_to_regime_to_V_arr=None,
        seed=1234,
    ).to_dataframe()
    # Age 20 income is the initial condition, and dead-regime rows are NaN; keep only
    # the drawn alive ones. `uncertainty` comes back as its category label.
    drawn = result[(result["age"] > 20) & result["income"].notna()]
    low = drawn[drawn["uncertainty"] == "low"]["income"].to_numpy()
    high = drawn[drawn["uncertainty"] == "high"]["income"].to_numpy()
    assert low.size > 100, "no low-regime draws to test"
    assert high.size > 100, "no high-regime draws to test"
    return low, high


def test_simulated_shock_variance_follows_the_conditioned_sigma():
    """F1 regression: the SIMULATED draw must use the current regime's sigma.

    The original wiring gathered sigma for the solve rows but let `draw_shock` keep the
    scalar common-grid sigma, so both regimes simulated at `max(sigma_low, sigma_high)`
    — solving one DGP and simulating another. Every solve-only test passed regardless.
    """
    sigma_low, sigma_high = 0.05, 0.30
    low, high = _simulate_income_by_uncertainty(sigma_low, sigma_high)
    assert low.std() == pytest.approx(sigma_low, rel=0.15)
    assert high.std() == pytest.approx(sigma_high, rel=0.15)
    # The defect made this ratio 1.0 (both drawn at the common grid sigma of 0.30).
    assert low.std() / high.std() == pytest.approx(sigma_low / sigma_high, rel=0.2)


def test_equal_sigma_simulates_one_law():
    """Degeneracy in simulation: equal sigmas leave both regimes the same spread."""
    low, high = _simulate_income_by_uncertainty(0.25, 0.25)
    assert low.std() == pytest.approx(high.std(), rel=0.15)


def _model_with_income(income_proc) -> Model:
    """Build the same alive/dead model with a swapped-in income process."""
    alive = Regime(
        active=lambda age: age <= 21,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": income_proc,
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = Regime(
        transition=None, active=lambda age: age > 21, functions={"utility": lambda: 0.0}
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=40, step="10Y"),
        fixed_params={"final_age_alive": 21},
    )


def test_gauss_hermite_state_conditioned_rejected():
    """GH + StateConditioned must raise: its nodes scale with sigma (audit F3)."""
    income = NormalIIDProcess(
        n_points=5,
        gauss_hermite=True,
        mu=0.0,
        sigma=0.3,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="Gauss-Hermite"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


def test_rouwenhorst_state_conditioned_rejected():
    """Rouwenhorst + StateConditioned must raise (rho-only transition, audit F2)."""
    income = RouwenhorstAR1Process(
        n_points=5,
        rho=0.9,
        sigma=0.3,
        mu=0.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="only supported for CDF-binned"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


def test_unknown_conditioning_state_rejected():
    """`on` naming a non-existent (or non-discrete) state raises a clear error."""
    income = NormalIIDProcess(
        n_points=5,
        gauss_hermite=False,
        mu=0.0,
        sigma=0.3,
        n_std=3.0,
        state_conditioned=StateConditioned(on="nope", by={"low": 0.1, "high": 0.3}),
    )
    with pytest.raises(ModelInitializationError, match="must name a DiscreteGrid"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


@categorical(ordered=True)
class UncertaintyReversed:
    """Same two categories as `Uncertainty`, opposite code order."""

    high: ScalarInt
    low: ScalarInt


def test_cross_regime_code_map_mismatch_rejected():
    """F4: the sigma array is ordered by the TARGET grid but indexed by the SOURCE code.

    Those agree only if every regime maps the conditioning categories to the same
    integer codes. A regime that reverses them would silently swap the two volatilities,
    so v1 requires one shared map.
    """
    all_grids = {
        "alive": {"uncertainty": DiscreteGrid(Uncertainty)},
        "retired": {"uncertainty": DiscreteGrid(UncertaintyReversed)},
    }
    with pytest.raises(ModelInitializationError, match="map categories to the same"):
        _validate_conditioning_codes_agree_across_regimes(
            on="uncertainty", all_grids=all_grids
        )


def test_matching_code_maps_accepted():
    """The same categorical in every regime is the normal, allowed case."""
    all_grids = {
        "alive": {"uncertainty": DiscreteGrid(Uncertainty)},
        "retired": {"uncertainty": DiscreteGrid(Uncertainty)},
        "dead": {},  # regimes without the conditioning state are simply skipped
    }
    _validate_conditioning_codes_agree_across_regimes(
        on="uncertainty", all_grids=all_grids
    )


def test_gauss_hermite_tauchen_state_conditioned_rejected():
    """F6: the GH rejection is blanket, as documented — Tauchen included, not just IID.

    GH nodes are placed from the *grid* sigma, so a conditioned sigma would bin one law
    on a quadrature rule chosen for another.
    """
    income = TauchenAR1Process(
        n_points=5,
        gauss_hermite=True,
        rho=0.9,
        sigma=0.3,
        mu=0.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="Gauss-Hermite"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


def test_runtime_grid_param_rejected():
    """F3: a grid parameter left for runtime is never bound on the conditioned branch.

    That branch is chosen *before* the runtime-parameter mechanism, so `get_gridpoints`
    would return all-NaN and the closure would capture those nodes forever. Reject.
    """
    income = NormalIIDProcess(
        n_points=5,
        gauss_hermite=False,
        mu=0.0,
        sigma=None,  # would be supplied at runtime
        n_std=3.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="every grid parameter fixed"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_sigma_rejected(bad):
    """F3: NaN/inf sail through a bare `v <= 0` test and poison every row silently."""
    income = NormalIIDProcess(
        n_points=5,
        gauss_hermite=False,
        mu=0.0,
        sigma=0.3,
        n_std=3.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": bad, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="finite positive sigmas"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


def test_nonpositive_sigma_rejected():
    """Non-positive per-regime sigma raises at construction."""
    income = NormalIIDProcess(
        n_points=5,
        gauss_hermite=False,
        mu=0.0,
        sigma=0.3,
        n_std=3.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.0, "high": 0.3}
        ),
    )
    with pytest.raises(ModelInitializationError, match="finite positive sigmas"):
        _model_with_income(income).solve(params=_params(), log_level="warning")
