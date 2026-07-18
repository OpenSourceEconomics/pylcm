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


_AR1_RHO = 0.6


def _ar1_model(sigma_low: float, sigma_high: float) -> Model:
    """The alive/dead model with a state-conditioned Tauchen AR(1) income process."""
    income = TauchenAR1Process(
        n_points=15,
        gauss_hermite=False,
        rho=_AR1_RHO,
        sigma=max(sigma_low, sigma_high),  # fixed common grid = widest regime
        mu=0.0,
        n_std=4.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": sigma_low, "high": sigma_high}
        ),
    )
    alive = Regime(
        active=lambda age: age <= 60,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=40.0, n_points=8),
            "income": income,
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
        transition=None, active=lambda age: age > 60, functions={"utility": lambda: 0.0}
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=70, step="10Y"),
        fixed_params={"final_age_alive": 60},
    )


def test_ar1_simulated_innovation_std_follows_the_conditioned_sigma():
    """Round-2 F4: the AR(1) *draw* wrapper must use the current regime's sigma.

    Every other simulate test rides the IID family, so the F1 repair to the Tauchen
    branch (`_create_ar1_next_func` threads the conditioning state into `draw_shock`)
    was source-verified but never exercised end to end. For an AR(1) the level variance
    mixes rho and sigma; the innovation is what the conditioned sigma scales, so the
    discriminating statistic is the residual `income_t - rho*income_{t-1}` (the
    reviewer's own suggestion). `uncertainty` is absorbing here, so each agent's
    residuals are a clean draw from one regime's innovation law.
    """
    sigma_low, sigma_high, n = 0.05, 0.30, 6000
    half = n // 2
    result = (
        _ar1_model(sigma_low, sigma_high)
        .simulate(
            params=_params(),
            log_level="warning",
            initial_conditions={
                "wealth": jnp.full(n, 20.0),
                "income": jnp.zeros(n),
                "uncertainty": jnp.array(
                    [Uncertainty.low] * half + [Uncertainty.high] * half
                ),
                "age": jnp.full(n, 20.0),
                "regime_id": jnp.array([RegimeId.alive] * n),
            },
            period_to_regime_to_V_arr=None,
            seed=7,
        )
        .to_dataframe()
    )
    panel = result.sort_values(["subject_id", "period"]).copy()
    panel["inc_lag"] = panel.groupby("subject_id")["income"].shift(1)
    panel["resid"] = panel["income"] - _AR1_RHO * panel["inc_lag"]
    drawn = panel[
        (panel["age"] > 20) & panel["income"].notna() & panel["inc_lag"].notna()
    ]
    low = drawn[drawn["uncertainty"] == "low"]["resid"].to_numpy()
    high = drawn[drawn["uncertainty"] == "high"]["resid"].to_numpy()
    assert low.size > 100
    assert high.size > 100
    # The F1 defect drew both regimes at the common grid sigma (0.30), so the low
    # residual std collapsed to ~0.30 rather than 0.05.
    assert low.std() == pytest.approx(sigma_low, rel=0.15)
    assert high.std() == pytest.approx(sigma_high, rel=0.15)


def next_uncertainty_switching(age: int, uncertainty: DiscreteState) -> FloatND:
    """Switch low -> high once `age` reaches 30; `high` is absorbing (so NOT static)."""
    to_high = jnp.asarray((age >= 30) | (uncertainty == Uncertainty.high))[..., None]
    return jnp.where(to_high, jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0]))


@functools.cache
def _get_switching_model(sigma_low: float, sigma_high: float) -> Model:
    """The `_get_model` twin whose `uncertainty` MOVES over the life cycle."""
    final_age_alive = 50
    alive = Regime(
        active=lambda age, n=final_age_alive: age <= n,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": _income_process(sigma_low, sigma_high),
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty_switching),
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
        ages=AgeGrid(start=20, stop=60, step="10Y"),
        fixed_params={"final_age_alive": final_age_alive},
    )


def test_simulated_sigma_tracks_the_conditioning_state_THROUGH_TIME():
    """The gather must read the CURRENT period's `uncertainty`, not a frozen one.

    Every other simulate test here rides `next_uncertainty`, which is ABSORBING: each
    agent keeps its regime for life, so `uncertainty_t == uncertainty_{t+1}` and the
    draw's *timing* is unobservable. Gathering sigma at the current state, at the
    initial state, or one period stale all produce identical output — so none of those
    tests can tell a correctly-timed gather from a frozen one.

    Here `uncertainty` MOVES: all agents start `low` and switch to `high` at age 30.
    `income_{t+1}` is drawn from the state at `t`, so with ages 20/30/40/50 the law is

        income@30 ~ sigma(u@20=low)    income@40 ~ sigma(u@30=low)
        income@50 ~ sigma(u@40=high)

    Pairing each draw with the LAGGED `uncertainty` therefore separates the two sigmas,
    and it fails under both realistic mis-wirings: a gather frozen at the initial state
    leaves income@50 at sigma_low, while an off-by-one (contemporaneous) gather draws
    income@40 at sigma_high and inflates the low group.
    """
    sigma_low, sigma_high, n = 0.05, 0.30, 4000
    result = (
        _get_switching_model(sigma_low, sigma_high)
        .simulate(
            params=_params(),
            log_level="warning",
            initial_conditions={
                "wealth": jnp.full(n, 10.0),
                "income": jnp.zeros(n),
                "uncertainty": jnp.array([Uncertainty.low] * n),
                "age": jnp.full(n, 20.0),
                "regime_id": jnp.array([RegimeId.alive] * n),
            },
            period_to_regime_to_V_arr=None,
            seed=20260717,
        )
        .to_dataframe()
    )

    panel = result.sort_values(["subject_id", "period"]).copy()
    # The sigma that drew `income_t` is the one keyed by `uncertainty_{t-1}`.
    panel["sigma_state"] = panel.groupby("subject_id")["uncertainty"].shift(1)
    drawn = panel[(panel["age"] > 20) & panel["income"].notna()]

    # Pin the switch actually happened, so a degenerate panel cannot pass this test.
    assert set(drawn["sigma_state"].dropna()) == {"low", "high"}

    low = drawn[drawn["sigma_state"] == "low"]["income"].to_numpy()
    high = drawn[drawn["sigma_state"] == "high"]["income"].to_numpy()
    assert low.size > 100
    assert high.size > 100
    assert low.std() == pytest.approx(sigma_low, rel=0.15)
    assert high.std() == pytest.approx(sigma_high, rel=0.15)


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


def next_uncertainty_age_only(age: int) -> FloatND:
    """`uncertainty` driven by age alone, so `state_conditioned` is its ONLY reader."""
    return jnp.where(
        jnp.asarray(age >= 30)[..., None],
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 0.0]),
    )


def _alive_regime_without_local_uncertainty() -> Regime:
    """The alive regime with NO `uncertainty` of its own (it arrives model-level)."""
    return Regime(
        active=lambda age: age <= 50,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": _income_process(0.05, 0.30),
        },
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )


def _dead_regime() -> Regime:
    return Regime(
        transition=None, active=lambda age: age > 50, functions={"utility": lambda: 0.0}
    )


def test_model_level_conditioning_state_survives_pruning():
    """Round-2 F1: a broadcast conditioning state must not be pruned before use.

    `state_conditioned.on` is grid METADATA, not a user function, so the broadcast
    pruner's callable-DAG ancestry could not see it. A model-level `uncertainty` that
    reaches nothing else was therefore pruned, and the conditioned builder then rejected
    the model with the actively misleading "must name a DiscreteGrid state in the same
    regime as the process" — for a state the user *had* declared.

    It must also stay pruned where nothing reads it, so this pins both directions.
    """
    model = Model(
        regimes={
            "alive": _alive_regime_without_local_uncertainty(),
            "dead": _dead_regime(),
        },
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=60, step="10Y"),
        fixed_params={"final_age_alive": 50},
        states={"uncertainty": DiscreteGrid(Uncertainty)},
        state_transitions={"uncertainty": MarkovTransition(next_uncertainty)},
    )
    assert (
        "uncertainty" not in model.pruned_variables["alive"]
    )  # kept: the process reads it
    assert "uncertainty" in model.pruned_variables["dead"]  # pruned: nothing reads it


def test_conditioning_only_state_is_not_reported_unused():
    """Round-2: the usage validator shared the pruner's blind spot.

    A state read ONLY through `state_conditioned` was rejected as "defined but never
    used", even though the generated weights function takes it as an argument. Every
    other test masked this by having the transition read `uncertainty` for an unrelated
    reason; here the law depends on `age` alone. This is the natural use case: sigma
    conditioned on an exogenous volatility regime.
    """
    alive = Regime(
        active=lambda age: age <= 50,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=20.0, n_points=6),
            "income": _income_process(0.05, 0.30),
            "uncertainty": DiscreteGrid(Uncertainty),
        },
        state_transitions={
            "wealth": next_wealth,
            "uncertainty": MarkovTransition(next_uncertainty_age_only),
        },
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_regime,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    model = Model(
        regimes={"alive": alive, "dead": _dead_regime()},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=20, stop=60, step="10Y"),
        fixed_params={"final_age_alive": 50},
    )
    assert model is not None


def _conditioned_income(**kwargs) -> NormalIIDProcess:
    """A conditioned IID income process with the grid geometry under test."""
    return NormalIIDProcess(
        gauss_hermite=False,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": 0.1, "high": 0.3}
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        # `sigma`/`n_std` are NOT required to be positive by the process itself, so a
        # sign slip silently reverses the axis rather than failing.
        pytest.param(
            {"n_points": 5, "sigma": -0.3, "n_std": 3.0}, id="descending-neg-sigma"
        ),
        pytest.param(
            {"n_points": 5, "sigma": 0.3, "n_std": -3.0}, id="descending-neg-n_std"
        ),
        pytest.param(
            {"n_points": 5, "sigma": 0.0, "n_std": 3.0}, id="collapsed-sigma-0"
        ),
        pytest.param(
            {"n_points": 5, "sigma": 0.3, "n_std": 0.0}, id="collapsed-n_std-0"
        ),
    ],
)
def test_non_increasing_node_axis_rejected(kwargs):
    """Round-2 F2: a finite axis is not necessarily a *usable* one.

    The row bins on node midpoints, so only a strictly increasing axis makes the CDF
    differences probabilities. A DESCENDING axis is the dangerous case: measured on the
    real builder, nodes [3, 1.5, 0, -1.5, -3] give the row

        [1.0, -0.00620967, -0.98758066, -0.00620967, 1.0]

    which **sums to exactly 1.0** — so a row-sum or normalization check passes — while
    carrying negative mass, and a payoff of 1 at the zero node flips from +0.9876 to
    -0.9876, reversing a comparison against a sure 0.5. All four configurations here are
    reachable from the public constructor; before this guard every one was accepted.
    """
    income = _conditioned_income(mu=0.0, **kwargs)
    with pytest.raises(ModelInitializationError, match="strictly increasing"):
        _model_with_income(income).solve(params=_params(), log_level="warning")


def test_singleton_node_axis_rejected():
    """Round-2 F2: one node leaves no midpoint edges, so the row assembler returns an
    EMPTY (shape `(0,)`) vector rather than the only coherent one-state row `[1.0]`."""
    income = _conditioned_income(n_points=1, mu=0.0, sigma=0.3, n_std=3.0)
    with pytest.raises(ModelInitializationError, match="at least 2 nodes"):
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


# --- Cross-regime conditioning (round-3 review F1) ------------------------------- #
#
# A conditioned process in a TARGET regime has its transition weight built into every
# SOURCE regime that can reach it, evaluated at the source's own `on` state. So the
# conditioner is a dependency of the source too — invisible to the broadcast pruner and
# the usage validator, which walk only the callable DAG and only local processes. Before
# the fix a source regime carrying the conditioner (used solely via that cross-regime
# transition) was pruned / reported unused, and the model failed to build or solve.


@categorical(ordered=False)
class Phase:
    young: ScalarInt
    old: ScalarInt
    gone: ScalarInt


def next_phase(age: int) -> ScalarInt:
    return jnp.where(
        age >= 40, jnp.where(age >= 60, Phase.gone, Phase.old), Phase.young
    )


def next_uncertainty_phase_age(age: int) -> FloatND:
    """Age-only law: `uncertainty` is used ONLY through the conditioned process."""
    return jnp.where(
        jnp.asarray(age >= 30)[..., None],
        jnp.array([0.0, 1.0]),
        jnp.array([1.0, 0.0]),
    )


def next_uncertainty_phase_absorbing(uncertainty: DiscreteState) -> FloatND:
    return jnp.where(
        uncertainty == Uncertainty.low,
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
    )


def _uncond_income(n_points: int = 5) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=n_points, gauss_hermite=False, mu=0.0, sigma=0.30, n_std=4.0
    )


def _cond_income(
    sigma_low: float, sigma_high: float, n_points: int = 5
) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=n_points,
        gauss_hermite=False,
        mu=0.0,
        sigma=max(sigma_low, sigma_high),
        n_std=4.0,
        state_conditioned=StateConditioned(
            on="uncertainty", by={"low": sigma_low, "high": sigma_high}
        ),
    )


def _cross_regime_alive(active, income_proc, uncertainty_law, *, local_uncertainty):
    """An alive regime with income `income_proc`; `uncertainty` local or model-level."""
    states = {
        "wealth": LinSpacedGrid(start=1.0, stop=30.0, n_points=6),
        "income": income_proc,
    }
    transitions = {"wealth": next_wealth}
    if local_uncertainty:
        states["uncertainty"] = DiscreteGrid(Uncertainty)
        transitions["uncertainty"] = MarkovTransition(uncertainty_law)
    return Regime(
        active=active,
        states=states,
        state_transitions=transitions,
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=7)},
        transition=next_phase,
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )


def _cross_params():
    return {
        "discount_factor": 0.95,
        "young": {"next_wealth": {"interest_rate": 0.03}},
        "old": {"next_wealth": {"interest_rate": 0.03}},
    }


def test_cross_regime_regime_local_conditioner_builds_and_solves():
    """F1 (usage validator): source income UNconditioned, target income conditioned, and
    `uncertainty` (regime-local, age-only law) used only via the target's process.

    Before the fix the usage validator reported young's `uncertainty` as "defined but
    never used" — it is a real input to `weight_old__next_income` built into young's Q.
    """
    young = _cross_regime_alive(
        lambda age: age <= 40,
        _uncond_income(),
        next_uncertainty_phase_age,
        local_uncertainty=True,
    )
    old = _cross_regime_alive(
        lambda age: (age > 40) & (age <= 60),
        _cond_income(0.05, 0.30),
        next_uncertainty_phase_age,
        local_uncertainty=True,
    )
    gone = Regime(
        transition=None, active=lambda age: age > 60, functions={"utility": lambda: 0.0}
    )
    model = Model(
        regimes={"young": young, "old": old, "gone": gone},
        regime_id_class=Phase,
        ages=AgeGrid(start=20, stop=70, step="10Y"),
        fixed_params={},
    )
    V = model.solve(params=_cross_params(), log_level="warning")
    for leaf in jax.tree_util.tree_leaves(V):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_cross_regime_model_level_conditioner_survives_pruning():
    """F1 (broadcast pruner): the conditioner is a MODEL-LEVEL state used only via a
    reachable target's process.

    The pruner must keep it in every source reaching the process (young, old) and may
    prune it from the terminal `gone`. Before the fix it was pruned from young and the
    solve then failed for a missing DAG argument.
    """
    young = _cross_regime_alive(
        lambda age: age <= 40,
        _uncond_income(),
        next_uncertainty_phase_age,
        local_uncertainty=False,
    )
    old = _cross_regime_alive(
        lambda age: (age > 40) & (age <= 60),
        _cond_income(0.05, 0.30),
        next_uncertainty_phase_age,
        local_uncertainty=False,
    )
    gone = Regime(
        transition=None, active=lambda age: age > 60, functions={"utility": lambda: 0.0}
    )
    model = Model(
        regimes={"young": young, "old": old, "gone": gone},
        regime_id_class=Phase,
        ages=AgeGrid(start=20, stop=70, step="10Y"),
        fixed_params={},
        states={"uncertainty": DiscreteGrid(Uncertainty)},
        state_transitions={"uncertainty": MarkovTransition(next_uncertainty_phase_age)},
    )
    assert "uncertainty" not in model.pruned_variables["young"]  # source: reaches old
    assert "uncertainty" not in model.pruned_variables["old"]  # carries the process
    assert "uncertainty" in model.pruned_variables["gone"]  # terminal: reaches nothing
    V = model.solve(params=_cross_params(), log_level="warning")
    for leaf in jax.tree_util.tree_leaves(V):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_cross_regime_conditioned_draw_uses_the_source_sigma():
    """The cross-regime draw must gather the SOURCE's conditioning code, not a default.

    Build+solve+finite-V does not prove the conditioning is *correct*. Here income is
    conditioned only in `old`; `uncertainty` is absorbing so each agent keeps its start
    value. The income drawn on entry to `old` is transitioned using old's conditioned
    spec at the current (source) uncertainty, so its spread must follow sigma_low /
    sigma_high — otherwise the cross-regime gather silently used one common sigma.
    """
    sigma_low, sigma_high, n = 0.05, 0.30, 6000
    half = n // 2
    young = _cross_regime_alive(
        lambda age: age <= 40,
        _uncond_income(9),
        next_uncertainty_phase_absorbing,
        local_uncertainty=True,
    )
    old = _cross_regime_alive(
        lambda age: (age > 40) & (age <= 60),
        _cond_income(sigma_low, sigma_high, 9),
        next_uncertainty_phase_absorbing,
        local_uncertainty=True,
    )
    gone = Regime(
        transition=None, active=lambda age: age > 60, functions={"utility": lambda: 0.0}
    )
    model = Model(
        regimes={"young": young, "old": old, "gone": gone},
        regime_id_class=Phase,
        ages=AgeGrid(start=20, stop=70, step="10Y"),
        fixed_params={},
    )
    result = model.simulate(
        params=_cross_params(),
        log_level="warning",
        initial_conditions={
            "wealth": jnp.full(n, 15.0),
            "income": jnp.zeros(n),
            "uncertainty": jnp.array(
                [Uncertainty.low] * half + [Uncertainty.high] * half
            ),
            "age": jnp.full(n, 20.0),
            "regime_id": jnp.array([Phase.young] * n),
        },
        period_to_regime_to_V_arr=None,
        seed=11,
    ).to_dataframe()
    drawn = result[(result["regime_name"] == "old") & result["income"].notna()]
    low = drawn[drawn["uncertainty"] == "low"]["income"].to_numpy()
    high = drawn[drawn["uncertainty"] == "high"]["income"].to_numpy()
    assert low.size > 100
    assert high.size > 100
    assert low.std() == pytest.approx(sigma_low, rel=0.15)
    assert high.std() == pytest.approx(sigma_high, rel=0.15)
