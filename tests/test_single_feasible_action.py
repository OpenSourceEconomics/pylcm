"""Reproduce the aca-model NaN failure (issue OpenSourceEconomics/aca-model#9):
solve raises `InvalidValueFunctionError: ... regime 'dead': all values are NaN`.

Two hypotheses are exercised here:

1. **single/all infeasible action**: when constraints leave a single (or no)
   consumption gridpoint feasible. `max(Q, where=F, initial=-inf)` is supposed
   to mask infeasible Q values, but if Q itself contains NaN at infeasible
   cells (e.g. `log(0)` at `consumption=0`), the mask is enough — proven by
   the tests in this file. `-inf` from all-infeasible cells does not cascade
   to NaN by itself either.

2. **CRRA bequest with pref_type indexing under jnp.where**: the bequest
   function evaluates *both* branches of `jnp.where(jnp.isclose(gamma, 1),
   log_branch, power_branch)`. For a parameter set where gamma is exactly 1
   for one preference type but not the other, the power_branch divides by
   `1 - gamma = 0` for the gamma=1 type. NaN/Inf from the *unselected*
   branch leaks through `jnp.where`'s gradient/forward pass under JIT
   tracing in some configurations.

The model uses an `IrregSpacedGrid` consumption action with runtime-supplied
points to also exercise the `feature/runtime-action-grids` path (PR #338).
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.grids import IrregSpacedGrid
from lcm.grids.coordinates import get_irreg_coordinate
from lcm.regime_building.ndimage import map_coordinates
from lcm.typing import ContinuousAction, ContinuousState, FloatND


@categorical(ordered=False)
class RegimeId:
    alive: int
    dead: int


def _utility(consumption: ContinuousAction) -> FloatND:
    """CRRA-like utility. log requires consumption > 0."""
    return jnp.log(consumption)


def _next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
) -> ContinuousState:
    return wealth - consumption


def _borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _next_regime(age: int, last_alive_age: int) -> FloatND:
    """Deterministic regime transition: alive→alive while age<last, then alive→dead."""
    return jnp.where(age >= last_alive_age, RegimeId.dead, RegimeId.alive)


def _build_model(
    *,
    wealth_lo: float,
    wealth_hi: float,
    n_wealth: int,
    consumption_lo: float,
    consumption_hi: float,
    n_consumption: int,
    n_periods: int,
) -> tuple[Model, dict]:
    """Build a 2-regime (alive, dead) model with runtime consumption points.

    The wealth grid is fixed-LinSpaced (so changing it doesn't perturb the
    runtime-action-grid path). The consumption grid is the runtime-supplied
    IrregSpacedGrid.
    """
    last_alive_age = n_periods - 2  # alive at ages 0..n-2; dead at n-1
    alive = Regime(
        functions={"utility": _utility},
        states={
            "wealth": LinSpacedGrid(start=wealth_lo, stop=wealth_hi, n_points=n_wealth)
        },
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": IrregSpacedGrid(n_points=n_consumption)},
        constraints={"borrowing_constraint": _borrowing_constraint},
        transition=_next_regime,
        active=lambda age: age <= last_alive_age,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age > last_alive_age,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        regime_id_class=RegimeId,
    )
    consumption_points = jnp.linspace(consumption_lo, consumption_hi, n_consumption)
    params = {
        "discount_factor": 0.95,
        "alive": {
            "consumption": {"points": consumption_points},
            "next_regime": {"last_alive_age": last_alive_age},
        },
    }
    return model, params


def test_baseline_no_nan():
    """Healthy regime: at every wealth, every consumption point is feasible.

    consumption_lo == wealth_lo so the smallest action is always feasible at
    every wealth gridpoint. `consumption_hi <= wealth_lo` ensures even the
    largest consumption is feasible at the lowest wealth state.
    """
    model, params = _build_model(
        wealth_lo=10.0,
        wealth_hi=20.0,
        n_wealth=5,
        consumption_lo=1.0,
        consumption_hi=5.0,
        n_consumption=5,
        n_periods=3,
    )
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    for regime_to_V in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V.values():
            assert not jnp.any(jnp.isnan(V_arr))


def test_some_states_have_only_one_feasible_action():
    """At low-wealth states, only consumption[0] satisfies `consumption <= wealth`.

    consumption_lo < wealth_lo (so smallest action feasible at every wealth),
    but consumption[1:] > wealth_lo (so at the lowest wealth gridpoint only
    consumption[0] is feasible).
    """
    model, params = _build_model(
        wealth_lo=1.0,
        wealth_hi=20.0,
        n_wealth=5,
        consumption_lo=0.5,
        consumption_hi=5.0,
        n_consumption=5,
        n_periods=3,
    )
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    # At wealth = 1.0, only consumption = 0.5 is feasible (1.625, 2.75, 3.875,
    # 5.0 all > 1.0).
    for regime_to_V in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V.values():
            assert not jnp.any(jnp.isnan(V_arr)), (
                "single-feasible-action state should not produce NaN V"
            )


def test_some_states_have_no_feasible_action():
    """At sufficiently low wealth, *every* consumption gridpoint is infeasible.

    consumption_lo > wealth_lo means at the lowest wealth state no
    consumption point satisfies the borrowing constraint.
    """
    model, params = _build_model(
        wealth_lo=0.1,
        wealth_hi=20.0,
        n_wealth=5,
        consumption_lo=1.0,
        consumption_hi=5.0,
        n_consumption=5,
        n_periods=3,
    )
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    # At wealth = 0.1, no consumption point is feasible. max returns -inf.
    # The next period interpolates V over the resulting -inf cells.
    for regime_to_V in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V.values():
            assert not jnp.any(jnp.isnan(V_arr)), (
                "all-infeasible state should yield -inf, not NaN"
            )


def test_log_zero_consumption_propagates_nan_via_max_when_unconstrained():
    """U(c=0) = -inf is fine, but U evaluated at infeasible negative wealth
    via `next_wealth = wealth - c` going through interpolation in the next
    period should not pollute V."""
    # consumption_lo = 0 → log(0) = -inf at the smallest action, regardless
    # of feasibility. `where=F_arr` should mask this out of the max.
    model, params = _build_model(
        wealth_lo=1.0,
        wealth_hi=10.0,
        n_wealth=5,
        consumption_lo=0.0,
        consumption_hi=5.0,
        n_consumption=5,
        n_periods=3,
    )
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    for regime_to_V in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V.values():
            # log(0) = -inf is not NaN, but combined with where-mask edge
            # cases this could in principle leak; check it does not.
            assert not jnp.any(jnp.isnan(V_arr))


@pytest.mark.parametrize(
    ("wealth_lo", "consumption_lo", "label"),
    [
        (1.0, 0.5, "single-feasible"),
        (0.1, 1.0, "all-infeasible"),
    ],
)
def test_simulate_with_constrained_action_grid(wealth_lo, consumption_lo, label):
    """End-to-end solve+simulate for both regimes."""
    model, params = _build_model(
        wealth_lo=wealth_lo,
        wealth_hi=20.0,
        n_wealth=5,
        consumption_lo=consumption_lo,
        consumption_hi=5.0,
        n_consumption=5,
        n_periods=3,
    )
    initial_conditions = {
        "age": jnp.array([0.0, 0.0, 0.0]),
        "wealth": jnp.array([wealth_lo, 5.0, 20.0]),
        "regime": jnp.array(
            [RegimeId.alive, RegimeId.alive, RegimeId.alive], dtype=jnp.int32
        ),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        check_initial_conditions=False,
        log_level="off",
    )
    df = result.to_dataframe()
    assert not df["value"].isna().any(), (
        f"{label}: simulated value column should not contain NaN"
    )


# ---------------------------------------------------------------------------
# Replicas of the aca-baseline failure path: dead regime with a CRRA bequest
# whose `gamma` is per-pref_type, evaluated through `jnp.where`.
# ---------------------------------------------------------------------------


@categorical(ordered=False)
class PrefType:
    type_0: int
    type_1: int


@categorical(ordered=False)
class AliveDeadRegimeId:
    alive: int
    dead: int


def _crra_bequest(
    assets: ContinuousState,
    pref_type,
    bequest_shifter: float,
    consumption_weight,
    coefficient_rra,
) -> FloatND:
    """Replica of aca_model.agent.preferences.bequest, simplified.

    `consumption_weight` and `coefficient_rra` are FloatND indexed by
    `pref_type`. Both branches of the `jnp.where` are traced.
    """
    alpha = consumption_weight[pref_type]
    gamma = coefficient_rra[pref_type]
    assets_shifted = jnp.maximum(0.0, assets) + bequest_shifter
    one_minus_gamma = jnp.where(jnp.isclose(gamma, 1.0), 1.0, 1.0 - gamma)
    return jnp.where(
        jnp.isclose(gamma, 1.0),
        jnp.log(assets_shifted),
        assets_shifted ** (one_minus_gamma * alpha) / one_minus_gamma,
    )


def _alive_utility(
    consumption: ContinuousAction, pref_type, consumption_weight
) -> FloatND:
    """Make pref_type matter in alive's utility too (otherwise pylcm complains)."""
    alpha = consumption_weight[pref_type]
    return alpha * jnp.log(consumption)


def _next_assets(
    assets: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return assets - consumption


def _alive_borrow(consumption: ContinuousAction, assets: ContinuousState) -> FloatND:
    return consumption <= assets


def _alive_to_dead(age: int, last_alive_age: int) -> FloatND:
    """Deterministic regime transition; returns a scalar regime ID."""
    return jnp.where(
        age >= last_alive_age, AliveDeadRegimeId.dead, AliveDeadRegimeId.alive
    )


def _build_alive_dead_model(
    *,
    coefficient_rra: tuple[float, float],
    consumption_weight: tuple[float, float],
    n_periods: int = 3,
) -> tuple[Model, dict]:
    last_alive_age = n_periods - 2

    alive = Regime(
        functions={"utility": _alive_utility},
        states={
            "assets": LinSpacedGrid(start=1.0, stop=20.0, n_points=5),
            "pref_type": DiscreteGrid(PrefType, batch_size=1),
        },
        state_transitions={"assets": _next_assets, "pref_type": None},
        actions={"consumption": IrregSpacedGrid(n_points=5)},
        constraints={"borrowing_constraint": _alive_borrow},
        transition=_alive_to_dead,
        active=lambda age: age <= last_alive_age,
    )
    dead = Regime(
        transition=None,
        functions={"utility": _crra_bequest},
        states={
            "assets": LinSpacedGrid(start=1.0, stop=20.0, n_points=5),
            "pref_type": DiscreteGrid(PrefType, batch_size=1),
        },
        active=lambda _age: True,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=n_periods - 1, step="Y"),
        regime_id_class=AliveDeadRegimeId,
    )
    cw_arr = jnp.asarray(consumption_weight)
    params = {
        "discount_factor": 0.95,
        "alive": {
            "utility": {"consumption_weight": cw_arr},
            "consumption": {"points": jnp.linspace(0.5, 5.0, 5)},
            "next_regime": {"last_alive_age": last_alive_age},
        },
        "dead": {
            "utility": {
                "bequest_shifter": 100.0,
                "consumption_weight": cw_arr,
                "coefficient_rra": jnp.asarray(coefficient_rra),
            },
        },
    }
    return model, params


def test_bequest_gamma_close_to_one_is_safe():
    """gamma=0.999077 (the benchmark value for type_1) should not produce NaN."""
    model, params = _build_alive_dead_model(
        coefficient_rra=(3.84, 0.999077),
        consumption_weight=(0.68, 0.88),
    )
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    for regime_to_V in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V.values():
            assert not jnp.any(jnp.isnan(V_arr))


def test_bequest_gamma_exactly_one_for_one_type_only():
    """gamma=1.0 exactly for one pref type triggers `jnp.where(isclose, log, power)`.

    The unselected `power` branch divides by `1 - gamma = 0` for that type. JAX
    evaluates both branches of `jnp.where`; the non-finite from the unselected
    branch is masked, but `0/0` produces NaN that may not be masked correctly
    when the operand to `jnp.where` is itself NaN under XLA's nan-prop rules.
    """
    model, params = _build_alive_dead_model(
        coefficient_rra=(3.84, 1.0),  # type_1 hits the log branch
        consumption_weight=(0.68, 0.88),
    )
    period_to_regime_to_V_arr = model.solve(params=params, log_level="off")
    for period, regime_to_V in period_to_regime_to_V_arr.items():
        for regime, V_arr in regime_to_V.items():
            assert not jnp.any(jnp.isnan(V_arr)), (
                f"NaN in V[{regime}, period={period}] when one type has gamma=1.0"
            )


# ---------------------------------------------------------------------------
# Direct probe: `map_coordinates` produces NaN at ±inf / NaN coordinates.
# This is the concrete NaN source — `lower_weight = 1 - inf = -inf` and
# `upper_weight = inf` combined with positive grid values gives `inf - inf =
# NaN`. The aca-baseline NaN-in-V at age 51 is most plausibly traced back to
# *some* upstream computation (next_assets / next_aime, or a state coordinate
# from `get_irreg_coordinate` / `get_*_coordinate` that divides by zero on a
# degenerate grid segment) producing inf, which then poisons the value
# function via this interpolation path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_coord", [jnp.inf, -jnp.inf, jnp.nan])
def test_map_coordinates_returns_nan_for_non_finite_coordinate(bad_coord):
    """`map_coordinates` cannot recover from a non-finite continuous-state
    coordinate: the linear-interp weights become `inf` and `1 - inf = -inf`,
    and `inf * V[k] - inf * V[k-1]` reduces to NaN.

    Implication for callers: any path that can feed `inf` or `NaN` into the
    coordinate finder (e.g. division by zero in a state transition, an
    overflow when V values are O(1e8), or a `0/0` in a degenerate
    IrregSpacedGrid segment) will produce NaN in V.
    """
    V_arr = jnp.array([1.0, 5.0, 12.0])
    out = map_coordinates(V_arr, coordinates=[jnp.array(bad_coord)])
    assert jnp.isnan(out)


def test_irreg_coordinate_divides_by_zero_on_duplicate_grid_points():
    """`get_irreg_coordinate` divides by `upper_point - lower_point`. If a
    runtime-supplied points array contains duplicates, the divisor is 0.

    Reproduces a class of failures that is *only* possible under
    `feature/runtime-action-grids` / runtime-state-grids when the caller
    constructs the points from a parameter that can collapse (e.g.
    `geomspace(consumption_floor, MAX, n_points)` with `consumption_floor ==
    MAX`, or any param-driven `linspace` whose endpoints can coincide).
    """
    # Duplicate adjacent points where the query value equals the duplicate.
    # `searchsorted([0, 1, 1], 1.0, side='right')=3` → clipped to n-1=2,
    # idx_lower=1, lower_point=points[1]=1.0, upper_point=points[2]=1.0,
    # step_size=0 → decimal_part = 0/0 = nan.
    points = jnp.array([0.0, 1.0, 1.0])
    coord = get_irreg_coordinate(value=jnp.array(1.0), points=points)
    assert not jnp.isfinite(coord), (
        "Duplicate adjacent grid points cause `step_size = 0` in "
        "`get_irreg_coordinate`; current behaviour is to silently return "
        "inf/nan, which then poisons V interpolation downstream."
    )


# ---------------------------------------------------------------------------
# `validate_initial_conditions` uses `_base_state_action_space` directly,
# which still holds the placeholder zeros for runtime-supplied
# `IrregSpacedGrid`. With a feasibility constraint that the all-zero
# placeholder fails, every subject is reported infeasible — even though the
# real (post-substitution) grid would pass. This affects runtime grids
# regardless of whether they are state or action grids.
# ---------------------------------------------------------------------------


def _runtime_state_grid_model() -> tuple[Model, dict, dict]:
    """A 2-regime model with a runtime-supplied IrregSpacedGrid *state*."""

    @categorical(ordered=False)
    class RuntimeRegimeId:
        alive: int
        dead: int

    def utility(consumption, wealth):
        return jnp.log(consumption) + 0.0 * wealth

    def next_wealth(wealth, consumption):
        return wealth - consumption

    def borrow(consumption, wealth):  # noqa: ARG001
        # The validator sees `wealth` as a per-subject array with the
        # subject-supplied initial values, but `consumption` as the *grid*
        # (placeholder zeros for runtime grids). With a feasibility check
        # that requires `consumption > 0`, every action gridpoint is
        # infeasible until the runtime points replace the placeholder.
        return consumption > 0

    def next_regime(age, last_alive_age):
        return jnp.where(
            age >= last_alive_age, RuntimeRegimeId.dead, RuntimeRegimeId.alive
        )

    last_alive_age = 1
    alive = Regime(
        functions={"utility": utility},
        states={"wealth": IrregSpacedGrid(n_points=4)},  # runtime state grid
        state_transitions={"wealth": next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=5.0, n_points=5)},
        constraints={"borrow": borrow},
        transition=next_regime,
        active=lambda age: age <= last_alive_age,
    )
    dead = Regime(
        transition=None,
        functions={"utility": lambda: 0.0},
        active=lambda age: age > last_alive_age,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regime_id_class=RuntimeRegimeId,
    )
    params = {
        "discount_factor": 0.95,
        "alive": {
            "wealth": {"points": jnp.linspace(1.0, 10.0, 4)},
            "next_regime": {"last_alive_age": last_alive_age},
        },
    }
    initial_conditions = {
        "age": jnp.array([0.0, 0.0, 0.0]),
        "wealth": jnp.array([2.0, 5.0, 9.0]),
        "regime": jnp.array(
            [RuntimeRegimeId.alive, RuntimeRegimeId.alive, RuntimeRegimeId.alive],
            dtype=jnp.int32,
        ),
    }
    return model, params, initial_conditions


def test_runtime_action_grid_passes_initial_conditions_validation():
    """`feature/runtime-action-grids` regression: initial-conditions
    feasibility check must use the *substituted* action grid, not the
    `_base_state_action_space` placeholder zeros."""
    model, params = _build_model(
        wealth_lo=10.0,
        wealth_hi=20.0,
        n_wealth=5,
        consumption_lo=1.0,
        consumption_hi=5.0,
        n_consumption=5,
        n_periods=3,
    )
    initial_conditions = {
        "age": jnp.array([0.0, 0.0, 0.0]),
        "wealth": jnp.array([10.0, 15.0, 20.0]),
        "regime": jnp.array(
            [RegimeId.alive, RegimeId.alive, RegimeId.alive], dtype=jnp.int32
        ),
    }
    # `check_initial_conditions=True` (the default) must pass — the
    # runtime-supplied consumption points are well-formed.
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    assert result.n_subjects == 3


def test_runtime_state_grid_passes_initial_conditions_validation():
    """Same regression for runtime-supplied *state* grids."""
    model, params, initial_conditions = _runtime_state_grid_model()
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    assert result.n_subjects == 3
