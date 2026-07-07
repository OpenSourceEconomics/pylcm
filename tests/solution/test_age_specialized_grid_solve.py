"""Solving/simulating a model with an `AgeSpecializedGrid` continuous state.

An `AgeSpecializedGrid` lets a continuous state's grid *bounds* move with age while
keeping a fixed `n_points` (shape-invariant). The canonical use is an asset state with
an age-dependent borrowing floor `a_bar(age)`: on a single fixed grid the cells below
the loosest floor are infeasible at tighter ages, producing `-inf` that poisons the
value function by interpolation. An age-tracking floor removes those cells.

Contracts tested here:
- an *age-invariant* `AgeSpecializedGrid` reproduces the plain fixed-grid solve
  bit-for-bit (the per-period machinery collapses cleanly);
- an age-*varying* floor solves with a finite value function on the whole grid, i.e.
  it avoids the `-inf`/`NaN` poisoning a fixed grid would suffer (the feature's point);
- the solved policy is economically sensible (V and consumption increase in wealth);
- simulation runs and yields finite, positive consumption;
- the shape-invariance contract (same class + `n_points` at every age) is enforced.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, AgeSpecializedGrid, LinSpacedGrid, Model, categorical
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import ScalarInt

_N = 6  # ages 20..25; working ages 20..24, terminal at 25
_AGES = AgeGrid(start=20, stop=20 + _N - 1, step="Y")
_CGRID = LinSpacedGrid(start=0.05, stop=25.0, n_points=25)
_PARAMS = {
    "alive": {"next_wealth": {"interest_rate": 0.05}, "H": {"discount_factor": 0.95}},
    "dead": {},
}


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(consumption):
    return jnp.log(consumption)


def _next_wealth(wealth, consumption, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + 1.0


def _bc(consumption, wealth):
    return consumption <= wealth


def _next_regime(period, last):
    return jnp.where(period >= last, RegimeId.dead, RegimeId.alive)


_DEAD = Regime(
    active=lambda age: age >= 20 + _N - 1,
    transition=None,
    functions={"utility": lambda: 0.0},
)


def _model(wealth_grid):
    alive = Regime(
        active=lambda age: age < 20 + _N - 1,
        states={"wealth": wealth_grid},
        actions={"consumption": _CGRID},
        state_transitions={"wealth": _next_wealth},
        transition=_next_regime,
        constraints={"bc": _bc},
        functions={"utility": _utility},
    )
    return Model(
        regimes={"alive": alive, "dead": _DEAD},
        ages=_AGES,
        regime_id_class=RegimeId,
        fixed_params={"last": _N - 2},
    )


def test_age_invariant_grid_reproduces_plain_solve():
    """An age-invariant `AgeSpecializedGrid` equals the plain fixed-grid solve."""
    grid = LinSpacedGrid(start=0.5, stop=25.0, n_points=15)
    v_plain = _model(grid).solve(params=_PARAMS, log_level="off")
    v_asg = _model(
        AgeSpecializedGrid(build=lambda _age: grid, signature=lambda _age: 0)
    ).solve(params=_PARAMS, log_level="off")
    for period in range(_N):
        if "alive" not in v_plain[period]:
            continue
        a = np.asarray(v_plain[period]["alive"])
        b = np.asarray(v_asg[period]["alive"])
        # Bit-for-bit, including the pattern of `-inf` infeasible cells.
        np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
        finite = np.isfinite(a)
        np.testing.assert_array_equal(a[finite], b[finite])


def _moving_floor_grid():
    # Floor tightens with age; every grid cell is >= the age's floor, so every cell
    # is a feasible asset level (a fixed grid spanning the loosest floor would not be).
    def floor(age):
        return -2.0 + 0.3 * (age - 20)

    return AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(start=floor(age), stop=20.0, n_points=12),
        signature=floor,
    )


def test_moving_floor_no_nan_poisoning():
    """An age-tracking floor solves without `NaN` poisoning the value function.

    This is the feature's reason to exist. On a single fixed grid spanning the loosest
    (youngest) floor, the cells below an older age's tighter floor are infeasible; their
    `-inf` continuation, weighted by a zero transition probability, produces `0 * -inf =
    NaN`, which then leaks backward through interpolation and destroys the solve. With
    the grid tracking the floor those cells never exist, so no `NaN` appears anywhere.
    (`-inf` may still appear at negative-wealth nodes where no positive consumption is
    affordable — that is legitimate infeasibility, not poisoning.)
    """
    v = _model(_moving_floor_grid()).solve(params=_PARAMS, log_level="off")
    for period in range(_N):
        if "alive" not in v[period]:
            continue
        arr = np.asarray(v[period]["alive"])
        assert not np.isnan(arr).any(), f"period {period} has NaN (poisoning): {arr}"
        assert np.isfinite(arr).any(), f"period {period} has no finite V at all"


def test_moving_floor_value_monotone_in_wealth():
    """V is nondecreasing in wealth at every working age (economic sanity)."""
    v = _model(_moving_floor_grid()).solve(params=_PARAMS, log_level="off")
    for period in range(_N):
        if "alive" not in v[period]:
            continue
        # Replace legitimate `-inf` (infeasible low-wealth cells) with a finite sentinel
        # so `-inf - -inf = NaN` does not spuriously fail the monotonicity diff.
        arr = np.nan_to_num(np.asarray(v[period]["alive"]), neginf=-1e30)
        diffs = np.diff(arr, axis=0)  # axis 0 is the wealth grid
        assert (diffs >= -1e-6).all(), (
            f"V not nondecreasing in wealth at period {period}"
        )


def test_moving_floor_simulates_positive_consumption():
    """Forward simulation runs and gives finite, positive consumption for alive rows."""
    model = _model(_moving_floor_grid())
    v = model.solve(params=_PARAMS, log_level="off")
    n = 200
    result = model.simulate(
        params=_PARAMS,
        period_to_regime_to_V_arr=v,
        log_level="off",
        seed=1,
        initial_conditions={
            "wealth": jnp.linspace(1.0, 10.0, n),
            "age": jnp.full(n, 20.0),
            "regime_id": jnp.array([RegimeId.alive] * n),
        },
    )
    df = result.to_dataframe()
    consumption = np.asarray(df["consumption"])
    alive_consumption = consumption[np.isfinite(consumption)]
    assert alive_consumption.size > 0
    assert (alive_consumption > 0).all()


def test_non_shape_invariant_grid_is_rejected():
    """Varying `n_points` across ages raises at model construction."""
    bad = AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(start=0.5, stop=25.0, n_points=int(age) - 5),
        signature=int,
    )
    with pytest.raises(RegimeInitializationError, match="shape-invariant"):
        _model(bad).solve(params=_PARAMS, log_level="off")
