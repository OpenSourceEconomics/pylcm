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
- the shape-invariance contract is enforced: same class, `batch_size`, points mode and
  resolved node shape/dtype at every active age — validated on the grid's actual
  `to_jax()` array, since `n_points` is not part of the `Grid` base contract;
- program sharing across periods is keyed on the explicit `AgeSpecializedGrid.signature`
  contract, so periods with different continuation grids never false-share a kernel.
"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids.continuous import ContinuousGrid
from _lcm.regime_building.age_specialization import _TRAIT_DESCRIPTIONS, _GridTraits
from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.regime import Regime
from lcm.typing import ScalarInt

_N = 6  # ages 20..25; working ages 20..24, terminal at 25
_AGES = AgeGrid(start=20, stop=20 + _N - 1, step="Y")
_CGRID = LinSpacedGrid(start=0.05, stop=25.0, n_points=25)
_PARAMS = {
    "alive": {
        "next_wealth": {"interest_rate": 0.05},
        "koopmans_aggregator": {"discount_factor": 0.95},
    },
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


def _alive_regime(wealth_grid, *, active=lambda age: age < 20 + _N - 1):
    return Regime(
        active=active,
        states={"wealth": wealth_grid},
        actions={"consumption": _CGRID},
        state_transitions={"wealth": _next_wealth},
        transition=_next_regime,
        constraints={"bc": _bc},
        functions={"utility": _utility},
    )


def _model(wealth_grid):
    return Model(
        regimes={"alive": _alive_regime(wealth_grid), "dead": _DEAD},
        ages=_AGES,
        regime_id_class=RegimeId,
        fixed_params={"last": _N - 2},
    )


def test_under_specified_signature_merging_distinct_grids_is_rejected():
    """Two periods sharing a `signature(age)` but resolving to distinct grids raise.

    `signature(age)` is a cheap, user-supplied dedup pre-filter, not a substitute
    for comparing resolved nodes: an equal signature across periods whose grids
    genuinely differ must be caught at build time (a loud, actionable error), not
    silently solved against the wrong period's continuation grid.
    """

    def floor(age):
        return -2.0 + 0.3 * (age - 20)

    under_specified = AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(start=floor(age), stop=20.0, n_points=12),
        signature=lambda _age: "constant",
    )
    with pytest.raises(RegimeInitializationError, match="resolved nodes"):
        _model(under_specified).solve(params=_PARAMS, log_level="off")


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


def test_weak_type_change_across_ages_is_rejected():
    """JAX `weak_type` steers promotion but is erased by `np.asarray`.

    Two axes can agree on dtype, shape and raw bytes yet promote differently in
    the shared trace, changing the argmax. Varying it across ages violates the
    only-node-values-may-vary contract, so the shape-invariance validation covers
    `weak_type` too and surfaces the change as a construction-time error, not a
    silent mis-share.
    """

    @dataclass(frozen=True)
    class _WeakTypeGrid(ContinuousGrid):
        weak: bool = False

        def to_jax(self):
            # Stacking Python scalars yields a weak array; converting it to its own
            # dtype strips the weak flag and nothing else. Deriving the strong array
            # from the weak one keeps the pair identical at either --precision.
            nodes = jnp.stack([jnp.asarray(v) for v in (0.5, 1.0, 25.0)])
            if self.weak:
                return nodes
            return jax.lax.convert_element_type(nodes, nodes.dtype)

        def get_coordinate(self, value):  # pragma: no cover - not exercised here
            raise NotImplementedError

    weak, strong = _WeakTypeGrid(weak=True), _WeakTypeGrid(weak=False)
    weak_nodes, strong_nodes = weak.to_jax(), strong.to_jax()

    # Identical in everything the host-side node array can see; only weak_type differs.
    assert weak_nodes.weak_type
    assert not strong_nodes.weak_type
    assert weak_nodes.dtype == strong_nodes.dtype
    assert np.asarray(weak_nodes).tobytes() == np.asarray(strong_nodes).tobytes()

    grid = AgeSpecializedGrid(
        build=lambda age: _WeakTypeGrid(weak=age == 20),
        signature=lambda age: age,
    )
    with pytest.raises(RegimeInitializationError, match=r"weak_type"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_every_grid_trait_is_described():
    """Every invariant trait must have a mismatch message.

    `_GridTraits` and `_TRAIT_DESCRIPTIONS` are two lists of the same thing, so a trait
    added to one and not the other would raise "no described trait does" instead of
    naming the real cause. Keeps them in lockstep.
    """
    described = {field for field, _label, _render in _TRAIT_DESCRIPTIONS}
    assert {f.name for f in dataclasses.fields(_GridTraits)} == described


def test_builtin_grids_are_never_weak_typed():
    """The weak-type key cannot split a supported grid.

    Guards the claim that justifies keying on `weak_type` at all: no built-in grid
    yields a weak array, so the constant-grid fast path is untouched.
    """
    for built_in in (
        LinSpacedGrid(start=0.0, stop=1.0, n_points=3),
        LogSpacedGrid(start=1.0, stop=10.0, n_points=3),
        IrregSpacedGrid(points=[0.0, 0.5, 1.0]),
    ):
        assert not built_in.to_jax().weak_type


def test_validation_rejects_actual_node_count_change_without_n_points():
    """A custom grid with no `n_points` must be validated on its resolved array.

    `n_points` is not part of the `Grid` base contract — only `to_jax()` is — so
    validation must derive shape from the resolved array rather than
    `getattr(grid, "n_points", 0)`, which would silently agree at 0 for two grids
    of different actual length and let the shape change reach the compiled kernel
    unnoticed.
    """

    @dataclass(frozen=True)
    class _NoNPointsGrid(ContinuousGrid):
        nodes: tuple[float, ...] = (0.0, 1.0)

        def to_jax(self):
            return jnp.asarray(self.nodes)

        def get_coordinate(self, value):  # pragma: no cover - not exercised here
            raise NotImplementedError

    grid = AgeSpecializedGrid(
        build=lambda age: _NoNPointsGrid(
            nodes=(0.5, 1.0, 25.0) if age == 20 else (0.5, 1.0, 12.0, 25.0)
        ),
        signature=lambda age: age,
    )
    with pytest.raises(RegimeInitializationError, match=r"n_points|node shape"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_validation_rejects_node_dtype_change():
    """A dtype change at constant `n_points` must be rejected.

    The shared kernel is lowered against the representative axis, so a later period axis
    of the same shape but a different dtype is rejected by the compiled executable.
    """

    @dataclass(frozen=True)
    class _DtypeGrid(ContinuousGrid):
        n_points: int = 3
        dtype: str = "float32"

        def to_jax(self):
            return jnp.linspace(0.5, 25.0, self.n_points, dtype=jnp.dtype(self.dtype))

        def get_coordinate(self, value):  # pragma: no cover - not exercised here
            raise NotImplementedError

    grid = AgeSpecializedGrid(
        build=lambda age: _DtypeGrid(dtype="float16" if age == 20 else "float32"),
        signature=lambda age: age,
    )
    with pytest.raises(RegimeInitializationError, match="dtype"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_validation_rejects_declared_n_points_disagreeing_with_nodes():
    """A grid whose declared `n_points` contradicts its own `to_jax()` is rejected."""

    @dataclass(frozen=True)
    class _LyingGrid(ContinuousGrid):
        n_points: int = 15

        def to_jax(self):
            return jnp.linspace(0.5, 25.0, 4)

        def get_coordinate(self, value):  # pragma: no cover - not exercised here
            raise NotImplementedError

    grid = AgeSpecializedGrid(build=lambda _age: _LyingGrid(), signature=lambda _age: 0)
    with pytest.raises(RegimeInitializationError, match="declares n_points"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_grid_mode_switch_across_ages_is_rejected():
    """A grid may not supply points concretely at one age and at runtime at another.

    Class and `n_points` stay equal, so the shape check alone cannot catch this;
    without an explicit points-mode check, the failure would surface much later,
    out of `to_jax()` during period-axis construction, as an error naming the
    wrong cause entirely.
    """

    def build(age):
        if age == 20:
            return IrregSpacedGrid(points=[0.5, 5.0, 25.0])
        return IrregSpacedGrid(n_points=3)  # points supplied at runtime

    grid = AgeSpecializedGrid(build=build, signature=lambda age: age == 20)
    with pytest.raises(RegimeInitializationError, match="supplied at runtime"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_age_specialized_grid_on_never_active_regime_is_rejected():
    """An age-specialized grid on a regime active at no age is a modelling error.

    There is no age at which to resolve the builder, so the marker must be
    rejected up front, rather than travelling unresolved into the ordinary grid
    machinery it does not satisfy.
    """
    grid = AgeSpecializedGrid(
        build=lambda _age: LinSpacedGrid(start=0.5, stop=25.0, n_points=15),
        signature=lambda _age: 0,
    )
    with pytest.raises(RegimeInitializationError, match="active at no model age"):
        Model(
            regimes={
                "alive": _alive_regime(grid, active=lambda _age: False),
                "dead": _DEAD,
            },
            ages=_AGES,
            regime_id_class=RegimeId,
            fixed_params={"last": _N - 2},
        )


def test_builder_undefined_outside_active_ages_still_solves():
    """A grid builder that is undefined outside its regime's active ages must still
    build and solve.

    Validation and per-period resolution must call `build(age)` only at the
    regime's active ages, not the whole model horizon — otherwise a builder that
    deliberately raises outside its regime's active ages would turn a valid
    age-limited/terminal-only specialization into a construction failure. Here
    the `alive` regime is active through age 24 and inactive at the terminal age
    25; the builder raises at every inactive age.
    """
    inactive_age = 20 + _N - 1  # the terminal (dead) age; alive is inactive here

    def build(age):
        if age >= inactive_age:
            raise ValueError(f"grid undefined at inactive age {age}")
        return LinSpacedGrid(start=0.5, stop=25.0, n_points=15)

    grid = AgeSpecializedGrid(build=build, signature=lambda _age: 0)
    v = _model(grid).solve(params=_PARAMS, log_level="off")
    assert any("alive" in v[period] for period in range(_N))
