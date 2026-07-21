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
- grid identity keys on resolved nodes, so equal-shape/different-geometry grids never
  share current-state axes or continuation kernels.
"""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from _lcm.grids import PiecewiseGridSegment, PiecewiseLinSpacedGrid
from _lcm.grids.continuous import ContinuousGrid
from _lcm.regime_building.age_specialization import _TRAIT_DESCRIPTIONS, _GridTraits
from _lcm.regime_building.processing import _grid_identity
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


def test_grid_identity_distinguishes_piecewise_geometry():
    """Two piecewise grids with equal total `n_points` but different breakpoints must
    have distinct grid identities (audit F1).

    The old `_grid_identity` fell back to `(class, n_points)` for any grid without
    `start/stop` or a `points` attribute — which is *every* piecewise grid — so two
    different geometries collided. That identity drives both the per-period
    current-state axes and the continuation-cache grouping, so a collision silently
    reused the wrong
    axes/kernels with no shape error. The fix keys on the resolved `to_jax()` nodes.
    """
    left = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0, 1)", n_points=3),
            PiecewiseGridSegment(interval="[1, 5]", n_points=4),
        )
    )
    right = PiecewiseLinSpacedGrid(
        segments=(
            PiecewiseGridSegment(interval="[0, 4)", n_points=3),
            PiecewiseGridSegment(interval="[4, 5]", n_points=4),
        )
    )
    assert int(left.n_points) == int(right.n_points)
    assert not np.array_equal(np.asarray(left.to_jax()), np.asarray(right.to_jax()))
    assert _grid_identity(left) != _grid_identity(right)


def test_grid_identity_uses_nodes_for_custom_grid_exposing_start_stop():
    """A custom grid is keyed on its resolved nodes even when it exposes
    `start`/`stop`/`n_points` (re-review F1).

    The first fix keyed on the *presence* of those attributes, so a custom grid whose
    geometry also depends on another field (here a spacing exponent) kept one identity
    across ages while its nodes moved — silently reusing the wrong current-state axes
    and continuation kernels, with no shape error to catch it.
    """

    @dataclass(frozen=True)
    class _PowerSpacedGrid(ContinuousGrid):
        """A custom grid whose nodes depend on `power` as well as on start/stop."""

        start: float = 0.0
        stop: float = 1.0
        n_points: int = 5
        power: float = 1.0

        def to_jax(self):
            unit = jnp.linspace(0.0, 1.0, self.n_points) ** self.power
            return self.start + (self.stop - self.start) * unit

        def get_coordinate(self, value):  # pragma: no cover - not exercised here
            raise NotImplementedError

    linear = _PowerSpacedGrid(power=1.0)
    quadratic = _PowerSpacedGrid(power=2.0)
    assert not np.array_equal(
        np.asarray(linear.to_jax()), np.asarray(quadratic.to_jax())
    )
    assert _grid_identity(linear) != _grid_identity(quadratic)


def test_runtime_irregular_subclass_gets_structural_identity():
    """A *subclass* of `IrregSpacedGrid` with runtime-supplied points must key on its
    shape, not be sent to the node fingerprint (round-3 re-review F1).

    `V._get_coordinate_finder` dispatches the runtime-points path on `isinstance`, so
    such a subclass is a supported runtime grid there. Keying identity on the exact type
    alone sent it to the node branch, where its inherited `to_jax()` must raise — the
    two dispatches disagreed about the same object.
    """

    class _MyRuntimeGrid(IrregSpacedGrid):
        pass

    grid = _MyRuntimeGrid(n_points=3)  # points supplied at runtime
    assert grid.pass_points_at_runtime
    identity = _grid_identity(grid)  # must not raise
    assert identity == (_MyRuntimeGrid, 3)
    # A different concrete class with the same shape must not collide.
    assert identity != _grid_identity(IrregSpacedGrid(n_points=3))


def test_grid_identity_distinguishes_concrete_subclass_by_nodes():
    """A concrete subclass overriding `to_jax` is still keyed on its real nodes."""

    class _ShiftedIrreg(IrregSpacedGrid):
        def to_jax(self):
            return super().to_jax() + 1.0

    base = IrregSpacedGrid(points=[0.0, 1.0, 2.0])
    shifted = _ShiftedIrreg(points=[0.0, 1.0, 2.0])
    assert _grid_identity(base) != _grid_identity(shifted)


def test_grid_identity_distinguishes_signed_zero_endpoints():
    """A uniform grid is keyed on its nodes, not on its (start, stop) description.

    Round-4 re-review F1: the cheap `(class, float(start), float(stop), n_points)` key
    collapsed `-0.0` and `+0.0`, which `jnp.linspace` preserves as different endpoint
    bits. The two grids then shared one identity and the solver silently reused the
    representative axis.
    """
    negative_zero = LinSpacedGrid(start=-1.0, stop=-0.0, n_points=3)
    positive_zero = LinSpacedGrid(start=-1.0, stop=0.0, n_points=3)

    # The nodes really do differ — the collapse was in the key, not in the grid.
    assert np.asarray(negative_zero.to_jax()).tobytes() != (
        np.asarray(positive_zero.to_jax()).tobytes()
    )
    assert _grid_identity(negative_zero) != _grid_identity(positive_zero)


def test_grid_identity_distinguishes_extended_dtypes_with_equal_bytes():
    """`dtype.str` is not injective over JAX's extended floating types.

    Round-4 re-review F1: `float8_e4m3fnuz` and `float8_e5m2fnuz` both report `'<V1'`,
    so two same-shape arrays with identical raw bytes decoded to *different numbers*
    while their fingerprints compared equal. The exact `np.dtype` object separates them.
    """

    @dataclass(frozen=True)
    class _Float8Grid(ContinuousGrid):
        dtype: object = ml_dtypes.float8_e4m3fnuz

        def to_jax(self):
            raw = np.asarray([0x30, 0x38, 0x40], dtype=np.uint8)
            return jnp.asarray(raw.view(np.dtype(self.dtype)))

        def get_coordinate(self, value):  # pragma: no cover - not exercised here
            raise NotImplementedError

    left = _Float8Grid(dtype=ml_dtypes.float8_e4m3fnuz)
    right = _Float8Grid(dtype=ml_dtypes.float8_e5m2fnuz)

    left_nodes, right_nodes = np.asarray(left.to_jax()), np.asarray(right.to_jax())
    # Same class, same shape, same bytes, same dtype.str — different numbers.
    assert left_nodes.dtype.str == right_nodes.dtype.str == "<V1"
    assert left_nodes.tobytes() == right_nodes.tobytes()
    assert not np.array_equal(
        left_nodes.astype(np.float32), right_nodes.astype(np.float32)
    )

    assert _grid_identity(left) != _grid_identity(right)

    grid = AgeSpecializedGrid(
        build=lambda age: _Float8Grid(
            dtype=ml_dtypes.float8_e4m3fnuz if age == 20 else ml_dtypes.float8_e5m2fnuz
        ),
        signature=lambda age: age,
    )
    with pytest.raises(RegimeInitializationError, match=r"dtype"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_grid_identity_distinguishes_weak_type():
    """JAX `weak_type` steers promotion but is erased by `np.asarray`.

    Round-5 hardening note: two axes can agree on dtype, shape and raw bytes yet
    promote differently in the shared trace, changing the argmax. Varying it across
    ages violates the only-node-values-may-vary contract, so this is defence in depth:
    it must surface as a construction-time error, not a silent mis-share.
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

    # Identical in everything the host-side fingerprint can see.
    assert weak_nodes.weak_type
    assert not strong_nodes.weak_type
    assert weak_nodes.dtype == strong_nodes.dtype
    assert np.asarray(weak_nodes).tobytes() == np.asarray(strong_nodes).tobytes()

    assert _grid_identity(weak) != _grid_identity(strong)

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
    """A custom grid with no `n_points` must be validated on its resolved array
    (round-3 re-review F2).

    `n_points` is not part of the `Grid` base contract — only `to_jax()` is — so
    `getattr(grid, "n_points", 0)` silently agreed at 0 for two grids of different
    actual length, and the shape change slipped through to the compiled kernel.
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
    """A dtype change at constant `n_points` must be rejected (round-3 re-review F2).

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
    """A grid may not supply points concretely at one age and at runtime at another
    (re-review F2).

    Class and `n_points` stay equal, so the shape check passed and the failure surfaced
    much later, out of `to_jax()` during period-axis construction, as an error about
    the wrong thing entirely.
    """

    def build(age):
        if age == 20:
            return IrregSpacedGrid(points=[0.5, 5.0, 25.0])
        return IrregSpacedGrid(n_points=3)  # points supplied at runtime

    grid = AgeSpecializedGrid(build=build, signature=lambda age: age == 20)
    with pytest.raises(RegimeInitializationError, match="points mode"):
        _model(grid).solve(params=_PARAMS, log_level="off")


def test_age_specialized_grid_on_never_active_regime_is_rejected():
    """An age-specialized grid on a regime active at no age is a modelling error
    (re-review F3).

    There is no age at which to resolve the builder, so the unresolved marker used to
    travel into the ordinary grid machinery instead of being rejected up front.
    """
    grid = AgeSpecializedGrid(
        build=lambda _age: LinSpacedGrid(start=0.5, stop=25.0, n_points=15),
        signature=lambda _age: 0,
    )
    with pytest.raises(RegimeInitializationError, match="active at no age"):
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
    build and solve (audit F2).

    Validation and per-period resolution used to call `build(age)` over the whole model
    horizon, so a builder that deliberately raises where its regime is inactive turned a
    valid age-limited/terminal-only specialization into a construction failure. Here the
    `alive` regime is active through age 24 and inactive at the terminal age 25; the
    builder raises at every inactive age.
    """
    inactive_age = 20 + _N - 1  # the terminal (dead) age; alive is inactive here

    def build(age):
        if age >= inactive_age:
            raise ValueError(f"grid undefined at inactive age {age}")
        return LinSpacedGrid(start=0.5, stop=25.0, n_points=15)

    grid = AgeSpecializedGrid(build=build, signature=lambda _age: 0)
    v = _model(grid).solve(params=_PARAMS, log_level="off")
    assert any("alive" in v[period] for period in range(_N))
