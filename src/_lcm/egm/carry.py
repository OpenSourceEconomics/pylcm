"""Cross-period data channel of the DC-EGM solver.

Backward induction with DC-EGM threads more than the value-function array
between adjacent periods: the parent's Euler inversion needs the child's
value and marginal utility on the child's endogenous (resources-space) grid.
`EGMCarry` bundles these rows; the solve loop rolls a
`next_regime_to_continuation` mapping alongside `next_regime_to_V_arr`, with one
entry per carry-producing regime (DC-EGM regimes and terminal regimes a
DC-EGM regime can target).

The carry is also the shipped `ContinuationReader`: a parent asks it for a value
or a marginal in `EGM_ENDOGENOUS_COORDINATE` at a query instead of interpolating
its rows itself. `read_value_row`, `read_marginal_row`, and
`read_value_and_slope_row` are the per-row kernels behind those answers, shared
with the multi-target continuation aggregation, so the reader and the hot path
interpolate through one implementation.
"""

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.interp import (
    interp_and_derivative_on_prepared_grid,
    interp_on_prepared_grid,
    prepare_padded_grid,
)
from lcm.solver_api import (
    EGM_CONTINUATION,
    EGM_ENDOGENOUS_COORDINATE,
    ArtifactKey,
    ContinuationCapabilities,
)
from lcm.typing import (
    Float1D,
    Float2D,
    FloatND,
    Int1D,
    ScalarFloat,
    ScalarInt,
    StateName,
    StateOrActionName,
)


@dataclass(frozen=True, kw_only=True)
class EGMCarry:
    """Per-regime EGM solution rows threaded between adjacent periods.

    All rows share the trailing grid axis of static, per-regime length so the
    carry has a period-invariant pytree shape (periods sharing a compiled
    program never trigger retracing). Regimes with combo dimensions carry one
    row per combo: leading axes are the regime's discrete states (in V state
    order; process states are node-valued discrete dimensions), then its
    passive continuous states (in V continuous-state order, one node per
    combo), then its discrete actions. Every array is pinned to the canonical
    float dtype.
    """

    endog_grid: FloatND
    """Endogenous grid in resources space, NaN-padded in the tail.

    Weakly ascending per row: envelope kink abscissae appear twice, carrying
    the left- and right-extrapolated policy values.
    """

    value: FloatND
    """Choice-specific value at `endog_grid`; `-inf` marks infeasible rows."""

    marginal_utility: FloatND
    """Marginal value of resources $\\partial v / \\partial R$ at `endog_grid`.

    Exactly `0.0` (never NaN) wherever `value` is `-inf`: infeasible rows get
    zero choice probability, and `0 \\cdot \\mu` must stay finite in the
    parent's probability-weighted expectation.
    """

    taste_shock_scale: ScalarFloat
    """EV1 taste-shock scale of the regime as a 0-d array; `0.0` = hard max."""

    breakpoints: FloatND | None = None
    """Per-row one-sided boundary locations in the child's liquid state.

    NaN-padded and published by solvers whose value rows carry a declared jump
    or a finite-to-infeasible boundary; `None` for smooth-valued regimes.
    One-sided values and marginals ride inside `endog_grid` as exact/adjacent
    or duplicated abscissae. This field marks the rows' topology for the
    stochastic-dimension fold, which must not average rows whose boundary
    locations differ across the folded nodes, and lets an EGM parent add the
    fixed-savings candidate where the child boundary binds.
    """

    @property
    def artifact_key(self) -> ArtifactKey:
        """Versioned identity under which EGM kernels publish this carry."""
        return EGM_CONTINUATION

    policy: FloatND | None = None
    """Exact consumption at `endog_grid`, or `None` when not published.

    Populated only by the continuous-only, jump-free ride-along NB-EGM core
    (`_assemble_ride_carry`), whose per-cell EGM step already computes the
    optimal consumption on the liquid grid. The continuous-outer simulation
    replay (`derive_inner_sim_policy`) reads this directly instead of
    re-inverting `marginal_utility`: the marginal carries the budget's
    cash-on-hand slope in liquid (`marginal = (d coh/d liquid) * u'(c)`), so
    inverting it recovers `c` only when that slope is unit — carrying the
    exact policy is correct for any affine budget slope.
    `None` for every other carry (cross-period continuations, discrete-branch
    or jump-schedule rows), whose downstream never reaches that replay.
    """

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Return what these rows can answer.

        The rows carry a value everywhere and its exact slope in the endogenous
        coordinate. They name no owning candidate, and they locate their own
        boundaries only when a `breakpoints` row is published.
        """
        return ContinuationCapabilities(
            value=True,
            marginal_states=frozenset({EGM_ENDOGENOUS_COORDINATE}),
            exact_candidate_identity=False,
            discontinuities=self.breakpoints is not None,
        )

    def value_at(self, *, query: FloatND) -> FloatND:
        """Return the value of these rows at `query`, one query per row."""
        rows = self._row_arguments(query=query)
        return jax.vmap(read_value_row)(
            search_grid=rows.search_grid,
            valid_length=rows.valid_length,
            xp=rows.xp,
            fp=self._flat_rows(array=self.value),
            fp_slopes=rows.fp_slopes,
            x_query=rows.x_query,
        ).reshape(self.endog_grid.shape[:-1])

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Return the marginal in `state` at `query`, one query per row."""
        capabilities = self.capabilities
        if state not in capabilities.marginal_states:
            msg = (
                "An EGM carry publishes a marginal in "
                f"{sorted(capabilities.marginal_states)}, not in {state!r}."
            )
            raise ValueError(msg)
        rows = self._row_arguments(query=query)
        return jax.vmap(read_marginal_row)(
            search_grid=rows.search_grid,
            valid_length=rows.valid_length,
            xp=rows.xp,
            fp=self._flat_rows(array=self.marginal_utility),
            x_query=rows.x_query,
        ).reshape(self.endog_grid.shape[:-1])

    def leaves(self) -> MappingProxyType[tuple[str, ...], FloatND]:
        """Return every published array of this carry by its field path."""
        return MappingProxyType(
            {
                (name,): getattr(self, name)
                for name in _EGM_CARRY_FIELDS
                if getattr(self, name) is not None
            }
        )

    def _flat_rows(self, *, array: FloatND) -> Float2D:
        """Collapse every leading axis of one payload row set."""
        return array.reshape(-1, self.endog_grid.shape[-1])

    def _row_arguments(self, *, query: FloatND) -> _EGMRowArguments:
        """Build the per-row read arguments shared by both query methods."""
        grid_rows = self._flat_rows(array=self.endog_grid)
        search_rows, valid_rows = jax.vmap(prepare_padded_grid)(grid_rows)
        return _EGMRowArguments(
            search_grid=search_rows,
            valid_length=valid_rows,
            xp=grid_rows,
            fp_slopes=self._flat_rows(array=self.marginal_utility),
            x_query=jnp.broadcast_to(query, self.endog_grid.shape[:-1]).reshape(-1),
        )


# Pytree registration with an `__init__`-bypassing unflatten: JAX's transform
# and AOT-lowering machinery reconstructs pytrees with non-array leaves
# (`ArgInfo`, tracers, `None`), which the runtime-checked constructor would
# reject. Flatten order matches field declaration order.
_EGM_CARRY_FIELDS = (
    "endog_grid",
    "value",
    "marginal_utility",
    "taste_shock_scale",
    "breakpoints",
    "policy",
)


def _flatten_egm_carry(carry: EGMCarry) -> tuple[tuple[Any, ...], None]:
    return tuple(getattr(carry, name) for name in _EGM_CARRY_FIELDS), None


def _flatten_egm_carry_with_keys(
    carry: EGMCarry,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, Any], ...], None]:
    """Flatten with field-named keys so a leaf path reads `.endog_grid`."""
    return (
        tuple(
            (jax.tree_util.GetAttrKey(name), getattr(carry, name))
            for name in _EGM_CARRY_FIELDS
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_egm_carry(_aux: None, children: Iterable[Any]) -> EGMCarry:
    carry = object.__new__(EGMCarry)
    for name, child in zip(_EGM_CARRY_FIELDS, children, strict=True):
        object.__setattr__(carry, name, child)
    return carry


jax.tree_util.register_pytree_with_keys(
    EGMCarry,
    _flatten_egm_carry_with_keys,
    _unflatten_egm_carry,
    _flatten_egm_carry,
)


def read_value_row(
    *,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
    x_query: ScalarFloat,
) -> ScalarFloat:
    """Interpolate one carry value row at its query."""
    return interp_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


def read_marginal_row(
    *,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    x_query: ScalarFloat,
) -> ScalarFloat:
    """Interpolate one carry row at its own query."""
    return interp_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
    )


def read_value_and_slope_row(
    *,
    search_grid: Float1D,
    valid_length: ScalarInt,
    xp: Float1D,
    fp: Float1D,
    fp_slopes: Float1D,
    x_query: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Value read and its analytic derivative.

    The closed-form derivative of the selected piece — not autodiff through
    the bracket-selection program, whose `searchsorted`/`clip` representation
    returns arbitrary subgradients at exact grid nodes (a routine alignment: a
    zero-savings corner on a child grid that starts at zero).
    """
    return interp_and_derivative_on_prepared_grid(
        x_query=x_query,
        search_grid=search_grid,
        valid_length=valid_length,
        xp=xp,
        fp=fp,
        fp_slopes=fp_slopes,
    )


@dataclass(frozen=True, kw_only=True)
class _EGMRowArguments:
    """The per-row abscissae one query pass reads every carry row on."""

    search_grid: Float2D
    """Padded search grids, one row per flattened leading cell."""

    valid_length: Int1D
    """Number of valid nodes in each row's search grid."""

    xp: Float2D
    """The endogenous grid rows the values are tabulated on."""

    fp_slopes: Float2D
    """The marginal rows used as the value interpolant's slopes."""

    x_query: Float1D
    """One query per flattened leading cell."""


def egm_carry_role_tree(
    *,
    row: object,
    scalar: object,
    breakpoints: object | None,
    policy: object | None,
) -> EGMCarry:
    """Build an `EGMCarry`-shaped tree of output roles.

    A kernel declares its carry outputs with the same pytree structure as the
    carry it publishes: one role for each of the three grid rows, one for the
    0-d taste-shock scale, and `None` for a row it does not publish. The leaves
    are role declarations rather than arrays, so the tree is assembled through
    the pytree unflatten rather than the runtime-checked constructor.
    """
    return _unflatten_egm_carry(None, (row, row, row, scalar, breakpoints, policy))


def build_template_egm_carry(
    *,
    n_rows: int,
    leading_shape: tuple[int, ...] = (),
) -> EGMCarry:
    """Build a benign all-finite carry template with `n_rows` grid slots.

    Used to initialize the rolling `next_regime_to_continuation` mapping before
    a regime has been solved, and as the lowering argument when AOT-compiling
    EGM kernels. The endogenous grid is strictly ascending and every row is
    finite, so a parent kernel evaluated against the template produces finite
    (probability-zeroed) contributions rather than NaN.

    Args:
        n_rows: Static length of the carry rows.
        leading_shape: Sizes of the regime's combo dimensions (discrete
            states, then passive states, then discrete actions); empty for
            regimes without combo dimensions.

    Returns:
        Carry with an ascending unit-interval grid and all-zero value and
        marginal-utility rows, broadcast over `leading_shape`.

    """
    dtype = canonical_float_dtype()
    shape = (*leading_shape, n_rows)
    return EGMCarry(
        endog_grid=jnp.broadcast_to(jnp.linspace(0.0, 1.0, n_rows, dtype=dtype), shape),
        value=jnp.zeros(shape, dtype=dtype),
        marginal_utility=jnp.zeros(shape, dtype=dtype),
        taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
    )


def shard_carry_template(
    *,
    template: EGMCarry,
    grids: Mapping[StateOrActionName, Any],
    leading_axis_names: tuple[StateName, ...],
    devices: tuple[jax.Device, ...],
) -> EGMCarry:
    """Place a carry template on the same device sharding as runtime carries.

    The compiled cores accept one carry pytree layout across all periods. A
    carry publisher computes its runtime rows alongside the regime's sharded
    value array, so the published carry inherits the state sharding on its
    leading axes; the template — the compile-time lowering sample and the
    first backward-induction input — must carry that sharding too, or the
    cores compile for replicated carries and reject every runtime period.

    Consumed by every carry-template producer with state-shaped leading axes:

    - the NB-EGM ride-along template (leading axes = ride-along states);
    - the living-brute child template (leading axes = discrete then passive
      continuous states);
    - the terminal-wealth template (same leading-axes layout).

    Leading axes follow `leading_axis_names` order; trailing row axes stay
    unsharded. Scalars replicate across the mesh. `devices` are the devices the
    placement assigned to the regime that publishes the carry, so the
    template's mesh is the one its value array runs on.

    A regime no state grid of which is distributed has no mesh, and its
    template is committed to the one device the placement gave it — the same
    rule its value array follows. Where that device is the one a single-device
    solve would have used anyway, the template keeps the default placement.
    """
    from _lcm.engine import _build_regime_sharding  # noqa: PLC0415

    plan = _build_regime_sharding(grids=MappingProxyType(dict(grids)), devices=devices)
    if plan is None:
        return _place_carry_template_on_one_device(template=template, devices=devices)
    if not any(name in plan.distributed_state_names for name in leading_axis_names):
        return template
    leading_spec = jax.NamedSharding(
        plan.mesh,
        jax.P(
            *(
                name if name in plan.distributed_state_names else None
                for name in leading_axis_names
            )
        ),
    )
    scalar_spec = jax.NamedSharding(plan.mesh, jax.P())
    return jax.tree.map(
        functools.partial(
            _place_carry_leaf, scalar_spec=scalar_spec, leading_spec=leading_spec
        ),
        template,
    )


def _place_carry_template_on_one_device(
    *, template: EGMCarry, devices: tuple[jax.Device, ...]
) -> EGMCarry:
    """Commit an unsharded template to the single device its regime runs on.

    The default placement is kept where a single-device solve would have put
    the template there anyway, which is what `placed_V_sharding` does for the
    same regime's value array.
    """
    from _lcm.execution.execution_plan import visible_devices  # noqa: PLC0415

    visible = visible_devices()
    if devices == (visible[0],) or len(devices) == len(visible):
        return template
    return jax.device_put(template, jax.sharding.SingleDeviceSharding(devices[0]))


# keyword-only-exempt: library-callback=jax.tree.map
def _place_carry_leaf(
    leaf: FloatND,
    *,
    scalar_spec: jax.NamedSharding,
    leading_spec: jax.NamedSharding,
) -> FloatND:
    """Place one carry leaf: a scalar replicates, a row takes the leading sharding."""
    return jax.device_put(leaf, scalar_spec if leaf.ndim == 0 else leading_spec)
