"""Logical-output layout planning for distributed solve kernels.

The solver names what each output *means*.  The engine, which owns the regime
topology and device mesh, resolves those roles to concrete shardings before lowering.
Every planned output layout comes from the owning core program's declared roles.

A role tree is any pytree whose leaves are roles; a `None` entry marks an output the
program does not publish. Exactly one leaf is `VALUE`, and it is the first leaf: the
solve loop reads the period value from that position without knowing the rest of
the tree.
"""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.execution.internal_outputs import assert_internal_inputs
from _lcm.execution.value_transfer import (
    ResolvedValueTransfer,
    apply_value_transfer_plan,
)
from _lcm.typing import StateName


class OutputRole(Enum):
    """Logical role of one solve-core output leaf fixed by the value template."""

    VALUE = auto()
    DISSOLUTION_FLAG = auto()


VALUE = OutputRole.VALUE
DISSOLUTION_FLAG = OutputRole.DISSOLUTION_FLAG


@dataclass(frozen=True, kw_only=True)
class StateAxesLeading:
    """Role of an output leaf whose leading axes are named state axes.

    The leaf is placed on the value template's placement of the named state axes,
    in the given order, after `n_free_leading_axes` replicated leading axes; every
    later axis is replicated. A 0-d leaf names no state. `dtype` and `shape` are
    checked against the lowered and the executed leaf only when declared.
    """

    state_names: tuple[StateName, ...]
    """State axes leading the leaf, in the leaf's own axis order."""

    n_free_leading_axes: int = 0
    """Replicated axes in front of the state axes (a branch or candidate axis)."""

    dtype: object | None = None
    """Exact dtype the leaf must have, or `None` to leave it unchecked."""

    shape: tuple[int, ...] | None = None
    """Absolute shape the leaf must have, or `None` to leave it unchecked."""

    def __post_init__(self) -> None:
        """Snapshot the declaration and normalize the optional metadata."""
        object.__setattr__(self, "state_names", tuple(self.state_names))
        if len(set(self.state_names)) != len(self.state_names):
            msg = f"StateAxesLeading names a state twice: {self.state_names!r}."
            raise ValueError(msg)
        if (
            isinstance(self.n_free_leading_axes, bool)
            or not isinstance(self.n_free_leading_axes, int)
            or self.n_free_leading_axes < 0
        ):
            msg = (
                "StateAxesLeading n_free_leading_axes must be a non-negative "
                f"integer; got {self.n_free_leading_axes!r}."
            )
            raise TypeError(msg)
        if self.dtype is not None:
            object.__setattr__(self, "dtype", jnp.dtype(self.dtype))
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(int(size) for size in self.shape))


type OutputRoleLeaf = OutputRole | StateAxesLeading


@dataclass(frozen=True, kw_only=True)
class ExpectedOutputLeaf:
    """Absolute contract one resolved output leaf must satisfy."""

    label: str
    """Leaf name in messages: `value`, `dissolution`, or its pytree path."""

    shape: tuple[int, ...] | None
    """Absolute shape, or `None` when the role leaves it unchecked."""

    dtype: object | None
    """Exact dtype, or `None` when the role leaves it unchecked."""

    sharding: jax.sharding.Sharding
    """Placement the leaf is born in."""


@dataclass(frozen=True, kw_only=True)
class ResolvedOutputLayout:
    """Concrete output shardings and their AOT compilation identity."""

    out_shardings: object
    """Pytree accepted by ``jax.jit(..., out_shardings=...)``."""

    compilation_key: Hashable
    """Hashable layout identity paired with the callable deduplication key."""

    expected_value_shape: tuple[int, ...]
    """Absolute shape of the V leaf captured from the regime template."""

    expected_value_dtype: object
    """Exact dtype of the V leaf captured from the regime template."""

    expected_leaves: tuple[ExpectedOutputLeaf, ...]
    """Per-leaf contracts in pytree leaf order; the V leaf is first."""


def resolve_output_layout(
    *,
    core_key: str,
    value_template: object,
    state_order: tuple[StateName, ...],
    output_roles: object,
) -> ResolvedOutputLayout:
    """Resolve one program-owned output-role tree on the V-template placement.

    Native :class:`CoreProgram` declarations receive a concrete contract even on one
    device. The value leaf takes the template's own placement. A dissolution flag is
    state-valued, so it is placed on the canonical state-axis prefix. A
    :class:`StateAxesLeading` leaf is placed on the template's placement of the
    states it names. Every program the engine lowers enters this resolver.
    """
    if output_roles is None:
        msg = "Explicit CoreProgram output_roles cannot be None."
        raise ValueError(msg)
    _validate_output_roles(
        roles=output_roles,
        core_key=core_key,
        value_template=value_template,
        state_order=state_order,
    )

    value_sharding = _require_value_sharding(value_template=value_template)
    value_shape = getattr(value_template, "shape", None)
    value_dtype = getattr(value_template, "dtype", None)
    if value_shape is None or value_dtype is None:
        msg = "A planned value template must expose an absolute shape and dtype."
        raise TypeError(msg)
    expected_value_shape = tuple(int(size) for size in value_shape)
    expected_value_dtype = value_dtype
    state_spec = _state_axis_spec(
        value_sharding=value_sharding, state_order=state_order
    )

    roles_with_paths, tree = jax.tree_util.tree_flatten_with_path(output_roles)
    expected_leaves = tuple(
        _resolve_output_leaf(
            path=path,
            role=role,
            value_sharding=value_sharding,
            value_shape=expected_value_shape,
            value_dtype=expected_value_dtype,
            state_order=state_order,
            state_spec=state_spec,
        )
        for path, role in roles_with_paths
    )
    out_shardings = jax.tree.unflatten(
        tree, [leaf.sharding for leaf in expected_leaves]
    )
    return ResolvedOutputLayout(
        out_shardings=out_shardings,
        compilation_key=(tree, expected_leaves),
        expected_value_shape=expected_value_shape,
        expected_value_dtype=expected_value_dtype,
        expected_leaves=expected_leaves,
    )


def _require_value_sharding(*, value_template: object) -> jax.sharding.Sharding:
    """Return the concrete sharding required by a program-owned output contract."""
    value_sharding = getattr(value_template, "sharding", None)
    if not isinstance(value_sharding, jax.sharding.Sharding):
        msg = (
            "An explicit CoreProgram output contract requires a JAX value "
            "template with concrete output sharding."
        )
        raise TypeError(msg)
    return value_sharding


def _state_axis_spec(
    *,
    value_sharding: jax.sharding.Sharding,
    state_order: tuple[StateName, ...],
) -> tuple[object, ...] | None:
    """Read the template's partition entry per canonical state axis.

    Returns `None` off a named mesh: a single-device contract still fixes where
    every output leaf is born, and the value sharding itself is that place.
    """
    if not isinstance(value_sharding, jax.NamedSharding):
        return None

    # PartitionSpec may omit replicated trailing axes. Extend it to the
    # canonical state rank, then deliberately discard anything beyond that
    # prefix (the collective stakeholder axis).
    template_spec = tuple(value_sharding.spec)
    if len(template_spec) > len(state_order) and any(
        axis is not None for axis in template_spec[len(state_order) :]
    ):
        msg = (
            "The value template shards a non-state trailing output axis; only a "
            "replicated stakeholder axis can follow the state axes."
        )
        raise ValueError(msg)
    return (*template_spec[: len(state_order)],) + (None,) * max(
        0, len(state_order) - len(template_spec)
    )


def _resolve_output_leaf(
    *,
    path: tuple[object, ...],
    role: OutputRoleLeaf,
    value_sharding: jax.sharding.Sharding,
    value_shape: tuple[int, ...],
    value_dtype: object,
    state_order: tuple[StateName, ...],
    state_spec: tuple[object, ...] | None,
) -> ExpectedOutputLeaf:
    """Map one validated logical role to its concrete contract."""
    if role is VALUE:
        return ExpectedOutputLeaf(
            label="value",
            shape=value_shape,
            dtype=value_dtype,
            sharding=value_sharding,
        )
    if role is DISSOLUTION_FLAG:
        label = "dissolution"
        role = StateAxesLeading(
            state_names=state_order, dtype=jnp.dtype(bool), shape=value_shape[:-1]
        )
    else:
        label = f"leaf {jax.tree_util.keystr(path)}"
    if not isinstance(role, StateAxesLeading):
        msg = f"unreachable output role: {role!r}"
        raise TypeError(msg)
    return ExpectedOutputLeaf(
        label=label,
        shape=role.shape,
        dtype=role.dtype,
        sharding=_state_axes_leading_sharding(
            role=role,
            value_sharding=value_sharding,
            state_order=state_order,
            state_spec=state_spec,
        ),
    )


def _state_axes_leading_sharding(
    *,
    role: StateAxesLeading,
    value_sharding: jax.sharding.Sharding,
    state_order: tuple[StateName, ...],
    state_spec: tuple[object, ...] | None,
) -> jax.sharding.Sharding:
    """Place the named state axes as the template places them; replicate the rest."""
    if state_spec is None:
        return value_sharding
    if not isinstance(value_sharding, jax.NamedSharding):
        msg = f"unreachable value sharding: {value_sharding!r}"
        raise TypeError(msg)
    spec = (None,) * role.n_free_leading_axes + tuple(
        state_spec[state_order.index(name)] for name in role.state_names
    )
    return jax.NamedSharding(
        mesh=value_sharding.mesh,
        spec=jax.P(*spec),
        memory_kind=value_sharding.memory_kind,
    )


def _validate_output_roles(
    *,
    roles: object,
    core_key: str,
    value_template: object,
    state_order: tuple[StateName, ...],
) -> None:
    """Fail closed outside the supported logical output trees."""
    role_leaves = jax.tree.leaves(roles)
    if not role_leaves:
        msg = f"Core {core_key!r} declared an empty output-role tree."
        raise ValueError(msg)
    unknown = [
        role
        for role in role_leaves
        if not isinstance(role, OutputRole | StateAxesLeading)
    ]
    if unknown:
        msg = (
            f"Core {core_key!r} declared non-role leaves in its output-role "
            f"tree: {unknown!r}."
        )
        raise TypeError(msg)
    value_positions = [index for index, role in enumerate(role_leaves) if role is VALUE]
    if value_positions != [0]:
        msg = (
            f"Core {core_key!r} must declare exactly one VALUE leaf, in the first "
            f"position of its output-role tree; got {roles!r}."
        )
        raise ValueError(msg)
    if (
        any(role is DISSOLUTION_FLAG for role in role_leaves)
        and getattr(value_template, "ndim", None) != len(state_order) + 1
    ):
        msg = (
            f"Core {core_key!r} declared a dissolution output, but its value "
            "template does not have exactly one trailing stakeholder axis."
        )
        raise ValueError(msg)
    for role in role_leaves:
        if not isinstance(role, StateAxesLeading):
            continue
        outside = [name for name in role.state_names if name not in state_order]
        if outside:
            msg = (
                f"Core {core_key!r} output role {role!r} names states outside the "
                f"value template's state order {state_order!r}: {outside!r}."
            )
            raise ValueError(msg)


def assert_output_layout(*, output: object, layout: ResolvedOutputLayout) -> None:
    """Assert that a planned core output was born in its requested layout."""
    output_tree = jax.tree.structure(output)
    planned_tree = jax.tree.structure(layout.out_shardings)
    if output_tree != planned_tree:
        msg = (
            "planned output tree does not match the runtime output tree: "
            f"expected {planned_tree}, got {output_tree}"
        )
        raise AssertionError(msg)

    output_with_paths, _ = jax.tree_util.tree_flatten_with_path(output)
    for (path, leaf), expected in zip(
        output_with_paths, layout.expected_leaves, strict=True
    ):
        _assert_output_leaf(
            output=leaf, path=jax.tree_util.keystr(path), expected=expected
        )


def assert_value_leaf_layout(*, value: object, layout: ResolvedOutputLayout) -> None:
    """Assert that a published period value is the layout's first leaf."""
    _assert_output_leaf(output=value, path="[0]", expected=layout.expected_leaves[0])


def _assert_output_leaf(
    *,
    output: object,
    path: str,
    expected: ExpectedOutputLeaf,
) -> None:
    """Check one executed leaf against its declared shape, dtype, and placement."""
    _assert_output_metadata(
        output=output,
        label=expected.label,
        expected_shape=expected.shape,
        expected_dtype=expected.dtype,
    )
    actual = getattr(output, "sharding", None)
    if actual != expected.sharding:
        msg = (
            f"planned output sharding mismatch at {path} ({expected.label}): "
            f"expected {expected.sharding}, got {actual}. The output must be born "
            "in its planned layout; post-run repair is not permitted."
        )
        raise AssertionError(msg)


def _assert_output_metadata(
    *,
    output: object,
    label: str,
    expected_shape: tuple[int, ...] | None,
    expected_dtype: object | None,
) -> None:
    """Check one planned leaf against the absolute metadata its role declares."""
    if expected_shape is not None:
        actual_shape = getattr(output, "shape", None)
        if actual_shape != expected_shape:
            msg = (
                f"planned {label} shape mismatch: expected {expected_shape}, "
                f"got {actual_shape}"
            )
            raise AssertionError(msg)
    if expected_dtype is not None:
        actual_dtype = getattr(output, "dtype", None)
        if actual_dtype != expected_dtype:
            msg = (
                f"planned {label} dtype mismatch: expected {expected_dtype}, "
                f"got {actual_dtype}"
            )
            raise AssertionError(msg)


@dataclass(frozen=True, kw_only=True)
class PlannedCore:
    """Callable compiled core carrying the output and input plans used to lower it."""

    compiled: Callable
    layout: ResolvedOutputLayout
    tile_widths: Mapping[str, int]
    input_transfer_plan: tuple[ResolvedValueTransfer, ...] = ()
    internal_input_templates: Mapping[str, object] = MappingProxyType({})
    """Abstract template per internal input this core was lowered against."""
    name: str = ""
    """Graph key of the program this core was compiled for."""

    def __post_init__(self) -> None:
        """Snapshot the exact lowering widths and resolved input transfer plan."""
        widths = dict(self.tile_widths)
        if any(not isinstance(name, str) or not name for name in widths):
            msg = "PlannedCore tile-width names must be non-empty strings."
            raise TypeError(msg)
        if any(type(width) is not int for width in widths.values()):
            msg = "PlannedCore tile widths must be integers."
            raise TypeError(msg)
        if any(width <= 0 for width in widths.values()):
            msg = "PlannedCore tile widths must be positive."
            raise ValueError(msg)
        object.__setattr__(self, "tile_widths", MappingProxyType(widths))

        plan = tuple(self.input_transfer_plan)
        if any(not isinstance(item, ResolvedValueTransfer) for item in plan):
            msg = "PlannedCore input_transfer_plan must contain resolved transfers."
            raise TypeError(msg)
        object.__setattr__(self, "input_transfer_plan", plan)

        templates = dict(self.internal_input_templates)
        if any(not isinstance(name, str) or not name for name in templates):
            msg = "PlannedCore internal-input names must be non-empty strings."
            raise TypeError(msg)
        object.__setattr__(
            self, "internal_input_templates", MappingProxyType(templates)
        )

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute and enforce the layout contract at the compiled-core seam."""
        if self.internal_input_templates:
            assert_internal_inputs(
                arguments=kwargs,
                templates=self.internal_input_templates,
                label=self.name,
            )
        if self.input_transfer_plan and args:
            msg = (
                "A PlannedCore with input transfers accepts dynamic arguments "
                "only by keyword."
            )
            raise TypeError(msg)
        planned_kwargs = (
            apply_value_transfer_plan(arguments=kwargs, plan=self.input_transfer_plan)
            if self.input_transfer_plan
            else kwargs
        )
        output = self.compiled(*args, **planned_kwargs)
        assert_output_layout(output=output, layout=self.layout)
        return output
