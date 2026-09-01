"""Logical-output layout planning for distributed solve kernels.

The solver names what each output *means*.  The engine, which owns the regime
topology and device mesh, resolves those roles to concrete shardings.  This is
deliberately narrower than an execution configuration: it plans output
placement for kernels that declare logical output roles.
"""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, cast, runtime_checkable

import jax

from _lcm.typing import StateName


class OutputRole(Enum):
    """Logical role of one solve-core output leaf."""

    VALUE = auto()
    DISSOLUTION_FLAG = auto()


VALUE = OutputRole.VALUE
DISSOLUTION_FLAG = OutputRole.DISSOLUTION_FLAG


@runtime_checkable
class OutputLayoutAware(Protocol):
    """A period kernel that declares the logical tree of a named core's output."""

    def output_roles(self, *, core_key: str) -> object:
        """Return a pytree of :class:`OutputRole` leaves for ``core_key``."""
        ...

    def core_for_output_layout(self, *, core_key: str) -> Callable:
        """Return the callable to lower with the resolved output shardings."""
        ...


class _Unplanned(Enum):
    TOKEN = auto()


UNPLANNED = _Unplanned.TOKEN
# No explicit output plan; preserve the backend-selected layout.


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

    expected_dissolution_shape: tuple[int, ...] | None
    """Absolute D shape for a collective output, otherwise ``None``."""

    expected_dissolution_dtype: object | None
    """Boolean D dtype for a collective output, otherwise ``None``."""


def resolve_output_layout(
    *,
    kernel: object,
    core_key: str,
    value_template: object,
    state_order: tuple[StateName, ...],
) -> ResolvedOutputLayout | _Unplanned:
    """Resolve an opted-in core's logical output roles on the V-template mesh.

    An unsharded regime has nothing to plan.  A distributed collective value
    has one trailing, replicated stakeholder axis; its dissolution flag is
    state-valued, so its spec is derived from the canonical state-axis prefix.
    """
    if not isinstance(kernel, OutputLayoutAware):
        return UNPLANNED

    value_sharding = getattr(value_template, "sharding", None)
    if not isinstance(value_sharding, jax.NamedSharding):
        return UNPLANNED

    roles = kernel.output_roles(core_key=core_key)
    if roles is None:
        # An engine-owned decorator may wrap an unaware custom kernel.  Its
        # method exists for structural delegation, but no plan exists to pass
        # through for that core.
        return UNPLANNED
    _validate_output_roles(
        roles=roles,
        core_key=core_key,
        value_template=value_template,
        state_order=state_order,
    )

    # PartitionSpec may omit replicated trailing axes.  Extend it to the
    # canonical state rank, then deliberately discard anything beyond that
    # prefix (the collective stakeholder axis).
    template_spec = tuple(value_sharding.spec)
    if len(template_spec) > len(state_order) and any(
        axis is not None for axis in template_spec[len(state_order) :]
    ):
        msg = (
            "The value template shards a non-state trailing output axis; only a "
            "replicated stakeholder axis can be dropped for dissolution output."
        )
        raise ValueError(msg)
    state_spec = (*template_spec[: len(state_order)],) + (None,) * max(
        0, len(state_order) - len(template_spec)
    )
    dissolution_sharding = jax.NamedSharding(
        mesh=value_sharding.mesh,
        spec=jax.P(*state_spec),
        memory_kind=value_sharding.memory_kind,
    )

    def _resolve(role: OutputRole) -> jax.NamedSharding:
        if role is VALUE:
            return value_sharding
        if role is DISSOLUTION_FLAG:
            return dissolution_sharding
        msg = f"unreachable output role: {role!r}"
        raise AssertionError(msg)

    out_shardings = jax.tree.map(_resolve, roles)
    tree = jax.tree.structure(out_shardings)
    leaves = tuple(jax.tree.leaves(out_shardings))
    value_shape = getattr(value_template, "shape", None)
    value_dtype = getattr(value_template, "dtype", None)
    if value_shape is None or value_dtype is None:
        msg = "A planned value template must expose an absolute shape and dtype."
        raise TypeError(msg)
    expected_value_shape = tuple(int(size) for size in value_shape)
    expected_value_dtype = value_dtype
    collective = roles == (VALUE, DISSOLUTION_FLAG)
    expected_dissolution_shape = expected_value_shape[:-1] if collective else None
    expected_dissolution_dtype = jax.numpy.dtype(bool) if collective else None
    return ResolvedOutputLayout(
        out_shardings=out_shardings,
        compilation_key=(
            tree,
            leaves,
            expected_value_shape,
            expected_value_dtype,
            expected_dissolution_shape,
            expected_dissolution_dtype,
        ),
        expected_value_shape=expected_value_shape,
        expected_value_dtype=expected_value_dtype,
        expected_dissolution_shape=expected_dissolution_shape,
        expected_dissolution_dtype=expected_dissolution_dtype,
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
    unknown = [role for role in role_leaves if not isinstance(role, OutputRole)]
    if unknown:
        msg = (
            f"Core {core_key!r} declared non-OutputRole leaves in its output-role "
            f"tree: {unknown!r}."
        )
        raise TypeError(msg)
    if roles is not VALUE and roles != (VALUE, DISSOLUTION_FLAG):
        msg = (
            f"Core {core_key!r} declared unsupported output roles {roles!r}. "
            "Supported role trees are exactly VALUE or "
            "(VALUE, DISSOLUTION_FLAG)."
        )
        raise ValueError(msg)
    if (
        roles == (VALUE, DISSOLUTION_FLAG)
        and getattr(value_template, "ndim", None) != len(state_order) + 1
    ):
        msg = (
            f"Core {core_key!r} declared a dissolution output, but its value "
            "template does not have exactly one trailing stakeholder axis."
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

    collective = layout.expected_dissolution_shape is not None
    if collective:
        value, dissolution = cast("tuple[object, object]", output)
    else:
        value = output
        dissolution = None

    _assert_output_metadata(
        output=value,
        label="value",
        expected_shape=layout.expected_value_shape,
        expected_dtype=layout.expected_value_dtype,
    )
    if collective:
        assert layout.expected_dissolution_shape is not None  # noqa: S101
        assert layout.expected_dissolution_dtype is not None  # noqa: S101
        _assert_output_metadata(
            output=dissolution,
            label="dissolution",
            expected_shape=layout.expected_dissolution_shape,
            expected_dtype=layout.expected_dissolution_dtype,
        )
        if getattr(dissolution, "shape", None) != getattr(value, "shape", ())[:-1]:
            msg = "planned dissolution shape must equal V.shape[:-1]"
            raise AssertionError(msg)

    output_with_paths, _ = jax.tree_util.tree_flatten_with_path(output)
    expected_leaves = jax.tree.leaves(layout.out_shardings)
    for (path, leaf), expected in zip(output_with_paths, expected_leaves, strict=True):
        actual = getattr(leaf, "sharding", None)
        if actual != expected:
            msg = (
                f"planned output sharding mismatch at {jax.tree_util.keystr(path)}: "
                f"expected {expected}, got {actual}. The output must be born in "
                "its planned layout; post-run repair is not permitted."
            )
            raise AssertionError(msg)


def _assert_output_metadata(
    *,
    output: object,
    label: str,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
) -> None:
    """Check one planned leaf against absolute template shape and dtype."""
    actual_shape = getattr(output, "shape", None)
    if actual_shape != expected_shape:
        msg = (
            f"planned {label} shape mismatch: expected {expected_shape}, "
            f"got {actual_shape}"
        )
        raise AssertionError(msg)
    actual_dtype = getattr(output, "dtype", None)
    if actual_dtype != expected_dtype:
        msg = (
            f"planned {label} dtype mismatch: expected {expected_dtype}, "
            f"got {actual_dtype}"
        )
        raise AssertionError(msg)


@dataclass(frozen=True, kw_only=True)
class PlannedCore:
    """Callable compiled core carrying the output plan used to lower it."""

    compiled: Callable
    layout: ResolvedOutputLayout

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Execute and enforce the layout contract at the compiled-core seam."""
        output = self.compiled(*args, **kwargs)
        assert_output_layout(output=output, layout=self.layout)
        return output


def planned_output_layout(core: object) -> ResolvedOutputLayout | _Unplanned:
    """Return the plan attached to a compiled core, if any."""
    return core.layout if isinstance(core, PlannedCore) else UNPLANNED
