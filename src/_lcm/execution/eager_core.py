"""Place eager inputs and mapped computations before executing a numerical core."""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import jax
import jax.numpy as jnp

from _lcm.execution.core_program import ResolvedCoreProgram
from _lcm.execution.runtime_sharding import runtime_shardings_match
from lcm.exceptions import ExecutionPlanningError


def make_eager_core(
    *,
    program: ResolvedCoreProgram,
    execution_sharding: jax.sharding.Sharding,
    internal_input_templates: Mapping[str, object] = MappingProxyType({}),
) -> Callable[..., object]:
    """Bind the same function and static widths, retaining only input descriptors."""
    if any(
        not isinstance(leaf, jax.ShapeDtypeStruct)
        or not isinstance(leaf.sharding, jax.sharding.Sharding)
        for leaf in jax.tree.leaves(program.arguments)
    ):
        raise ExecutionPlanningError("An eager core requires abstract input layouts.")
    if (
        internal_input_templates.keys() != program.requirements.internal_inputs.keys()
        or internal_input_templates.keys() & program.arguments.keys()
        or any(
            not isinstance(leaf, jax.ShapeDtypeStruct)
            or (
                leaf.sharding is not None
                and not isinstance(leaf.sharding, jax.sharding.Sharding)
            )
            for leaf in jax.tree.leaves(internal_input_templates)
        )
    ):
        raise ExecutionPlanningError(
            "An eager core requires exactly its declared abstract internal inputs."
        )
    mesh = (
        jax.sharding.Mesh(
            execution_sharding.mesh.devices,
            execution_sharding.mesh.axis_names,
            axis_types=(jax.sharding.AxisType.Explicit,)
            * len(execution_sharding.mesh.axis_names),
        )
        if isinstance(execution_sharding, jax.NamedSharding)
        else None
    )
    devices = (
        tuple(mesh.devices.flat)
        if mesh is not None
        else tuple(execution_sharding.device_set)
    )
    return _EagerCore(
        function=functools.partial(program.function, **program.static_kwargs),
        arguments=MappingProxyType(dict(program.arguments)),
        internal_input_templates=MappingProxyType(dict(internal_input_templates)),
        device=devices[0],
        mesh=mesh,
    )


@dataclass(frozen=True, kw_only=True)
class _EagerCore:
    """A call-local eager placement context with no retained concrete arguments."""

    function: Callable[..., object]
    """Original numerical body with its declared static widths bound."""
    arguments: Mapping[str, object]
    """Input descriptor tree; no concrete runtime operands are retained."""
    internal_input_templates: Mapping[str, object]
    """Producer metadata; absent sharding preserves the guarded producer layout."""
    device: jax.Device
    """Default device for eager constants in this core."""
    mesh: jax.sharding.Mesh | None
    """Equivalent Explicit mesh for mapped-axis propagation, when sharded."""

    def __call__(self, **arguments: object) -> object:
        """Place operands first and return the numerical function's exact tree."""
        placement = _EagerPlacement(core=self)
        try:
            placed = jax.tree.map(
                placement,
                {
                    name: value
                    for name, value in arguments.items()
                    if name not in self.internal_input_templates
                },
                dict(self.arguments),
            )
            placed.update(
                jax.tree.map(
                    placement.internal,
                    {name: arguments[name] for name in self.internal_input_templates},
                    dict(self.internal_input_templates),
                )
            )
            with jax.default_device(self.device):
                if self.mesh is not None:
                    with jax.set_mesh(self.mesh):
                        return self.function(**placed)
                return self.function(**placed)
        finally:
            placement.results.clear()

    def place_operand(
        self, *, value: object, template: jax.ShapeDtypeStruct
    ) -> jax.Array:
        """Preserve committed placement and give ordinary operands their layout."""
        expected = template.sharding
        if not isinstance(expected, jax.sharding.Sharding):
            raise ExecutionPlanningError("An eager operand has no planned sharding.")
        if isinstance(value, jax.Array) and value.committed:
            if not runtime_shardings_match(
                actual=value.sharding, expected=expected, ndim=value.ndim
            ):
                raise ExecutionPlanningError(
                    "A committed eager operand changed its planned physical layout."
                )
            expected = value.sharding
        sharding = self._typed_sharding(sharding=expected)
        placed = (
            value
            if isinstance(value, jax.Array)
            and value.committed
            and value.sharding == sharding
            else jax.device_put(value, sharding)
        )
        # AOT execution accepts a weak runtime value at a declared strong input
        # and uses the declared type for promotion. Establish that same eager
        # binding explicitly; this allocates a new array and never repairs a
        # wrong shape/dtype or manufactures a weak value from a strong input.
        if (
            placed.weak_type
            and not template.weak_type
            and placed.shape == template.shape
            and placed.dtype == template.dtype
        ):
            placed = jnp.asarray(placed, dtype=template.dtype)
            if not runtime_shardings_match(
                actual=placed.sharding, expected=sharding, ndim=placed.ndim
            ):
                raise ExecutionPlanningError(
                    "A normalized eager operand changed its planned physical layout."
                )
        return placed

    def _typed_sharding(
        self, *, sharding: jax.sharding.Sharding
    ) -> jax.sharding.Sharding:
        """Expose an existing physical mesh to eager mapped-axis propagation."""
        if self.mesh is None or not isinstance(sharding, jax.NamedSharding):
            return sharding
        if (
            sharding.mesh.devices.shape != self.mesh.devices.shape
            or sharding.mesh.axis_names != self.mesh.axis_names
            or tuple(sharding.mesh.devices.flat) != tuple(self.mesh.devices.flat)
        ):
            return sharding
        return jax.NamedSharding(
            self.mesh, sharding.spec, memory_kind=sharding.memory_kind
        )


@dataclass(frozen=True, kw_only=True)
class _EagerPlacement:
    """Transient tree callback; its concrete memo is cleared at every call exit."""

    core: _EagerCore
    """Immutable numerical callable and layout metadata for this invocation."""
    results: dict[tuple[int, jax.ShapeDtypeStruct], jax.Array] = field(
        default_factory=dict
    )
    """One placement per original identity and complete destination descriptor."""

    # keyword-only-exempt: library-callback=jax.tree.map
    def internal(self, value: object, template: jax.ShapeDtypeStruct) -> jax.Array:
        """Use an internal producer's actual guarded layout when tracing omitted it.

        PlannedCore checks the producer's output placement before its consumer
        runs. The abstract trace supplies shape, dtype and weak typing, but may
        omit sharding. Only these declared producer inputs use their live layout;
        ordinary arguments always require an explicit planned input layout.
        """
        if template.sharding is None:
            if not isinstance(value, jax.Array):
                raise ExecutionPlanningError(
                    "An eager internal input requires an actual JAX producer array."
                )
            template = jax.ShapeDtypeStruct(
                template.shape,
                template.dtype,
                weak_type=template.weak_type,
                sharding=value.sharding,
            )
        return self(value, template)

    # keyword-only-exempt: library-callback=jax.tree.map
    def __call__(self, value: object, template: jax.ShapeDtypeStruct) -> jax.Array:
        """Share placements while validating each occurrence's numerical metadata."""
        key = (id(value), template)
        if key not in self.results:
            self.results[key] = self.core.place_operand(value=value, template=template)
        placed = self.results[key]
        if (
            placed.shape != template.shape
            or placed.dtype != template.dtype
            or placed.weak_type != template.weak_type
        ):
            raise ExecutionPlanningError(
                "An eager operand changed its planned shape, dtype or weak typing."
            )
        return placed
