"""Tests for logical-output layout planning."""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
    core_program_graph,
)
from _lcm.execution.output_layout import (
    DISSOLUTION_FLAG,
    UNPLANNED,
    VALUE,
    ResolvedOutputLayout,
    assert_output_layout,
    resolve_output_layout,
)
from _lcm.regime_building.processing import _TerminalCarryPeriodKernel
from _lcm.solution.backward_induction import _lowering_key, _publish_kernel_value


@dataclass(frozen=True)
class _GraphKernel:
    program: CoreProgram

    def core_programs(self):
        return {"main": self.program}


def _legacy_core() -> None:
    """Stand in for an unmigrated solver core."""


class _LegacyKernel:
    """Publish only the interface consumed by the central legacy adapter."""

    def cores(self) -> Mapping[str, Callable[..., object]]:
        return {"main": _legacy_core}

    def build_lower_args(
        self, *, core_key: str, **_context: object
    ) -> Mapping[str, object]:
        assert core_key == "main"
        return {}


def _mesh() -> jax.sharding.Mesh:
    return jax.sharding.Mesh(np.asarray(jax.devices()), ("kind",))


def _template(*, collective: bool = False):
    shape = (len(jax.devices()), 3, 2) if collective else (len(jax.devices()), 3)
    spec = jax.P("kind", None, None) if collective else jax.P("kind", None)
    return jax.device_put(
        jnp.zeros(shape),
        jax.NamedSharding(mesh=_mesh(), spec=spec),
    )


def test_resolver_maps_singleton_value_to_exact_template_sharding():
    template = _template()

    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )

    assert isinstance(resolved, ResolvedOutputLayout)
    assert resolved.out_shardings == template.sharding
    assert resolved.expected_value_shape == template.shape
    assert resolved.expected_value_dtype == template.dtype
    assert resolved.expected_dissolution_shape is None
    assert resolved.expected_dissolution_dtype is None


def test_resolver_drops_collective_stakeholder_axis_from_dissolution_spec():
    template = _template(collective=True)

    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=(VALUE, DISSOLUTION_FLAG),
    )

    assert isinstance(resolved, ResolvedOutputLayout)
    value_sharding, dissolution_sharding = cast(
        "tuple[jax.NamedSharding, jax.NamedSharding]", resolved.out_shardings
    )
    assert value_sharding == template.sharding
    assert dissolution_sharding.mesh == template.sharding.mesh
    assert dissolution_sharding.spec == jax.P("kind", None)
    assert resolved.expected_value_shape == template.shape
    assert resolved.expected_value_dtype == template.dtype
    assert resolved.expected_dissolution_shape == template.shape[:-1]
    assert resolved.expected_dissolution_dtype == jnp.bool_


def test_collective_template_may_omit_replicated_stakeholder_spec_entry():
    template = jax.device_put(
        jnp.zeros((len(jax.devices()), 3, 2)),
        jax.NamedSharding(mesh=_mesh(), spec=jax.P("kind", None)),
    )

    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=(VALUE, DISSOLUTION_FLAG),
    )

    assert isinstance(resolved, ResolvedOutputLayout)
    value_sharding, dissolution_sharding = cast(
        "tuple[jax.NamedSharding, jax.NamedSharding]", resolved.out_shardings
    )
    assert value_sharding == template.sharding
    assert dissolution_sharding.spec == jax.P("kind", None)


def test_explicit_roles_resolve_a_single_device_layout():
    template = jnp.zeros((2, 3))

    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )

    assert resolved.out_shardings == template.sharding


def test_terminal_carry_decorator_delegates_grid_search_program_roles():
    program = CoreProgram(
        name="main",
        function=lambda x: x,
        argument_builder=lambda _context: {},
        requirements=CoreExecutionRequirements(),
        output_roles=VALUE,
        disposition=CoreExecutionDisposition.DENSE,
        disposition_reason="test_dense_terminal_carry_layout",
    )
    wrapper = object.__new__(_TerminalCarryPeriodKernel)
    object.__setattr__(wrapper, "base", _GraphKernel(program=program))

    assert wrapper.core_programs()["main"] is program


def test_terminal_carry_decorator_rejects_unaware_base():
    wrapper = object.__new__(_TerminalCarryPeriodKernel)
    object.__setattr__(wrapper, "base", object())

    with pytest.raises(TypeError, match="native core-program graph"):
        wrapper.core_programs()


@pytest.mark.parametrize(
    "roles",
    [DISSOLUTION_FLAG, (VALUE, VALUE), (DISSOLUTION_FLAG, VALUE)],
)
def test_resolver_rejects_malformed_role_trees(roles):
    with pytest.raises(ValueError, match="exactly VALUE"):
        resolve_output_layout(
            core_key="main",
            value_template=_template(collective=True),
            state_order=("kind", "wealth"),
            output_roles=roles,
        )


def test_resolver_rejects_dissolution_without_one_stakeholder_axis():
    with pytest.raises(ValueError, match="trailing stakeholder axis"):
        resolve_output_layout(
            core_key="main",
            value_template=_template(),
            state_order=("kind", "wealth"),
            output_roles=(VALUE, DISSOLUTION_FLAG),
        )


def test_compilation_key_tracks_layout_tree_and_shardings():
    first = resolve_output_layout(
        core_key="main",
        value_template=_template(),
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )
    same = resolve_output_layout(
        core_key="main",
        value_template=_template(),
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )
    different = resolve_output_layout(
        core_key="main",
        value_template=_template(collective=True),
        state_order=("kind", "wealth"),
        output_roles=(VALUE, DISSOLUTION_FLAG),
    )

    assert isinstance(first, ResolvedOutputLayout)
    assert isinstance(same, ResolvedOutputLayout)
    assert isinstance(different, ResolvedOutputLayout)
    assert first.compilation_key == same.compilation_key
    assert first.compilation_key != different.compilation_key

    def func(x):
        return x

    def other_func(x):
        return x

    assert _lowering_key(func=func, layout_key=first.compilation_key) == _lowering_key(
        func=func, layout_key=same.compilation_key
    )
    assert _lowering_key(func=func, layout_key=first.compilation_key) != _lowering_key(
        func=func, layout_key=different.compilation_key
    )
    assert _lowering_key(func=func, layout_key=first.compilation_key) != _lowering_key(
        func=other_func, layout_key=first.compilation_key
    )


def test_lowering_key_tracks_positional_partial_bindings() -> None:
    def core(_static_policy: object, /) -> object:
        return _static_policy

    policy = object()
    first = functools.partial(core, policy)
    equivalent = functools.partial(core, policy)
    different = functools.partial(core, object())

    first_key = _lowering_key(func=first, layout_key=UNPLANNED)
    equivalent_key = _lowering_key(func=equivalent, layout_key=UNPLANNED)
    different_key = _lowering_key(func=different, layout_key=UNPLANNED)

    assert first_key == equivalent_key
    assert first_key != different_key


def test_role_tree_output_mismatch_fails_during_lowering():
    resolved = resolve_output_layout(
        core_key="main",
        value_template=_template(collective=True),
        state_order=("kind", "wealth"),
        output_roles=(VALUE, DISSOLUTION_FLAG),
    )
    assert isinstance(resolved, ResolvedOutputLayout)

    with pytest.raises(ValueError, match="out_shardings"):
        jax.jit(
            lambda x: x,
            out_shardings=resolved.out_shardings,
        ).lower(jnp.zeros((len(jax.devices()), 3)))


def test_assert_output_layout_rejects_post_run_repair_need():
    resolved = resolve_output_layout(
        core_key="main",
        value_template=_template(),
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )
    assert isinstance(resolved, ResolvedOutputLayout)
    expected = _template()

    assert_output_layout(output=expected, layout=resolved)
    replicated = jax.device_put(
        jnp.zeros(expected.shape),
        jax.NamedSharding(mesh=_mesh(), spec=jax.P()),
    )
    with pytest.raises(AssertionError, match="output sharding"):
        assert_output_layout(output=replicated, layout=resolved)


def test_assert_output_layout_rejects_wrong_absolute_value_shape():
    template = _template()
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )
    assert isinstance(resolved, ResolvedOutputLayout)
    wrong_shape = jax.device_put(
        jnp.zeros((len(jax.devices()), 4), dtype=template.dtype),
        template.sharding,
    )

    with pytest.raises(AssertionError, match="planned value shape"):
        assert_output_layout(output=wrong_shape, layout=resolved)


def test_assert_output_layout_rejects_wrong_absolute_value_dtype():
    template = _template()
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=VALUE,
    )
    assert isinstance(resolved, ResolvedOutputLayout)
    wrong_dtype = jax.device_put(
        jnp.zeros(template.shape, dtype=jnp.int32),
        template.sharding,
    )

    with pytest.raises(AssertionError, match="planned value dtype"):
        assert_output_layout(output=wrong_dtype, layout=resolved)


def test_assert_output_layout_rejects_coherent_wrong_collective_shape():
    template = _template(collective=True)
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=(VALUE, DISSOLUTION_FLAG),
    )
    assert isinstance(resolved, ResolvedOutputLayout)
    value_sharding, dissolution_sharding = cast(
        "tuple[jax.NamedSharding, jax.NamedSharding]", resolved.out_shardings
    )
    wrong_value = jax.device_put(
        jnp.zeros((len(jax.devices()), 2, 2), dtype=template.dtype),
        value_sharding,
    )
    wrong_dissolution = jax.device_put(
        jnp.zeros((len(jax.devices()), 2), dtype=bool),
        dissolution_sharding,
    )

    with pytest.raises(AssertionError, match="planned value shape"):
        assert_output_layout(
            output=(wrong_value, wrong_dissolution),
            layout=resolved,
        )


@pytest.mark.parametrize(
    "dissolution",
    [
        jnp.zeros((len(jax.devices()), 3)),
        jnp.zeros((len(jax.devices()), 2), dtype=bool),
    ],
)
def test_assert_output_layout_rejects_malformed_dissolution(dissolution):
    template = _template(collective=True)
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=(VALUE, DISSOLUTION_FLAG),
    )
    assert isinstance(resolved, ResolvedOutputLayout)

    with pytest.raises(AssertionError, match="dissolution"):
        assert_output_layout(output=(template, dissolution), layout=resolved)


def test_unplanned_legacy_kernel_retains_publication_repair():
    template = _template()
    replicated = jax.device_put(
        jnp.zeros(template.shape),
        jax.NamedSharding(mesh=_mesh(), spec=jax.P()),
    )

    graph = core_program_graph(kernel=_LegacyKernel())
    assert graph["main"].disposition is CoreExecutionDisposition.LEGACY_UNPLANNED

    published = _publish_kernel_value(
        value=replicated,
        dissolution=None,
        template=template,
        compiled_cores={"main": graph["main"].function},
    )

    assert published.sharding == template.sharding
