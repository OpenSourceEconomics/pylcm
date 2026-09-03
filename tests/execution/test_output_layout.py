"""Tests for logical-output layout planning."""

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
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
    StateAxesLeading,
    assert_output_layout,
    assert_value_leaf_layout,
    resolve_output_layout,
)
from _lcm.regime_building.processing import _TerminalCarryPeriodKernel
from _lcm.solution.backward_induction import (
    _assert_lowered_output_roles,
    _lowering_key,
    _publish_kernel_value,
)


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
    assert tuple(leaf.label for leaf in resolved.expected_leaves) == ("value",)
    assert resolved.expected_leaves[0].shape == template.shape


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
    assert resolved.expected_leaves[1].shape == template.shape[:-1]
    assert resolved.expected_leaves[1].dtype == jnp.bool_


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
    with pytest.raises(ValueError, match="exactly one VALUE"):
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
        template=template,
        compiled_cores={"main": graph["main"].function},
    )

    assert published.sharding == template.sharding


def _state_axes_roles():
    return (
        VALUE,
        StateAxesLeading(state_names=("kind",)),
        StateAxesLeading(state_names=("wealth", "kind"), n_free_leading_axes=1),
        StateAxesLeading(state_names=()),
    )


def test_state_axes_leading_role_places_named_state_prefix_and_replicates_the_rest():
    template = _template()

    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=_state_axes_roles(),
    )

    mesh = template.sharding.mesh
    assert resolved.out_shardings == (
        template.sharding,
        jax.NamedSharding(mesh=mesh, spec=jax.P("kind")),
        jax.NamedSharding(mesh=mesh, spec=jax.P(None, None, "kind")),
        jax.NamedSharding(mesh=mesh, spec=jax.P()),
    )
    assert tuple(leaf.shape for leaf in resolved.expected_leaves) == (
        template.shape,
        None,
        None,
        None,
    )


def _carry_roles():
    """Role tree shaped like a published carry; `None` marks rows it does not carry."""
    _, carry = _carry_core(value=_template())
    roles = [
        StateAxesLeading(state_names=("kind",)),
        StateAxesLeading(state_names=("kind",)),
        StateAxesLeading(state_names=("kind",)),
        StateAxesLeading(state_names=(), shape=()),
    ]
    return (VALUE, jax.tree.unflatten(jax.tree.structure(carry), roles))


def _carry_core(*, value):
    rows = jnp.zeros((value.shape[0], 4), dtype=value.dtype)
    return (
        value,
        EGMCarry(
            endog_grid=rows,
            value=rows,
            marginal_utility=rows,
            taste_shock_scale=jnp.zeros((), dtype=value.dtype),
            breakpoints=None,
            policy=None,
        ),
    )


def test_a_registered_pytree_of_roles_with_none_leaves_resolves_lowers_and_runs():
    """A carry publishes as its own pytree; `None` marks the rows it does not carry."""
    template = _template()
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=_carry_roles(),
    )

    _, carry_shardings = cast("tuple[object, EGMCarry]", resolved.out_shardings)
    assert isinstance(carry_shardings, EGMCarry)
    assert carry_shardings.breakpoints is None
    assert carry_shardings.policy is None
    assert carry_shardings.taste_shock_scale == jax.NamedSharding(
        mesh=template.sharding.mesh, spec=jax.P()
    )
    assert tuple(leaf.label for leaf in resolved.expected_leaves) == (
        "value",
        "leaf [1].endog_grid",
        "leaf [1].value",
        "leaf [1].marginal_utility",
        "leaf [1].taste_shock_scale",
    )

    jitted = jax.jit(_carry_core, out_shardings=resolved.out_shardings)
    lowered = jitted.lower(value=template)
    _assert_lowered_output_roles(
        lowered=lowered,
        output_roles=_carry_roles(),
        layout=resolved,
        label="carry core",
    )
    output = jitted(value=template)
    assert_output_layout(output=output, layout=resolved)


def test_lowered_leaf_metadata_mismatch_names_the_offending_leaf():
    template = _template()
    roles = (VALUE, StateAxesLeading(state_names=("kind",), dtype=jnp.int32))
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=roles,
    )

    lowered = jax.jit(
        lambda value: (value, value), out_shardings=resolved.out_shardings
    ).lower(template)

    with pytest.raises(TypeError, match=r"leaf \[1\] output dtype mismatch"):
        _assert_lowered_output_roles(
            lowered=lowered,
            output_roles=roles,
            layout=resolved,
            label="carry core",
        )


@pytest.mark.parametrize(
    "roles",
    [
        (StateAxesLeading(state_names=("kind",)), VALUE),
        (VALUE, VALUE),
        StateAxesLeading(state_names=("kind",)),
    ],
)
def test_resolver_requires_exactly_one_value_leaf_in_the_first_position(roles):
    with pytest.raises(ValueError, match="exactly one VALUE"):
        resolve_output_layout(
            core_key="main",
            value_template=_template(),
            state_order=("kind", "wealth"),
            output_roles=roles,
        )


def test_resolver_rejects_a_state_axes_leading_role_naming_an_unknown_state():
    with pytest.raises(ValueError, match="outside"):
        resolve_output_layout(
            core_key="main",
            value_template=_template(),
            state_order=("kind", "wealth"),
            output_roles=(VALUE, StateAxesLeading(state_names=("health",))),
        )


def test_compilation_key_tracks_declared_leaf_metadata():
    def resolve(role):
        return resolve_output_layout(
            core_key="main",
            value_template=_template(),
            state_order=("kind", "wealth"),
            output_roles=(VALUE, role),
        )

    untyped = resolve(StateAxesLeading(state_names=("kind",)))
    typed = resolve(StateAxesLeading(state_names=("kind",), dtype=jnp.int32))
    shaped = resolve(StateAxesLeading(state_names=("kind",), shape=(2, 3)))

    assert untyped.compilation_key != typed.compilation_key
    assert untyped.compilation_key != shaped.compilation_key
    assert (
        resolve(StateAxesLeading(state_names=("kind",))).compilation_key
        == untyped.compilation_key
    )


def test_assert_value_leaf_layout_checks_only_the_value_leaf():
    template = _template()
    resolved = resolve_output_layout(
        core_key="main",
        value_template=template,
        state_order=("kind", "wealth"),
        output_roles=(VALUE, StateAxesLeading(state_names=("kind",))),
    )

    assert_value_leaf_layout(value=template, layout=resolved)
    replicated = jax.device_put(
        jnp.zeros(template.shape),
        jax.NamedSharding(mesh=_mesh(), spec=jax.P()),
    )
    with pytest.raises(AssertionError, match="output sharding"):
        assert_value_leaf_layout(value=replicated, layout=resolved)
