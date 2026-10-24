"""Integrate current joint innovations around the canonical scalar decision.

Node rows are temporary numerical support. Only the decision's real-state value
and mandatory integrity summaries cross the native program boundary.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial
from itertools import product
from types import MappingProxyType
from typing import Literal, NamedTuple, Self, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.dtypes import canonical_float_dtype
from _lcm.probability import is_represented_zero
from _lcm.processes.base import standard_normal
from _lcm.shock_overrides import CurrentIIDOverride, apply_overrides
from _lcm.shock_registry import ShockBinding, ShockRegistry, law_input_validity
from _lcm.typing import ArgmaxQOverAFunction, PRNGKeyND, PytreeValue, QAndFArg
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import InvalidParamsError, ShockDeclarationError
from lcm.shocks import (
    BaseInnovations,
    Discrete,
    LawValue,
    NodeSet,
    PhysicalValues,
    ShockGroupName,
    ShockName,
    ShockOverride,
    ShockReader,
    Uniform,
)
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    ParameterName,
    ReferenceName,
    ScalarBool,
    UserParams,
    UserParamsLeaf,
)

CURRENT_IID_ARGUMENT = "_lcm_current_iid"
CURRENT_IID_NODES_ARGUMENT = "_lcm_current_iid_nodes"
CURRENT_IID_ROWS_ARGUMENT = "_lcm_current_iid_rows"
CURRENT_IID_KEYS_ARGUMENT = "_lcm_current_iid_keys"
CURRENT_IID_OVERRIDES_ARGUMENT = "_lcm_current_iid_overrides"
CORRELATION_SUPPORT_ARGUMENT = "_lcm_corr_zero_pivots"

type CurrentIIDMaxOutput = tuple[FloatND, BoolND, ScalarBool]
type CurrentIIDDrawOutput = tuple[
    IntND, FloatND, Mapping[ShockName, FloatND | IntND], ScalarBool
]
type NodeOperands = Mapping[
    ShockGroupName, Mapping[str, FloatND | Mapping[ShockName, IntND]]
]


class CurrentIIDDrawFrame(NamedTuple):
    """One produced realization and its explicit independent-base custody.

    The plan owns the producing bindings. Source parameters and resolved source
    inputs remain dynamic fields beside the physical values and both base maps.
    Supplied values carry no policy-observedness claim.
    """

    physical: Mapping[ShockName, FloatND | IntND]
    addressed_bases: Mapping[ShockName, FloatND]
    held_bases: Mapping[ShockName, FloatND]
    source_params: Mapping[ShockGroupName, Mapping[ReferenceName, LawValue | BoolND]]
    source_inputs: Mapping[ShockGroupName, Mapping[ReferenceName, LawValue]]
    recovery_error: Mapping[ShockName, FloatND]
    valid: BoolND


@dataclass(frozen=True, kw_only=True)
class CurrentIIDDrawPlan:
    """Bind one fresh physical realization to one native subject decision."""

    bindings: Mapping[ShockGroupName, ShockBinding]
    """Phase-specific law/helper bindings, preserving complete joint groups."""

    def __post_init__(self) -> None:
        """Refuse signed ancestry before it reaches a native decision or draw."""
        from lcm.exceptions import ShockDeclarationError  # noqa: PLC0415

        for name, binding in self.bindings.items():
            if binding.group.signed:
                raise ShockDeclarationError(
                    f"Signed shock group {name!r} cannot feed a simulation decision."
                )

    @property
    def physical_names(self) -> tuple[ShockName, ...]:
        """Return named physical outputs in the declared joint column order."""
        return tuple(
            name for binding in self.bindings.values() for name in binding.group.shocks
        )

    @property
    def state_inputs(self) -> frozenset[ReferenceName]:
        """Return actual current-state operands, including carried-only states."""
        return frozenset(
            name for binding in self.bindings.values() for name in binding.state_inputs
        )

    def realise(
        self,
        *,
        keys: Mapping[ShockName, PRNGKeyND],
        group_params: Mapping[
            ShockGroupName, Mapping[ReferenceName, LawValue | BoolND]
        ],
        arguments: Mapping[ReferenceName, PytreeValue],
        overrides: Mapping[ShockName, ShockOverride] = MappingProxyType({}),
        override_active: Mapping[ShockName, BoolND] = MappingProxyType({}),
    ) -> CurrentIIDDrawFrame:
        """Produce one scalar cell, retaining its producing context and named bases.

        Caller routing selects original-row keys and override cells before this
        producer. Invalid law/support status must be checked before any decision.
        """
        unknown = overrides.keys() - set(self.physical_names)
        if unknown:
            raise ShockDeclarationError(f"Unknown shock overrides {sorted(unknown)}.")
        physical: dict[ShockName, FloatND | IntND] = {}
        addressed_bases: dict[ShockName, FloatND] = {}
        held_bases: dict[ShockName, FloatND] = {}
        source_params = {}
        source_inputs = {}
        recovery_error = {}
        valid = jnp.asarray(a=True, dtype=jnp.bool_)
        for name, binding in self.bindings.items():
            supplied = group_params.get(name, {})
            params = cast(
                "Mapping[ReferenceName, LawValue]",
                {
                    key: value
                    for key, value in supplied.items()
                    if key != CORRELATION_SUPPORT_ARGUMENT
                },
            )
            addressed = {
                shock: _draw_base_value(
                    key=keys[shock], uniform=isinstance(law, Uniform | Discrete)
                )
                for shock, law in binding.group.shocks.items()
            }
            inputs = _law_inputs(
                binding=binding,
                arguments={
                    name: cast("QAndFArg", arguments[name])
                    for name in (*binding.state_inputs, "period", "age")
                },
                params=params,
            )
            support = cast("BoolND | None", supplied.get(CORRELATION_SUPPORT_ARGUMENT))
            active = {
                shock: enabled
                for shock, enabled in override_active.items()
                if shock in binding.group.shocks
            }
            fallback_physical = {}
            if active:
                ordinary, _ = binding.group._realise(  # noqa: SLF001
                    base_nodes=jnp.stack(tuple(addressed.values()))[None],
                    nodes=None,
                    params=params,
                    inputs=inputs,
                    scalar_cell_group=name,
                    admitted_correlation_support=support,
                )
                fallback_physical = {
                    shock: value[0] for shock, value in ordinary.values.items()
                }
            override = apply_overrides(
                group=binding.group,
                addressed=addressed,
                overrides={
                    key: value
                    for key, value in overrides.items()
                    if key in binding.group.shocks
                },
                params=params,
                inputs=inputs,
                admitted_correlation_support=support,
                active=active,
                fallback_physical=fallback_physical,
            )
            addressed_bases.update(addressed)
            held_bases.update(override.bases)
            source_params[name] = MappingProxyType(dict(supplied))
            source_inputs[name] = MappingProxyType(dict(inputs))
            base = jnp.stack(tuple(override.bases.values()))[None]
            realised, _ = binding.group._realise(  # noqa: SLF001 - shared internal law kernel
                base_nodes=base,
                nodes=None,
                params=params,
                scalar_cell_group=name,
                inputs=inputs,
                admitted_correlation_support=cast(
                    "BoolND | None", supplied.get(CORRELATION_SUPPORT_ARGUMENT)
                ),
            )
            physical.update({key: value[0] for key, value in realised.values.items()})
            for shock, supplied_value in override.physical.items():
                produced = physical[shock]
                if produced.dtype != supplied_value.dtype:
                    raise ShockDeclarationError(
                        "Physical values must use their producing law's dtype."
                    )
                enabled = jnp.asarray(active.get(shock, True))
                supplied_physical = jnp.where(enabled, supplied_value, produced)
                if produced.dtype.kind == "f":
                    bits = jnp.uint64 if produced.dtype.itemsize == 8 else jnp.uint32  # noqa: PLR2004 - active IEEE float word width
                    witnessed = jax.lax.bitcast_convert_type(
                        produced, bits
                    ) == jax.lax.bitcast_convert_type(supplied_physical, bits)
                else:
                    witnessed = produced == supplied_physical
                # Singular joint support imposes cross-coordinate restrictions;
                # independent/full-rank support is not an exact-native-image test.
                valid &= ~override.requires_witness | witnessed
                recovery_error[shock] = produced.astype(
                    canonical_float_dtype()
                ) - supplied_physical.astype(canonical_float_dtype())
                valid &= jnp.isfinite(recovery_error[shock])
                physical[shock] = supplied_physical
            valid &= (
                override.valid
                & realised.valid
                & law_input_validity(binding=binding, inputs=inputs)
            )
        return CurrentIIDDrawFrame(
            physical=MappingProxyType(physical),
            addressed_bases=MappingProxyType(addressed_bases),
            held_bases=MappingProxyType(held_bases),
            source_params=MappingProxyType(source_params),
            source_inputs=MappingProxyType(source_inputs),
            recovery_error=MappingProxyType(recovery_error),
            valid=valid,
        )

    def wrap(
        self, function: Callable[..., PytreeValue]
    ) -> ArgmaxQOverAFunction[CurrentIIDDrawOutput]:
        """Realize before the canonical action reducer, retaining replay inputs."""
        names = inspect.signature(function).parameters
        return cast(
            "ArgmaxQOverAFunction[CurrentIIDDrawOutput]",
            with_signature(
                _CurrentIIDDraw(function=function, plan=self),
                args=list(
                    dict.fromkeys(
                        (
                            *(
                                name
                                for name in names
                                if name not in self.physical_names
                            ),
                            *sorted(self.state_inputs),
                            "period",
                            "age",
                            CURRENT_IID_ARGUMENT,
                            CURRENT_IID_KEYS_ARGUMENT,
                            CURRENT_IID_OVERRIDES_ARGUMENT,
                        )
                    )
                ),
                enforce=False,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _CurrentIIDDraw:
    """Apply the physical law once and return it beside the selected decision."""

    function: Callable[..., PytreeValue]
    plan: CurrentIIDDrawPlan

    def __call__(
        self, **kwargs: PytreeValue
    ) -> tuple[IntND, FloatND, Mapping[ShockName, FloatND | IntND], ScalarBool]:
        group_params = cast(
            "Mapping[ShockGroupName, Mapping[ReferenceName, LawValue | BoolND]]",
            kwargs[CURRENT_IID_ARGUMENT],
        )
        keys = cast("Mapping[ShockName, PRNGKeyND]", kwargs[CURRENT_IID_KEYS_ARGUMENT])
        overrides = cast(
            "Mapping[ShockName, CurrentIIDOverride]",
            kwargs.get(CURRENT_IID_OVERRIDES_ARGUMENT, {}),
        )
        arguments = {
            name: value
            for name, value in kwargs.items()
            if name
            not in {
                CURRENT_IID_ARGUMENT,
                CURRENT_IID_KEYS_ARGUMENT,
                CURRENT_IID_OVERRIDES_ARGUMENT,
            }
        }
        frame = self.plan.realise(
            keys=keys,
            group_params=group_params,
            arguments=arguments,
            overrides={
                name: BaseInnovations(values=cast("FloatND", item.values))
                if item.kind == "base"
                else PhysicalValues(values=cast("FloatND | IntND", item.values))
                for name, item in overrides.items()
            },
            override_active={
                name: cast("BoolND", item.active) for name, item in overrides.items()
            },
        )
        physical, valid = frame.physical, frame.valid
        available = {**arguments, **physical}
        index, value = cast(
            "tuple[IntND, FloatND]",
            self.function(
                **{
                    name: available[name]
                    for name in inspect.signature(self.function).parameters
                }
            ),
        )
        return index, value, dict(physical), ~valid


def _draw_base_value(*, key: PRNGKeyND, uniform: bool) -> FloatND:
    """Preserve S1's scalar-dispatch marker under the outer subject vmap."""
    draw = (
        partial(jax.random.uniform, dtype=canonical_float_dtype())
        if uniform
        else standard_normal
    )
    if key.shape == (1,):
        return jax.custom_batching.sequential_vmap(draw)(key[0])
    return draw(key)


def resolve_group_params(
    *,
    registry: ShockRegistry,
    params: UserParams,
    fixed_params: UserParams = MappingProxyType({}),
) -> Mapping[ShockGroupName, Mapping[ParameterName, UserParamsLeaf]]:
    """Resolve only explicitly named group slots, without regime broadcasting."""
    templates = registry.params_template
    unknown = (set(params) | set(fixed_params)) - templates.keys()
    if unknown:
        raise InvalidParamsError(
            f"Unknown shock group parameter roots: {sorted(unknown)!r}."
        )
    resolved = {}
    for name, slots in templates.items():
        supplied = params.get(name, {})
        fixed = fixed_params.get(name, {})
        if not isinstance(supplied, Mapping) or not isinstance(fixed, Mapping):
            raise InvalidParamsError(
                f"Shock group {name!r} parameters must be a mapping."
            )
        if supplied.keys() & fixed.keys():
            raise InvalidParamsError(
                f"Shock group {name!r} cannot override fixed law slots "
                f"{sorted(supplied.keys() & fixed.keys())!r}."
            )
        supplied = {**fixed, **supplied}
        missing = slots.keys() - supplied.keys()
        unknown_slots = supplied.keys() - slots.keys()
        if missing or unknown_slots:
            raise InvalidParamsError(
                f"Shock group {name!r}: missing {sorted(missing)!r}; "
                f"unknown {sorted(unknown_slots)!r}."
            )
        if any(isinstance(value, Mapping) for value in supplied.values()):
            raise InvalidParamsError(
                f"Shock group {name!r} expects direct parameter leaves."
            )
        resolved[name] = MappingProxyType(
            cast("Mapping[ParameterName, UserParamsLeaf]", dict(supplied))
        )
    return MappingProxyType(resolved)


@dataclass(frozen=True, kw_only=True)
class CurrentIIDPlan:
    """Own exact-age joint support and current-phase law bindings."""

    bindings: tuple[tuple[ShockGroupName, ShockBinding, NodeSet], ...]
    """Ordered independent groups, each preserving its joint rows."""
    indices: IntND
    """Cartesian row indices across groups, never across group columns."""

    @classmethod
    def create(
        cls,
        *,
        bindings: Mapping[ShockGroupName, ShockBinding],
        age: Fraction | int,
        coordinate_kind: Literal["age", "period"],
    ) -> Self | None:
        """Resolve the declared rules at one exact computational age."""
        if not bindings:
            return None
        resolved = tuple(
            (
                name,
                binding,
                binding.group.resolve_nodes(
                    group_name=name, age=age, coordinate_kind=coordinate_kind
                ),
            )
            for name, binding in sorted(bindings.items())
        )
        for _, _, nodes in resolved:
            nodes.admit(reader=ShockReader.DECISION)
        return cls(
            bindings=resolved,
            indices=jnp.asarray(
                tuple(
                    product(*(range(nodes.weights.size) for _, _, nodes in resolved))
                ),
                dtype=jnp.int32,
            ),
        )

    @property
    def node_operands(
        self,
    ) -> Mapping[ShockGroupName, Mapping[str, FloatND | Mapping[ShockName, IntND]]]:
        """Expose every fixed bank as a native operand for ownership and admission."""
        return MappingProxyType(
            {
                name: MappingProxyType(
                    {
                        "base": nodes.base_nodes,
                        "weights": nodes.weights,
                        "codes": nodes.discrete_codes,
                    }
                )
                for name, _, nodes in self.bindings
            }
        )

    @property
    def physical_names(self) -> frozenset[ShockName]:
        """Return inputs bound by the physical law rather than by parameters."""
        return frozenset(
            name for _, binding, _ in self.bindings for name in binding.group.shocks
        )

    @property
    def state_inputs(self) -> frozenset[ReferenceName]:
        """Return law dependencies retained independently of the objective DAG."""
        return frozenset(
            name for _, binding, _ in self.bindings for name in binding.state_inputs
        )

    def wrap(
        self, function: Callable[..., PytreeValue]
    ) -> Callable[..., tuple[FloatND, BoolND, ScalarBool]]:
        """Bind the current law inside the native scalar-cell decision program."""
        names = tuple(inspect.signature(function).parameters)
        return with_signature(
            _CurrentIIDExpectation(function=function, plan=self),
            args=list(
                dict.fromkeys(
                    (
                        *(name for name in names if name not in self.physical_names),
                        *sorted(self.state_inputs),
                        "period",
                        "age",
                        CURRENT_IID_ARGUMENT,
                        CURRENT_IID_NODES_ARGUMENT,
                        CURRENT_IID_ROWS_ARGUMENT,
                    )
                )
            ),
            enforce=False,
        )

    def realise_nodes(
        self,
        *,
        arguments: Mapping[ReferenceName, QAndFArg],
        group_params: Mapping[ShockGroupName, Mapping[ReferenceName, LawValue]],
        node_operands: NodeOperands,
        indices: IntND,
    ) -> tuple[Mapping[ShockName, FloatND | IntND], FloatND, ScalarBool]:
        """Expose the same physical joint rows, factors and mandatory law status.

        Query consumers retain node-level action identities. Ordinary integration
        consumes this same realization before reducing its selected values.
        """
        physical: dict[ShockName, FloatND | IntND] = {}
        factors = []
        valid = jnp.asarray(a=True, dtype=jnp.bool_)
        for column, (name, binding, fixed_nodes) in enumerate(self.bindings):
            operands = node_operands[name]
            nodes = replace(
                fixed_nodes,
                base_nodes=cast("FloatND", operands["base"]),
                weights=cast("FloatND", operands["weights"]),
                discrete_codes=cast("Mapping[ShockName, IntND]", operands["codes"]),
            )
            supplied = group_params[name]
            params = {
                key: value
                for key, value in supplied.items()
                if key != CORRELATION_SUPPORT_ARGUMENT
            }
            inputs = _law_inputs(binding=binding, arguments=arguments, params=params)
            realised, group_factors = binding.group._realise(  # noqa: SLF001 - shared internal law kernel
                base_nodes=nodes.base_nodes,
                nodes=nodes,
                params=params,
                inputs=inputs,
                scalar_cell_group=name,
                admitted_correlation_support=cast(
                    "BoolND | None", supplied.get(CORRELATION_SUPPORT_ARGUMENT)
                ),
            )
            row = indices[:, column]
            physical.update({key: value[row] for key, value in realised.values.items()})
            factors.extend(factor[row] for factor in group_factors)
            valid &= realised.valid & law_input_validity(binding=binding, inputs=inputs)
        return MappingProxyType(physical), jnp.stack(factors), valid


@dataclass(frozen=True, kw_only=True)
class _CurrentIIDExpectation:
    """Select each node's action before reducing the current innovation."""

    function: Callable[..., PytreeValue]
    """Canonical scalar action reducer."""
    plan: CurrentIIDPlan
    """Immutable law and joint support."""

    def __call__(self, **kwargs: QAndFArg) -> tuple[FloatND, BoolND, ScalarBool]:
        group_params = cast(
            "Mapping[ShockGroupName, Mapping[ReferenceName, LawValue]]",
            kwargs[CURRENT_IID_ARGUMENT],
        )
        node_operands = cast(
            "NodeOperands",
            kwargs[CURRENT_IID_NODES_ARGUMENT],
        )
        arguments = {
            name: value
            for name, value in kwargs.items()
            if name
            not in {
                CURRENT_IID_ARGUMENT,
                CURRENT_IID_NODES_ARGUMENT,
                CURRENT_IID_ROWS_ARGUMENT,
            }
        }
        indices = cast("IntND", kwargs[CURRENT_IID_ROWS_ARGUMENT])
        physical, factor_array, valid = self.plan.realise_nodes(
            arguments=arguments,
            group_params=group_params,
            node_operands=node_operands,
            indices=indices,
        )
        values, empty = jax.vmap(
            partial(_evaluate_node, function=self.function, arguments=arguments)
        )(physical)
        support = jnp.all(~is_represented_zero(factor_array), axis=0)
        terms = factored_weighted_term(factors=factor_array, values=values)
        return jnp.sum(terms), jnp.any(support & empty), ~valid


# keyword-only-exempt: library-callback=jax.vmap
def _evaluate_node(
    physical: Mapping[ShockName, FloatND | IntND],
    *,
    function: Callable[..., PytreeValue],
    arguments: Mapping[ReferenceName, QAndFArg],
) -> tuple[FloatND, BoolND]:
    inputs = {**arguments, **physical}
    return cast(
        "tuple[FloatND, BoolND]",
        function(
            **{name: inputs[name] for name in inspect.signature(function).parameters}
        ),
    )


def _law_inputs(
    *,
    binding: ShockBinding,
    arguments: Mapping[ReferenceName, QAndFArg],
    params: Mapping[ReferenceName, LawValue],
) -> Mapping[ReferenceName, LawValue]:
    """Evaluate the admitted current-state helper DAG in its group namespace."""
    inputs = {name: cast("LawValue", arguments[name]) for name in binding.state_inputs}
    inputs.update(
        {
            name: cast("LawValue", arguments[name])
            for name in ("age", "period")
            if name in arguments
        }
    )
    pending = dict(binding.functions)
    while pending:
        ready = [
            name
            for name, function in pending.items()
            if set(inspect.signature(function).parameters)
            <= (inputs.keys() | params.keys())
        ]
        if not ready:
            break
        for name in ready:
            function = pending.pop(name)
            available = {**params, **inputs}
            inputs[name] = function(
                **{
                    key: available[key]
                    for key in inspect.signature(function).parameters
                }
            )
    return MappingProxyType(inputs)
