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
from typing import Literal, Self, cast

import jax
import jax.numpy as jnp
from dags import with_signature

from _lcm.dtypes import canonical_float_dtype
from _lcm.probability import is_represented_zero
from _lcm.processes.base import standard_normal
from _lcm.shock_registry import ShockBinding, ShockRegistry, law_input_validity
from _lcm.typing import ArgmaxQOverAFunction, PRNGKeyND, PytreeValue, QAndFArg
from _lcm.zero_safe import factored_weighted_term
from lcm.exceptions import InvalidParamsError
from lcm.shocks import (
    Discrete,
    LawValue,
    NodeSet,
    ShockGroupName,
    ShockName,
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
CORRELATION_SUPPORT_ARGUMENT = "_lcm_corr_zero_pivots"

type CurrentIIDMaxOutput = tuple[FloatND, BoolND, ScalarBool]
type CurrentIIDDrawOutput = tuple[
    IntND, FloatND, Mapping[ShockName, FloatND | IntND], ScalarBool
]
type NodeOperands = Mapping[
    ShockGroupName, Mapping[str, FloatND | Mapping[ShockName, IntND]]
]


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
        arguments = {
            name: value
            for name, value in kwargs.items()
            if name not in {CURRENT_IID_ARGUMENT, CURRENT_IID_KEYS_ARGUMENT}
        }
        physical: dict[ShockName, FloatND | IntND] = {}
        valid = jnp.asarray(a=True, dtype=jnp.bool_)
        for name, binding in self.plan.bindings.items():
            supplied = group_params.get(name, {})
            params = cast(
                "Mapping[ReferenceName, LawValue]",
                {
                    key: value
                    for key, value in supplied.items()
                    if key != CORRELATION_SUPPORT_ARGUMENT
                },
            )
            base = jnp.stack(
                tuple(
                    _draw_base_value(
                        key=keys[shock], uniform=isinstance(law, Uniform | Discrete)
                    )
                    for shock, law in binding.group.shocks.items()
                )
            )[None]
            inputs = _law_inputs(
                binding=binding,
                arguments={
                    name: cast("QAndFArg", arguments[name])
                    for name in (*binding.state_inputs, "period", "age")
                },
                params=params,
            )
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
            valid &= realised.valid & law_input_validity(binding=binding, inputs=inputs)
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
        return index, value, physical, ~valid


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
        physical: dict[ShockName, FloatND | IntND] = {}
        factors = []
        valid = jnp.asarray(a=True, dtype=jnp.bool_)
        for column, (name, binding, fixed_nodes) in enumerate(self.plan.bindings):
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
        values, empty = jax.vmap(
            partial(_evaluate_node, function=self.function, arguments=arguments)
        )(physical)
        factor_array = jnp.stack(factors)
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
