"""Prove when a demanded query can reuse a shared discrete physical category.

This admission consumes actual read closures and canonical projection bindings. It
never infers an information set from a regime's permission to reference a group.
Continuous identity is a direct-return JAXpr proof; discrete identity may also be
proved exhaustively on the complete finite code domain. Failure is conservative.
"""

import inspect
import math
from collections.abc import Mapping
from dataclasses import fields
from itertools import product
from types import MappingProxyType
from typing import Literal

import jax
import numpy as np
from dags import concatenate_functions

from _lcm.grids import DiscreteGrid, Grid
from _lcm.shock_registry import ShockBinding
from _lcm.typing import QualifiedName
from lcm.exceptions import ShockDeclarationError
from lcm.phased import Phased
from lcm.regime import StateEntry
from lcm.shocks import (
    Discrete,
    LawCallable,
    ShockGroupName,
    ShockName,
    _ConditionedLookup,
    _fingerprint,
)
from lcm.typing import FunctionName, ReferenceName, RegimeName, StateName, UserFunction

type QueryKindName = Literal[
    "Raw", "Participating", "Full", "Arrival", "IntegratedFull"
]
type OperandPath = tuple[str | int, ...]

_MAX_FINITE_PROJECTION_CELLS = 4096


def validate_shared_discrete_projection(
    *,
    query_kind: QueryKindName,
    target: RegimeName,
    physical_inputs: frozenset[ShockName],
    source_physical_inputs: frozenset[ShockName],
    projection: Mapping[StateName, UserFunction],
    source_bindings: Mapping[ShockGroupName, ShockBinding],
    target_bindings: Mapping[ShockGroupName, ShockBinding],
    source_domains: Mapping[StateName, StateEntry],
    target_domains: Mapping[StateName, StateEntry],
    projection_functions: Mapping[FunctionName, UserFunction] = MappingProxyType({}),
    source_parameter_paths: Mapping[QualifiedName, OperandPath] = MappingProxyType({}),
    target_parameter_paths: Mapping[QualifiedName, OperandPath] = MappingProxyType({}),
) -> None:
    """Admit identical partitions only for actually demanded shared addresses.

    Bind fixed projection operands into the effective callable before admission;
    runtime operands must retain their canonical binding paths. A declaration that
    the source or query does not read contributes no shared-category obligation.
    Numerical node-rule identity and conditional integration remain separate.
    """
    source = _discrete_bindings(source_bindings)
    targets = _discrete_bindings(target_bindings)
    shared = source_physical_inputs & physical_inputs & source.keys() & targets.keys()
    for shock in sorted(shared):
        source_group, source_binding = source[shock]
        target_group, target_binding = targets[shock]
        source_record, source_states = _law_record(
            shock=shock,
            group=source_group,
            binding=source_binding,
            parameter_paths=source_parameter_paths,
        )
        target_record, target_states = _law_record(
            shock=shock,
            group=target_group,
            binding=target_binding,
            parameter_paths=target_parameter_paths,
        )
        context = f"Shock {shock!r}, {query_kind}[{target}], projection"
        if source_record != target_record or source_states != target_states:
            raise ShockDeclarationError(
                f"{context}: law/helper/parameter bindings differ."
            )
        for state in sorted(target_states):
            source_grid = _grid(source_domains.get(state))
            target_grid = _grid(target_domains.get(state))
            if (
                source_grid is None
                or target_grid is None
                or _fingerprint((source_grid,)) != _fingerprint((target_grid,))
            ):
                raise ShockDeclarationError(
                    f"{context} input {state!r}: state domain/codecs differ."
                )
            if state not in projection or not _preserves_input(
                state=state,
                func=projection[state],
                functions=projection_functions,
                source_domains=source_domains,
                grid=source_grid,
            ):
                raise ShockDeclarationError(
                    f"{context} input {state!r}: effective identity is unproved."
                )


def _discrete_bindings(
    bindings: Mapping[ShockGroupName, ShockBinding],
) -> dict[ShockName, tuple[ShockGroupName, ShockBinding]]:
    return {
        shock: (name, binding)
        for name, binding in bindings.items()
        for shock, law in binding.group.shocks.items()
        if isinstance(law, Discrete)
    }


def _law_record(  # noqa: C901 - walk bound law/helper dependencies and operand paths
    *,
    shock: ShockName,
    group: ShockGroupName,
    binding: ShockBinding,
    parameter_paths: Mapping[QualifiedName, OperandPath],
) -> tuple[str, frozenset[StateName]]:
    law = binding.group.shocks[shock]
    helpers: dict[FunctionName, LawCallable] = {}
    states: set[StateName] = set()
    paths: dict[ReferenceName, OperandPath] = {}

    def visit(name: ReferenceName) -> None:
        if name in binding.functions:
            if name not in helpers:
                helpers[name] = binding.functions[name]
                for argument in inspect.signature(helpers[name]).parameters:
                    visit(argument)
        elif name in binding.state_domains:
            states.add(name)
        elif name not in {"age", "period"}:
            key = f"{group}__{name}"
            paths[name] = parameter_paths.get(key, (group, name))

    for item in fields(law):
        value = getattr(law, item.name)
        if isinstance(value, _ConditionedLookup):
            visit(value.on)
        elif item.name != "categories" and callable(value):
            for argument in inspect.signature(value).parameters:
                visit(argument)
        elif value is None:
            visit(f"{shock}__{item.name}")
    return _fingerprint((law, helpers, paths)), frozenset(states)


def _grid(value: StateEntry | None) -> Grid | None:
    # A carried state's public domain is its simulate grid in both phases; its
    # solve imputation remains part of the transitive law/helper identity.
    if isinstance(value, Phased):
        value = value.simulate
    return value if isinstance(value, Grid) else None


def _preserves_input(  # noqa: C901, PLR0911 - refuse failed proof obligations
    *,
    state: StateName,
    func: UserFunction,
    functions: Mapping[FunctionName, UserFunction],
    source_domains: Mapping[StateName, StateEntry],
    grid: Grid,
) -> bool:
    try:
        composed = concatenate_functions(
            functions={**functions, "_lcm_projection_result": func},
            targets="_lcm_projection_result",
        )
        names = tuple(inspect.signature(composed).parameters)
        if state not in names:
            return False
        domains = {name: _grid(source_domains.get(name)) for name in names}
        if any(domain is None for domain in domains.values()):
            return False
        abstracts = tuple(
            jax.ShapeDtypeStruct((), domain.to_jax().dtype)
            for domain in domains.values()
            if domain is not None
        )
        closed = jax.make_jaxpr(
            lambda *values: composed(**dict(zip(names, values, strict=True)))
        )(*abstracts)
        jaxpr = closed.jaxpr
        if jaxpr.effects:
            return False
        original = jaxpr.invars[names.index(state)]
        if (
            not jaxpr.eqns
            and not jaxpr.constvars
            and len(jaxpr.outvars) == 1
            and jaxpr.outvars[0] is original
            and jaxpr.outvars[0].aval == original.aval
        ):
            return True
        if not isinstance(grid, DiscreteGrid) or any(
            not isinstance(domain, DiscreteGrid) for domain in domains.values()
        ):
            return False
        code_domains = tuple(
            domain.codes
            for domain in domains.values()
            if isinstance(domain, DiscreteGrid)
        )
        if math.prod(map(len, code_domains)) > _MAX_FINITE_PROJECTION_CELLS:
            return False
        # Every possible code tuple is checked, including combinations that the
        # solved population does not occupy. The private cap bounds build-time
        # enumeration; direct-return proofs above do not enumerate or need it.
        for codes in product(*code_domains):
            inputs = {
                name: np.asarray(code, dtype=abstract.dtype)
                for name, code, abstract in zip(names, codes, abstracts, strict=True)
            }
            actual = np.asarray(composed(**inputs))
            expected = inputs[state]
            if (
                actual.shape != ()
                or actual.dtype != expected.dtype
                or actual != expected
            ):
                return False
    except Exception:  # noqa: BLE001 - untraceable user projection is an unproved identity
        return False
    else:
        return True
