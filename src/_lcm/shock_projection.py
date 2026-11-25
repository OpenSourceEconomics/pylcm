"""Prove that an observed shock has one law at the source and at the landing.

This admission consumes actual read closures and canonical projection bindings. It
never infers an information set from a regime's permission to reference a group.
Continuous identity is a direct-return JAXpr proof; discrete identity may also be
proved exhaustively on the complete finite code domain. Failure is conservative.
"""

import inspect
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields
from itertools import product
from types import MappingProxyType

import jax
import numpy as np
from dags import concatenate_functions

from _lcm.grids import Grid
from _lcm.shock_registry import ShockBinding
from _lcm.typing import PytreeValue, QualifiedName
from lcm.exceptions import ShockDeclarationError
from lcm.grids import DiscreteGrid
from lcm.phased import Phased
from lcm.regime import StateEntry
from lcm.shocks import (
    LawCallable,
    ShockGroupName,
    ShockName,
    _ConditionedLookup,
    _fingerprint,
)
from lcm.typing import FunctionName, ReferenceName, RegimeName, StateName, UserFunction

type OperandPath = tuple[str | int, ...]

_MAX_FINITE_PROJECTION_CELLS = 4096


def validate_observed_laws(
    *,
    source: RegimeName,
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
    """Refuse an observed shock whose law can differ at the source and the landing.

    A group that the source draws, sees and the target reads is one draw, so each
    of its shocks needs one law at the source's state and at the landing state: the
    same functions and parameters in both regimes, reading only parameters and
    states that the projection returns unchanged. A group only the target draws has
    its law at the landing state alone and adds no obligation, nor does a
    declaration that the source or target does not read.
    """
    seen = source_physical_inputs & physical_inputs
    for group, binding in target_bindings.items():
        own = source_bindings.get(group)
        if own is None or not seen & binding.group.shocks.keys():
            continue
        for shock in binding.group.shocks:
            record = _law_record(
                shock=shock,
                group=group,
                binding=binding,
                parameter_paths=target_parameter_paths,
            )
            context = (
                f"Regime {source!r} observes shock {shock!r} of group {group!r} at "
                f"its exit to {target!r}"
            )
            if record != _law_record(
                shock=shock,
                group=group,
                binding=own,
                parameter_paths=source_parameter_paths,
            ):
                raise ShockDeclarationError(
                    f"{context}, but its law reads different functions or "
                    f"parameters in {source!r} and in {target!r}."
                )
            for state in sorted(record[1]):
                if state not in projection or not _kept(
                    state=state,
                    func=projection[state],
                    functions=projection_functions,
                    source_domains=source_domains,
                    target_domains=target_domains,
                ):
                    raise ShockDeclarationError(
                        f"{context}, and its law reads state {state!r}, which the "
                        "exit's projection changes or is not proven to keep. The "
                        "law of an observed shock must be the same at the source's "
                        "state and at the landing state, so it may read only "
                        "parameters and states the projection returns unchanged. "
                        f"Draw {shock!r} in {source!r} and carry it to {target!r} "
                        "as a state instead."
                    )


def _kept(
    *,
    state: StateName,
    func: UserFunction,
    functions: Mapping[FunctionName, UserFunction],
    source_domains: Mapping[StateName, StateEntry],
    target_domains: Mapping[StateName, StateEntry],
) -> bool:
    """Whether `func` lands `state` at its source value, on the same grid."""
    source_grid = _grid(source_domains.get(state))
    target_grid = _grid(target_domains.get(state))
    if (
        source_grid is None
        or target_grid is None
        or _fingerprint((source_grid,)) != _fingerprint((target_grid,))
    ):
        return False
    return _preserves_input(
        state=state,
        func=func,
        functions=functions,
        source_domains=source_domains,
        grid=source_grid,
    )


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


def returned_input(
    *, func: Callable[..., PytreeValue], dtypes: Mapping[str, np.dtype]
) -> str | None:
    """Name the scalar input that `func` returns unchanged, proved from its JAXpr.

    The trace over abstract scalars of the given dtypes must have no equations and
    no constants, and its one output must be an input variable of the same abstract
    value; None otherwise.

    Raises:
        ValueError: The trace has effects, so nothing about `func` is proved.

    An error raised while tracing propagates: such a callable proves nothing either.
    """
    names = tuple(dtypes)
    closed = jax.make_jaxpr(
        lambda *values: func(**dict(zip(names, values, strict=True)))
    )(*(jax.ShapeDtypeStruct((), dtype) for dtype in dtypes.values()))
    jaxpr = closed.jaxpr
    if jaxpr.effects:
        raise ValueError("A traced projection has effects.")
    if jaxpr.eqns or jaxpr.constvars or len(jaxpr.outvars) != 1:
        return None
    (output,) = jaxpr.outvars
    return next(
        (
            name
            for name, variable in zip(names, jaxpr.invars, strict=True)
            if output is variable and output.aval == variable.aval
        ),
        None,
    )


def finite_outputs(
    *, func: Callable[..., PytreeValue], domains: Mapping[str, np.ndarray]
) -> Iterator[tuple[Mapping[str, np.ndarray], np.ndarray]] | None:
    """Evaluate `func` eagerly at every combination of its inputs' finite domains.

    Every combination is visited, including ones that the solved population does
    not occupy, so a property that holds at each output holds for the callable.
    Each input is a scalar in its domain's dtype. None when the product of the
    domain sizes exceeds the private enumeration cap. The outputs are produced
    lazily, so a caller may stop at the first that fails its check.
    """
    names = tuple(domains)
    if math.prod(len(domain) for domain in domains.values()) > (
        _MAX_FINITE_PROJECTION_CELLS
    ):
        return None

    def outputs() -> Iterator[tuple[Mapping[str, np.ndarray], np.ndarray]]:
        for values in product(*domains.values()):
            inputs = {
                name: np.asarray(value, dtype=domains[name].dtype)
                for name, value in zip(names, values, strict=True)
            }
            yield inputs, np.asarray(func(**inputs))

    return outputs()


def _preserves_input(
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
        dtypes = {
            name: np.dtype(domain.to_jax().dtype)
            for name, domain in domains.items()
            if domain is not None
        }
        if returned_input(func=composed, dtypes=dtypes) == state:
            return True
        if not isinstance(grid, DiscreteGrid) or any(
            not isinstance(domain, DiscreteGrid) for domain in domains.values()
        ):
            return False
        outputs = finite_outputs(
            func=composed,
            domains={
                name: np.asarray(domain.codes, dtype=dtypes[name])
                for name, domain in domains.items()
                if isinstance(domain, DiscreteGrid)
            },
        )
        # The private cap bounds build-time enumeration; direct-return proofs above
        # do not enumerate or need it.
        return outputs is not None and all(
            actual.shape == ()
            and actual.dtype == inputs[state].dtype
            and actual == inputs[state]
            for inputs, actual in outputs
        )
    except Exception:  # noqa: BLE001 - untraceable user projection is an unproved identity
        return False
