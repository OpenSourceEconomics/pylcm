"""Bind named current innovations separately from legacy process state coordinates.

Registry records are declarations, not executable shock kernels. Legacy adapters retain
phase and regime ownership and their original physical-grid rule; they never replace
that rule by a superficially similar base-node rule.
"""

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import cast

from _lcm.grids import DiscreteGrid
from _lcm.processes.iid import _IIDProcess
from _lcm.processes.joint_normal import JointNormalIIDProcess
from _lcm.regime_law import RegimeLaw, RegimeLaws
from _lcm.utils.error_messages import path_segment_name_errors
from lcm.exceptions import ShockDeclarationError, UnsupportedOperationError
from lcm.phased import Phased
from lcm.regime import Regime, StateEntry
from lcm.shocks import (
    Discrete,
    LawCallable,
    ShockGroup,
    ShockGroupName,
    ShockName,
    Uniform,
    _arguments,
    _fingerprint,
)
from lcm.typing import (
    FunctionName,
    ParameterName,
    Phase,
    ReferenceName,
    RegimeName,
    StateName,
    UserFunction,
)


@dataclass(frozen=True, kw_only=True)
class BaseInnovation:
    """Identify a base coordinate without a physical value or local node index."""

    group: ShockGroupName
    """Model group name."""
    shock: ShockName
    """Named innovation within the group."""
    measure: str
    """Standard normal or unit uniform base measure."""

    @property
    def identity(self) -> tuple[ShockName, int]:
        """Return the scalar address name and component, independent of grouping."""
        return self.shock, 0


@dataclass(frozen=True, kw_only=True)
class ShockBinding:
    """Retain one consumer's law dependencies before ordinary variable pruning."""

    group: ShockGroup
    """Law with categorical conditioning bound to this phase's domain."""
    state_inputs: frozenset[StateName]
    """Transitive current-state inputs of the law."""
    state_domains: Mapping[StateName, StateEntry]
    """Unpruned state domains, including carried-state phase declarations."""
    functions: Mapping[FunctionName, LawCallable]
    """Current-state DAG available to the law in this phase."""
    params_template: Mapping[ParameterName, str]
    """Group-local runtime slots; these are not ordinary regime parameters."""


@dataclass(frozen=True, kw_only=True)
class LegacyIIDAdapter:
    """Describe a legacy process without changing its type, timing or rule."""

    declaration: _IIDProcess | JointNormalIIDProcess
    """Original physical-grid declaration, including fold and draw mode."""
    physical_names: tuple[ShockName, ...]
    """Physical outputs; a joint declaration stays joint."""
    phase: Phase
    """Phase whose process this declaration describes."""
    regime: RegimeName
    """Owner; legacy laws may differ across regimes and phases."""
    name: StateName
    """Original state or joint-process declaration key."""

    @property
    def fingerprint(self) -> str:
        """Identify the full compatibility declaration independently of its nodes."""
        return _fingerprint((self.regime, self.phase, self.name, self.declaration))


@dataclass(frozen=True, kw_only=True)
class ShockRegistry:
    """Snapshot explicit groups and regime/phase-specific compatibility records."""

    groups: Mapping[ShockGroupName, ShockGroup]
    """Model declarations; shared references resolve to these exact objects."""
    innovations: Mapping[ShockName, BaseInnovation]
    """Named base identities; category codes and quadrature positions are absent."""
    bindings: Mapping[tuple[RegimeName, Phase, ShockGroupName], ShockBinding]
    """Law input and parameter namespaces for every explicit consumer."""
    legacy: Mapping[tuple[RegimeName, Phase, StateName], LegacyIIDAdapter]
    """Unmodified legacy declarations, with their original ownership."""
    signed_origins: frozenset[ShockGroupName]
    """Declared signed ancestry, retained independently of evaluated weight signs."""

    def fail_if_execution_is_unsupported(self) -> None:
        """Refuse execution until current-IID kernels consume these bindings."""
        if self.bindings:
            raise UnsupportedOperationError(
                "Regime.shocks requires integrated current-IID kernels; "
                "registry declarations cannot execute as ordinary parameters."
            )


def create_shock_registry(
    *,
    shocks: Mapping[ShockGroupName, ShockGroup],
    regimes: Mapping[RegimeName, Regime],
    legacy_regimes: Mapping[RegimeName, Regime],
    model_states: Mapping[StateName, StateEntry],
    laws: RegimeLaws = MappingProxyType({}),
) -> ShockRegistry:
    """Validate explicit identity and bind merged, unpruned consumer dependencies."""
    groups = MappingProxyType(dict(shocks))
    names = (
        frozenset(
            name
            for regime in regimes.values()
            for name in (
                *regime.states,
                *regime.actions,
                *regime.functions,
                *regime.constraints,
                *regime.decomposed_functions,
                *regime.decomposed_constraints,
            )
        )
        | frozenset(regimes)
        | {"edges", "age", "period", "CE", "V", "D", "Q", "policy"}
    )
    for name in groups:
        if (
            not name.isidentifier()
            or path_segment_name_errors(kind="Shock group", names=(name,))
            or name in {*regimes, "edges"}
        ):
            raise ShockDeclarationError(
                f"Shock group name {name!r} conflicts with model vocabulary."
            )
    innovations: dict[ShockName, BaseInnovation] = {}
    for group_name, group in groups.items():
        for name, law in group.shocks.items():
            if (
                name in innovations
                or name in names
                or name.endswith("_code")
                or path_segment_name_errors(kind="Shock", names=(name,))
            ):
                raise ShockDeclarationError(
                    f"Shock name {name!r} has conflicting declarations."
                )
            innovations[name] = BaseInnovation(
                group=group_name,
                shock=name,
                measure="uniform" if isinstance(law, Discrete | Uniform) else "normal",
            )
    legacy = _create_legacy_adapters(regimes=legacy_regimes, model_states=model_states)
    legacy_names = frozenset(
        name for record in legacy.values() for name in record.physical_names
    )
    all_actions = frozenset(
        name for regime in regimes.values() for name in regime.actions
    )
    bindings: dict[tuple[RegimeName, Phase, ShockGroupName], ShockBinding] = {}
    for regime_name, regime in regimes.items():
        unknown = set(regime.shocks) - groups.keys()
        if unknown:
            raise ShockDeclarationError(
                f"Regime {regime_name!r} references unknown shock groups "
                f"{sorted(unknown)!r}."
            )
        for phase in ("solve", "simulate"):
            functions = _phase_functions(regime=regime, phase=phase)
            for group_name in regime.shocks:
                group = groups[group_name]
                state_inputs = group.validate_dependencies(
                    states=frozenset(regime.states),
                    actions=all_actions,
                    functions=functions,
                    shock_names=frozenset(innovations) | legacy_names,
                    downstream_nodes=frozenset(
                        name for item in regimes.values() for name in item.constraints
                    ),
                )
                state_grids = {
                    name: value
                    for name, raw in regime.states.items()
                    if isinstance(
                        value := _phase_value(value=raw, phase=phase), DiscreteGrid
                    )
                }
                bindings[regime_name, phase, group_name] = ShockBinding(
                    group=group.bind_conditioning(state_grids=state_grids),
                    state_inputs=state_inputs,
                    state_domains=MappingProxyType(
                        {name: regime.states[name] for name in state_inputs}
                    ),
                    functions=functions,
                    params_template=_create_group_template(
                        group=group,
                        states=frozenset(regime.states),
                        functions=functions,
                    ),
                )
            _fail_if_shock_reads_are_unlowered(
                declaration=(regime, laws.get(regime_name)),
                shock_names=frozenset(innovations),
            )
    return ShockRegistry(
        groups=groups,
        innovations=MappingProxyType(innovations),
        bindings=MappingProxyType(bindings),
        legacy=legacy,
        signed_origins=frozenset(
            name for name, group in groups.items() if group.signed
        ),
    )


def _create_group_template(  # noqa: C901 - enumerate law slots and callable leaves
    *,
    group: ShockGroup,
    states: frozenset[StateName],
    functions: Mapping[FunctionName, LawCallable],
) -> Mapping[ParameterName, str]:
    template: dict[ParameterName, str] = {}

    def visit(name: ReferenceName) -> None:
        if name in functions:
            for argument in _arguments(functions[name]):
                visit(argument)
        elif (
            name not in states
            and name not in {"period", "age"}
            and not (name.endswith("_code") and name[:-5] in states)
        ):
            template[name] = "float"

    for shock_name, law in group.shocks.items():
        for item in fields(law):
            if item.name == "categories":
                continue
            value = getattr(law, item.name)
            if value is None:
                template[f"{shock_name}__{item.name}"] = "float"
            elif callable(value):
                for argument in _arguments(value):
                    visit(argument)
    if group.correlation is None:
        template["corr"] = "array"
    return MappingProxyType(template)


def _phase_functions(
    *, regime: Regime, phase: Phase
) -> Mapping[FunctionName, LawCallable]:
    return MappingProxyType(
        {
            name: cast("LawCallable", selected)
            for name, value in regime.decomposed_functions.items()
            if callable(selected := _phase_value(value=value, phase=phase))
        }
    )


def _phase_value[T](*, value: T | Phased, phase: Phase) -> T:
    return (
        cast("T", value.solve if phase == "solve" else value.simulate)
        if isinstance(value, Phased)
        else value
    )


def _create_legacy_adapters(
    *,
    regimes: Mapping[RegimeName, Regime],
    model_states: Mapping[StateName, StateEntry],
) -> Mapping[tuple[RegimeName, Phase, StateName], LegacyIIDAdapter]:
    records: dict[tuple[RegimeName, Phase, StateName], LegacyIIDAdapter] = {}
    for regime_name, regime in regimes.items():
        for name, raw in {**model_states, **regime.states}.items():
            for phase in ("solve", "simulate"):
                value = _phase_value(value=raw, phase=phase)
                if isinstance(value, _IIDProcess | JointNormalIIDProcess):
                    records[regime_name, phase, name] = LegacyIIDAdapter(
                        declaration=value,
                        physical_names=value.states
                        if isinstance(value, JointNormalIIDProcess)
                        else (name,),
                        phase=phase,
                        regime=regime_name,
                        name=name,
                    )
    return MappingProxyType(records)


def _fail_if_shock_reads_are_unlowered(
    *,
    declaration: tuple[Regime, RegimeLaw | None],
    shock_names: frozenset[ShockName],
) -> None:
    """Prevent an undeclared shock argument becoming a free ordinary parameter."""
    if not shock_names:
        return
    for value in declaration:
        for func in _iter_declared_callables(value):
            reads = set(inspect.signature(func).parameters) & (
                shock_names | {f"{name}_code" for name in shock_names}
            )
            if reads:
                raise UnsupportedOperationError(
                    f"Current-IID inputs {sorted(reads)!r} require "
                    "integrated shock lowering."
                )


def _iter_declared_callables(
    value: object,  # noqa: PAN001 - walk heterogeneous user declaration fields without evaluating them
) -> Iterator[UserFunction]:
    if callable(value) and not isinstance(value, type):
        yield cast("UserFunction", value)
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_declared_callables(item)
    elif isinstance(value, tuple | list):
        for item in value:
            yield from _iter_declared_callables(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            if item.init:
                yield from _iter_declared_callables(getattr(value, item.name))
