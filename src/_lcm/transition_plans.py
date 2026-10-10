"""Canonical target-edge transition plans.

A target transition is represented once as a composition of finite lotteries and
genuine target-state outputs. Ordinary Markov laws lower to one lottery and one
output; a public JointTransition lowers to one transition-local lottery and every
output that shares its realization. Interpolation bases remain deterministic
coordinates and never enter the lottery mapping.

TargetTransitionPlan is the sole representation consumed by solve, simulation,
validation, diagnostics, and solver-specific continuation machinery.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import no_type_check

import numpy as np
from jax.tree_util import PyTreeDef

from _lcm.typing import (
    EconFunctionArg,
    QualifiedName,
    RegimeName,
    TransitionFunction,
    TransitionFunctionName,
)
from lcm.typing import (
    DiscreteState,
    FloatND,
    ReferenceName,
    StateName,
    UserFunction,
)


class SupportOrigin(Enum):
    """Where a lottery obtains its finite support."""

    TARGET_GRID = auto()
    DECLARED = auto()
    SOURCE_PROCESS = auto()


class LotteryLifetime(Enum):
    """Whether a lottery realization persists as a state."""

    PERSISTED_STATE = auto()
    TRANSITION_LOCAL = auto()


@dataclass(frozen=True)
class SupportSignature:
    """Static structure of one finite support."""

    size: int
    treedef: PyTreeDef | None = None
    leaves: tuple[tuple[tuple[int, ...], np.dtype], ...] = ()


@dataclass(frozen=True)
class ParameterBinding:
    """Public parameter provenance and compiled engine arguments."""

    public_path: tuple[str, ...] = ()
    engine_args: frozenset[ReferenceName] = frozenset()
    user_params: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PhysicalCoordinate:
    """Locate target V through the physical output and target grid logic."""


@dataclass(frozen=True)
class LotteryIndexCoordinate:
    """Use one lottery index directly as the target-V coordinate."""

    lottery_name: str


@dataclass(frozen=True)
class InterpolationBasisInfo:
    """A deterministic support basis whose coefficients are not probabilities."""

    axis_name: str
    support_provider: TransitionFunction | None
    support_signature: SupportSignature
    weight_function: UserFunction
    params: ParameterBinding
    weight_name: str


@dataclass(frozen=True)
class OutputProducerRef:
    """Public declaration that owns one target-state cell."""

    kind: str
    public_name: str


@dataclass(frozen=True)
class LotteryValue:
    """Physical value taken from one realized lottery node."""

    lottery_name: str
    tree_path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class OriginalLotteryLayout:
    """Original code slots of a factored lottery, before any probability product.

    The probability callable belongs to this exact target and phase. At lowering
    it is the declared law; in a transition plan its parameter names are bound
    in the same namespace as the restricted law. No closure is inspected to
    recover either the law or the verified code bijection.
    """

    state_name: StateName
    rest_of_code: tuple[int, ...]
    fixed_of_code: tuple[int, ...]
    probabilities: Callable[..., FloatND]


def declared_law_over_codes(
    func: Callable[..., FloatND],
) -> tuple[OriginalLotteryLayout, Callable[..., FloatND]] | None:
    """Return a restricted law's original layout and its declared law over codes.

    A restricted fixed-component law forwards to a bound method whose receiver
    carries the original layout. The declared law takes the original state code
    as an argument even when it does not read it, so every original source code
    can be evaluated. Returns `None` for any other law.
    """
    wrapped = getattr(func, "__wrapped__", None)
    layout = getattr(getattr(wrapped, "__self__", None), "original_layout", None)
    if not (inspect.ismethod(wrapped) and isinstance(layout, OriginalLotteryLayout)):
        return None
    declared = layout.probabilities
    state_name = layout.state_name
    signature, reads_state_directly = signature_with_state(
        func=declared, state_name=state_name
    )
    if reads_state_directly:
        return layout, declared

    @no_type_check
    def over_codes(**kwargs: EconFunctionArg) -> FloatND:
        return declared(**{k: v for k, v in kwargs.items() if k != state_name})

    over_codes.__signature__ = signature  # ty: ignore[unresolved-attribute]
    over_codes.__name__ = f"next_{state_name}"
    return layout, over_codes


def signature_with_state(
    *, func: Callable[..., FloatND], state_name: StateName
) -> tuple[inspect.Signature, bool]:
    """Return `func`'s signature declaring `state_name`, and whether `func` reads it."""
    signature = inspect.signature(func)
    if state_name in signature.parameters:
        return signature, True
    parameters = list(signature.parameters.values())
    position = next(
        (
            i
            for i, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters.insert(
        position,
        inspect.Parameter(
            state_name, inspect.Parameter.KEYWORD_ONLY, annotation=DiscreteState
        ),
    )
    return signature.replace(parameters=parameters), False


@dataclass(frozen=True)
class TransitionLotteryInfo:
    """One finite stochastic realization mechanism on a target edge."""

    name: str
    qualified_name: QualifiedName
    support_provider: TransitionFunction | None
    support_signature: SupportSignature
    probabilities: UserFunction
    support_origin: SupportOrigin
    lifetime: LotteryLifetime
    persisted_state: StateName | None
    support_params: ParameterBinding
    probability_params: ParameterBinding
    weight_name: str
    support_provider_name: str | None = None
    node_annotation: str | None = None
    original_layout: OriginalLotteryLayout | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Original slots for linear expectations and sampling; absent on ordinary laws."""


@dataclass(frozen=True)
class TransitionOutputInfo:
    """How one genuine target state obtains its next-period value."""

    state: StateName
    next_state_name: TransitionFunctionName
    qualified_name: QualifiedName
    producer: OutputProducerRef
    physical_resolver: TransitionFunction | LotteryValue
    continuation_coordinate: (
        PhysicalCoordinate | LotteryIndexCoordinate | InterpolationBasisInfo
    )
    lottery_dependencies: frozenset[str]
    output_dependencies: frozenset[str]
    params: ParameterBinding
    continuous_process: bool
    intrinsic_entry: bool
    emits_support_index: bool


@dataclass(frozen=True)
class TargetTransitionPlan:
    """Sole canonical transition representation for one target edge and phase."""

    source: RegimeName
    target: RegimeName
    phase: str
    lotteries: MappingProxyType[str, TransitionLotteryInfo]
    outputs: MappingProxyType[str, TransitionOutputInfo]
    output_order: tuple[str, ...]

    def is_lottery(self, transition_name: TransitionFunctionName) -> bool:
        """Return whether a transition name is a finite lottery axis."""
        return transition_name in self.lotteries

    def has_interpolation_basis(self, next_state_name: TransitionFunctionName) -> bool:
        """Return whether an output uses deterministic basis weights."""
        output = self.outputs.get(next_state_name.removeprefix("next_"))
        return output is not None and isinstance(
            output.continuation_coordinate, InterpolationBasisInfo
        )


# Immutable mapping of target regime names to complete target-edge plans.
type TargetTransitionPlans = MappingProxyType[RegimeName, TargetTransitionPlan]
