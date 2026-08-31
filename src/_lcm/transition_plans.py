"""Canonical target-edge transition plans.

A target transition is represented once as a composition of finite lotteries and
genuine target-state outputs. Ordinary Markov laws lower to one lottery and one
output; a public JointTransition lowers to one transition-local lottery and every
output that shares its realization. Interpolation bases remain deterministic
coordinates and never enter the lottery mapping.

TargetTransitionPlan is the sole representation consumed by solve, simulation,
validation, diagnostics, and solver-specific continuation machinery.
"""

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from _lcm.typing import RegimeName, TransitionFunctionName


class SupportOrigin(Enum):
    """Where a lottery obtains its finite support."""

    TARGET_GRID = auto()
    DECLARED = auto()


class LotteryLifetime(Enum):
    """Whether a lottery realization persists as a state."""

    PERSISTED_STATE = auto()
    TRANSITION_LOCAL = auto()


@dataclass(frozen=True)
class SupportSignature:
    """Static structure of one finite support."""

    size: int
    treedef: object | None = None
    leaves: tuple[object, ...] = ()


@dataclass(frozen=True)
class ParameterBinding:
    """Public parameter provenance and compiled engine arguments."""

    public_path: tuple[str, ...] = ()
    engine_args: frozenset[str] = frozenset()
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
    support_provider: object | None
    support_signature: SupportSignature
    weight_function: object
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
class TransitionLotteryInfo:
    """One finite stochastic realization mechanism on a target edge."""

    name: str
    qualified_name: str
    support_provider: object | None
    support_signature: SupportSignature
    probabilities: object
    support_origin: SupportOrigin
    lifetime: LotteryLifetime
    persisted_state: str | None
    support_params: ParameterBinding
    probability_params: ParameterBinding
    weight_name: str
    support_provider_name: str | None = None
    node_annotation: str | None = None


@dataclass(frozen=True)
class TransitionOutputInfo:
    """How one genuine target state obtains its next-period value."""

    state: str
    next_state_name: TransitionFunctionName
    qualified_name: str
    producer: OutputProducerRef
    physical_resolver: object | LotteryValue
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
