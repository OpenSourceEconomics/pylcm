"""Assigning every declared constraint exactly one disposition.

A solver declares what it can do with a constraint — which names it can read
before its inner stage, which constraints its own construction already
enforces, which boundaries it can compile — and this module turns those
declarations plus a regime's constraints into one verdict per constraint.

Keeping the assignment here rather than in each solver is what makes the four
dispositions exhaustive in practice and not just in principle. A solver that
decided for itself could silently arrive at none, and a constraint with no
disposition is neither honoured nor refused: it shows up as a wrong policy
rather than as an error.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from _lcm.constraints.dispositions import (
    CompileBoundary,
    ConstraintContext,
    ConstraintDisposition,
    Evaluate,
    EvaluationStage,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.typing import FunctionName


@runtime_checkable
class StructuralProof(Protocol):
    """Decides whether a solver's own construction enforces a constraint."""

    def __call__(
        self, *, constraint: ProcessedConstraint, context: ConstraintContext
    ) -> ProvedByConstruction | Reject | None:
        """Discharge the constraint, refuse it, or decline to judge it.

        Args:
            constraint: The constraint to judge.
            context: What the proof may read about the regime and the phase.

        Returns:
            The verdict, or `None` meaning "not mine" — the compiler then tries
            the next proof.

        """
        ...


@runtime_checkable
class BoundaryCompiler(Protocol):
    """Turns a constraint's boundary into a program a solver can act on."""

    def __call__(
        self, *, constraint: ProcessedConstraint, context: ConstraintContext
    ) -> CompileBoundary | Reject | None:
        """Compile the constraint's boundary, refuse it, or decline to judge it.

        Args:
            constraint: The constraint to compile.
            context: What the compiler may read about the regime and the phase.

        Returns:
            The verdict, or `None` meaning "not mine".

        """
        ...


@runtime_checkable
class ConstraintCapabilities(Protocol):
    """What a solver can do with a constraint it is handed."""

    @property
    def pre_inner_available_names(self) -> frozenset[str] | None:
        """Names the solver can read before its inner stage.

        `None` means the solver imposes no restriction, which is grid search:
        every name is in scope wherever it evaluates. A frozenset is an
        allow-list a constraint's dependencies must be a subset of, and an
        *empty* one means no name is readable at all — a different declaration
        from `None`, and one that refuses every constraint.
        """

    @property
    def evaluation_stage(self) -> EvaluationStage:
        """Where the solver evaluates a constraint it has to call.

        Declared rather than inferred from the allow-list. The two happen to
        line up for the solver families shipped today, and a correlation that
        holds for two and breaks on the third would break without a symptom.
        """

    @property
    def structural_proofs(self) -> tuple[StructuralProof, ...]:
        """Proofs, in the order they are consulted."""

    @property
    def boundary_compilers(self) -> tuple[BoundaryCompiler, ...]:
        """Boundary compilers, in the order they are consulted."""


def compile_constraints(
    *,
    constraints: Mapping[FunctionName, ProcessedConstraint],
    capabilities: ConstraintCapabilities,
    context: ConstraintContext,
) -> MappingProxyType[FunctionName, ConstraintDisposition]:
    """Assign exactly one disposition to every declared constraint.

    Proofs are consulted first, then boundary compilers, and the first to claim
    a constraint decides it. A constraint no capability claims is evaluated when
    the solver can read every name it depends on, and refused when it cannot.

    Args:
        constraints: The regime's normalized constraints.
        capabilities: What the regime's solver can do with a constraint.
        context: What a proof or a boundary compiler may read.

    Returns:
        Immutable mapping of constraint name to its disposition, with one entry
        per constraint.

    Raises:
        TypeError: If a proof or a boundary compiler answers with a verdict
            outside the kinds it is allowed to give.

    """
    return MappingProxyType(
        {
            name: _disposition_of(
                constraint=constraint, capabilities=capabilities, context=context
            )
            for name, constraint in constraints.items()
        }
    )


def _disposition_of(
    *,
    constraint: ProcessedConstraint,
    capabilities: ConstraintCapabilities,
    context: ConstraintContext,
) -> ConstraintDisposition:
    """Return the one disposition the solver's capabilities imply."""
    for proof in capabilities.structural_proofs:
        verdict = proof(constraint=constraint, context=context)
        if verdict is not None:
            _fail_if_verdict_is_outside_the_contract(
                verdict=verdict,
                allowed=(ProvedByConstruction, Reject),
                source=proof,
                allowed_description="prove a constraint or refuse it",
            )
            return verdict
    for boundary_compiler in capabilities.boundary_compilers:
        verdict = boundary_compiler(constraint=constraint, context=context)
        if verdict is not None:
            _fail_if_verdict_is_outside_the_contract(
                verdict=verdict,
                allowed=(CompileBoundary, Reject),
                source=boundary_compiler,
                allowed_description="compile a constraint's boundary or refuse it",
            )
            return verdict
    available = capabilities.pre_inner_available_names
    if available is None or constraint.dependencies <= available:
        return Evaluate(constraint=constraint, stage=capabilities.evaluation_stage)
    return Reject(
        constraint=constraint,
        reason=_unreadable_names_message(
            constraint=constraint, available=available, context=context
        ),
    )


def _unreadable_names_message(
    *,
    constraint: ProcessedConstraint,
    available: frozenset[str],
    context: ConstraintContext,
) -> str:
    """Say which names put the constraint out of the solver's reach."""
    unreadable = sorted(constraint.dependencies - available)
    readable = sorted(available)
    return (
        f"The constraint '{constraint.name}' of regime '{context.regime_name}' "
        f"reads {unreadable}, which the solver cannot supply where it evaluates "
        f"constraints. It has {readable} available there. Restate the constraint "
        f"over those names, or use a solver that evaluates over the whole "
        f"state-action product."
    )


def _fail_if_verdict_is_outside_the_contract(
    *,
    verdict: object,
    allowed: tuple[type, ...],
    source: object,
    allowed_description: str,
) -> None:
    """Refuse a capability that answered with a kind of verdict it may not give."""
    if isinstance(verdict, allowed):
        return
    name = getattr(source, "__name__", repr(source))
    msg = (
        f"The capability '{name}' returned {verdict!r}, but it may only "
        f"{allowed_description}, or return `None` to decline. Deciding where a "
        f"constraint is evaluated belongs to the compiler, so that no "
        f"constraint reaches a stage no capability vouched for."
    )
    raise TypeError(msg)
