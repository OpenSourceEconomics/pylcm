"""Every constraint a regime declares gets exactly one disposition.

A solver meets a constraint in one of four ways: it evaluates it, its own
construction already enforces it, it compiles it into a boundary it splits its
candidate grid on, or it refuses it. Those four are exhaustive by design — the
state a solver must never reach is holding a constraint it neither honours nor
refuses, because that one is silent, and a dropped constraint shows up as a
wrong policy rather than as an error.

The compiler is what makes the four exhaustive: it assigns a disposition to
every declared constraint, taking the solver's own capabilities as the only
input that varies between solvers.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import pytest

from _lcm.constraints.capabilities import (
    BoundaryCompiler,
    StructuralProof,
    compile_constraints,
)
from _lcm.constraints.dispositions import (
    BoundaryProgram,
    CompileBoundary,
    ConstraintContext,
    Evaluate,
    EvaluationStage,
    Proof,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.processed import ProcessedConstraint, normalize_constraints
from lcm import ref
from lcm.typing import FloatND


@dataclass(frozen=True, kw_only=True)
class _Capabilities:
    """A solver's constraint capabilities, spelled out for one test."""

    pre_inner_available_names: frozenset[str] | None = None
    evaluation_stage: EvaluationStage = "state_action"
    structural_proofs: tuple[StructuralProof, ...] = ()
    boundary_compilers: tuple[BoundaryCompiler, ...] = ()


@dataclass
class _RecordingProof:
    """A structural proof that records whether it was consulted."""

    verdict_reason: str | None
    consulted: list[str] = field(default_factory=list)

    def __call__(
        self,
        *,
        constraint: ProcessedConstraint,
        context: ConstraintContext,  # noqa: ARG002
    ) -> ProvedByConstruction | None:
        self.consulted.append(constraint.name)
        if self.verdict_reason is None:
            return None
        return ProvedByConstruction(
            constraint=constraint, proof=Proof(reason=self.verdict_reason)
        )


def _context() -> ConstraintContext:
    return ConstraintContext(
        regime_name="saving",
        phase="solve",
        grids=MappingProxyType({}),
        function_names=frozenset({"savings"}),
        param_names=frozenset(),
    )


def _borrowing() -> MappingProxyType[str, ProcessedConstraint]:
    return normalize_constraints(constraints={"borrowing": ref("savings") >= 0.0})


def test_a_constraint_no_capability_claims_is_evaluated() -> None:
    """A solver that restricts nothing evaluates whatever it is handed."""
    compiled = compile_constraints(
        constraints=_borrowing(), capabilities=_Capabilities(), context=_context()
    )

    assert isinstance(compiled["borrowing"], Evaluate)


def test_an_evaluated_constraint_carries_the_solvers_own_stage() -> None:
    """Where the solver evaluates is the solver's statement, not an inference."""
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(evaluation_stage="discrete_combo"),
        context=_context(),
    )

    assert compiled["borrowing"].stage == "discrete_combo"  # ty: ignore[unresolved-attribute]


def test_a_structural_proof_takes_the_constraint_out_of_evaluation() -> None:
    """A constraint the solver's construction enforces is not evaluated again."""
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(
            structural_proofs=(_RecordingProof(verdict_reason="the grid starts here"),)
        ),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], ProvedByConstruction)


def test_a_proof_that_declines_hands_the_constraint_to_the_next_one() -> None:
    """Returning nothing means 'not mine', so the next proof still gets a turn."""
    declining = _RecordingProof(verdict_reason=None)
    claiming = _RecordingProof(verdict_reason="the grid starts here")

    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(structural_proofs=(declining, claiming)),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], ProvedByConstruction)


def test_a_proof_after_the_one_that_claimed_is_not_consulted() -> None:
    """The first proof to claim a constraint decides it.

    Proofs are ordered, and a later proof seeing a constraint an earlier one
    already disposed of could reach a second verdict on the same object. That
    the second is never asked is what makes the order meaningful rather than
    incidental.
    """
    claiming = _RecordingProof(verdict_reason="the grid starts here")
    later = _RecordingProof(verdict_reason="a different reason")

    compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(structural_proofs=(claiming, later)),
        context=_context(),
    )

    assert later.consulted == []


def test_a_constraint_reading_an_unavailable_name_is_refused() -> None:
    """A solver that cannot read a name refuses the constraint rather than drop it.

    The alternative — accepting it and evaluating whatever is in scope — is the
    failure this compiler exists to prevent, because it shows up as a wrong
    policy instead of as an error.
    """
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(pre_inner_available_names=frozenset({"work"})),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], Reject)


def test_the_refusal_names_what_the_constraint_could_not_read() -> None:
    """The message carries the name the solver cannot supply."""
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(pre_inner_available_names=frozenset({"work"})),
        context=_context(),
    )

    assert "savings" in compiled["borrowing"].reason  # ty: ignore[unresolved-attribute]


def test_a_constraint_within_the_allow_list_is_evaluated() -> None:
    """An allow-list restricts what may be read, it does not refuse everything."""
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(pre_inner_available_names=frozenset({"savings"})),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], Evaluate)


def test_an_empty_allow_list_is_not_read_as_no_restriction() -> None:
    """Allowing nothing and restricting nothing are different declarations.

    A solver that can read no name before its inner stage refuses every
    constraint; a solver that restricts nothing evaluates every constraint.
    Spelling the second as an empty set would make the two indistinguishable.
    """
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(pre_inner_available_names=frozenset()),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], Reject)


def _boundary_compiler(
    *,
    constraint: ProcessedConstraint,
    context: ConstraintContext,  # noqa: ARG001
) -> CompileBoundary:
    surfaces = constraint.boundary_surfaces or ()
    return CompileBoundary(
        constraint=constraint,
        program=BoundaryProgram(surfaces=surfaces, payload=None),
    )


def test_a_boundary_compiler_claims_a_constraint_no_proof_took() -> None:
    """A solver that splits its grid on a boundary compiles it rather than calls it."""
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(boundary_compilers=(_boundary_compiler,)),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], CompileBoundary)


def test_a_proof_is_consulted_before_a_boundary_compiler() -> None:
    """Proving a constraint away beats compiling it, so proofs run first."""
    compiled = compile_constraints(
        constraints=_borrowing(),
        capabilities=_Capabilities(
            structural_proofs=(_RecordingProof(verdict_reason="already enforced"),),
            boundary_compilers=(_boundary_compiler,),
        ),
        context=_context(),
    )

    assert isinstance(compiled["borrowing"], ProvedByConstruction)


def _affordable(consumption: FloatND, wealth: FloatND) -> FloatND:
    return consumption <= wealth


def test_every_declared_constraint_is_disposed_of() -> None:
    """No constraint is left without a disposition, whatever its shape.

    An opaque predicate, a structured condition, and a proved one take three
    different routes through the compiler, and the count is what says none of
    them fell out on the way.
    """
    constraints = normalize_constraints(
        constraints={
            "borrowing": ref("savings") >= 0.0,
            "affordable": _affordable,
            "solvent": ref("wealth") >= 0.0,
        }
    )

    compiled = compile_constraints(
        constraints=constraints, capabilities=_Capabilities(), context=_context()
    )

    assert set(compiled) == {"borrowing", "affordable", "solvent"}


def test_declaring_no_constraints_compiles_to_nothing() -> None:
    """An empty pool is nothing to classify, not an error."""
    compiled = compile_constraints(
        constraints=MappingProxyType({}),
        capabilities=_Capabilities(),
        context=_context(),
    )

    assert dict(compiled) == {}


def _proof_returning_a_stage(
    *,
    constraint: ProcessedConstraint,
    context: ConstraintContext,  # noqa: ARG001
) -> object:
    """A capability that answers with a disposition it is not allowed to give."""
    return Evaluate(constraint=constraint, stage="state_action")


def test_a_proof_answering_outside_its_contract_is_an_error() -> None:
    """A capability returning the wrong kind of verdict fails loudly.

    A proof may prove or refuse; deciding that a constraint should be evaluated
    is the compiler's job, not a proof's. Accepting it here would let a solver
    route a constraint to a stage no capability vouched for.
    """
    with pytest.raises(TypeError, match="_proof_returning_a_stage"):
        compile_constraints(
            constraints=_borrowing(),
            # The type checker reads this violation off the annotations, which
            # is the point: the runtime refusal is the backstop for a solver
            # whose capability is not checked statically.
            capabilities=_Capabilities(structural_proofs=(_proof_returning_a_stage,)),  # ty: ignore[invalid-argument-type]
            context=_context(),
        )
