"""Every constraint gets one terminal disposition on every route it applies to.

A route is a solver's ordered candidate-production pipeline for one phase — the
sites at which it has candidates in hand, in the order it produces them. The
same constraint can legitimately be met differently on different routes: a
borrowing limit is enforced by construction where an endogenous-grid solve
inverts on its savings grid, and evaluated in simulation where there is no such
grid. What must never happen is a constraint reaching *no* disposition on a
route that applies to it, because that one is silent — it surfaces as a wrong
policy rather than as an error.

Which names a constraint needs is read off the site's own function pool rather
than off the constraint's surface, so two spellings of the same requirement are
disposed of alike.
"""

from dataclasses import dataclass, field
from types import MappingProxyType

import pytest

from _lcm.constraints.dispositions import (
    BoundaryProgram,
    CompileBoundary,
    ConstraintContext,
    Evaluate,
    Proof,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.ir import Compare, Condition, Const
from _lcm.constraints.materialize import transitive_arg_names
from _lcm.constraints.processed import ProcessedConstraint, normalize_constraints
from _lcm.constraints.routes import (
    BoundConstraint,
    ConstraintPlan,
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
    plan_constraints,
)
from lcm import ref
from lcm.typing import FloatND


def _spendable(wealth: FloatND, consumption: FloatND) -> FloatND:
    """A helper the constraint reaches through rather than naming its leaves."""
    return wealth - consumption


def _pool() -> MappingProxyType[str, object]:
    return MappingProxyType({"spendable": _spendable})


def _site(
    *,
    stage: str = "state_action",
    available_names: frozenset[str] | None = None,
    pool: MappingProxyType[str, object] | None = None,
    structural_proofs: tuple[object, ...] = (),
    boundary_compilers: tuple[object, ...] = (),
) -> ConstraintSite:
    return ConstraintSite(
        stage=stage,  # ty: ignore[invalid-argument-type]
        function_pool=_pool() if pool is None else pool,  # ty: ignore[invalid-argument-type]
        available_names=available_names,
        structural_proofs=structural_proofs,  # ty: ignore[invalid-argument-type]
        boundary_compilers=boundary_compilers,  # ty: ignore[invalid-argument-type]
    )


def _key(
    *, phase: str = "solve", solver_path: tuple[str, ...] = ("grid_search",)
) -> ConstraintRouteKey:
    return ConstraintRouteKey(
        phase=phase,  # ty: ignore[invalid-argument-type]
        period_group=None,
        solver_path=solver_path,
    )


def _route(*sites: ConstraintSite, phase: str = "solve", path: str = "grid_search"):
    return ConstraintRoute(key=_key(phase=phase, solver_path=(path,)), sites=sites)


def _plan(*routes: ConstraintRoute, phase: str = "solve") -> ConstraintPlan:
    """Plan the borrowing constraint over `routes`, in one line at each call site."""
    return plan_constraints(
        constraints=_borrowing(), routes=routes, context=_context(phase=phase)
    )


def _context(*, phase: str = "solve") -> ConstraintContext:
    return ConstraintContext(
        regime_name="working",
        phase=phase,  # ty: ignore[invalid-argument-type]
        grids=MappingProxyType({}),
        function_names=frozenset({"spendable"}),
        param_names=frozenset(),
    )


def _borrowing() -> MappingProxyType[str, ProcessedConstraint]:
    """`spendable >= 0`, which names a helper rather than the helper's leaves."""
    return normalize_constraints(constraints={"borrowing": ref("spendable") >= 0.0})


@dataclass
class _RecordingProof:
    """A structural proof that records the sites at which it was consulted."""

    verdict_reason: str | None
    consulted: list[str] = field(default_factory=list)

    def __call__(
        self,
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG002
    ) -> ProvedByConstruction | None:
        self.consulted.append(bound.site.stage)
        if self.verdict_reason is None:
            return None
        return ProvedByConstruction(
            constraint=bound.constraint, proof=Proof(reason=self.verdict_reason)
        )


def test_plan_gives_every_constraint_an_entry_on_every_route() -> None:
    """A plan holds exactly one entry per (constraint, route) pair."""
    routes = (
        _route(_site(), path="grid_search"),
        _route(_site(), path="other"),
    )

    plan = plan_constraints(constraints=_borrowing(), routes=routes, context=_context())

    assert {(entry.constraint_name, entry.route) for entry in plan.entries} == {
        ("borrowing", routes[0].key),
        ("borrowing", routes[1].key),
    }


def test_a_constraint_is_evaluated_where_its_transitive_leaves_are_readable() -> None:
    """What a constraint needs is its leaves through the site's pool, not its surface.

    `spendable >= 0` names a helper. A site that can read `wealth` and
    `consumption` can evaluate it, and one that could only read the name
    `spendable` could not — so reading the surface would classify two spellings
    of one requirement differently.
    """
    route = _route(_site(available_names=frozenset({"wealth", "consumption"})))

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, Evaluate)


def test_a_constraint_falls_through_to_the_first_site_that_can_read_it() -> None:
    """Sites are walked in the order the solver produces candidates."""
    route = _route(
        _site(stage="discrete_combo", available_names=frozenset()),
        _site(
            stage="savings_stage", available_names=frozenset({"wealth", "consumption"})
        ),
    )

    plan = _plan(route)

    assert plan.entries[0].disposition.stage == "savings_stage"  # ty: ignore[unresolved-attribute]


def test_a_constraint_no_site_can_read_is_refused() -> None:
    """Falling off the end of a route is a refusal, never silence."""
    route = _route(_site(available_names=frozenset({"wealth"})))

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, Reject)


def test_a_refusal_names_the_route_it_could_not_be_met_on() -> None:
    """The message has to say which pipeline could not meet it."""
    route = _route(_site(available_names=frozenset({"wealth"})), path="dcegm")

    plan = _plan(route)

    assert "dcegm" in plan.entries[0].disposition.reason  # ty: ignore[unresolved-attribute]


def test_one_constraint_can_carry_different_dispositions_on_two_routes() -> None:
    """A borrowing limit is proved where a grid enforces it and evaluated elsewhere."""
    proving = _route(
        _site(
            stage="savings_stage",
            structural_proofs=(_RecordingProof("the grid starts there"),),
        ),
        path="dcegm",
    )
    evaluating = _route(_site(stage="simulation"), phase="simulate", path="dcegm")

    plan = plan_constraints(
        constraints=_borrowing(), routes=(proving,), context=_context()
    ).merge(
        plan_constraints(
            constraints=_borrowing(),
            routes=(evaluating,),
            context=_context(phase="simulate"),
        )
    )

    assert [type(entry.disposition) for entry in plan.entries] == [
        ProvedByConstruction,
        Evaluate,
    ]


def test_a_plan_can_be_narrowed_to_one_solver_path() -> None:
    """A nested solver reads only the entries for the branch it is building."""
    adjuster = _route(_site(), path="adjuster")
    keeper = _route(_site(), path="keeper")
    plan = _plan(adjuster, keeper)

    narrowed = plan.for_solver_path(solver_path=("adjuster",))

    assert tuple(entry.route for entry in narrowed.entries) == (adjuster.key,)


def test_a_plan_exposes_only_its_compiled_boundary_dispositions() -> None:
    """A solver consumes boundary programs without reinterpreting other verdicts."""

    def compiler(
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG001
    ) -> CompileBoundary:
        return CompileBoundary(
            constraint=bound.constraint,
            program=BoundaryProgram(surfaces=(), payload=None),
        )

    compiled = _route(_site(boundary_compilers=(compiler,)), path="compiled")
    evaluated = _route(_site(), path="evaluated")
    plan = _plan(compiled, evaluated)

    assert plan.compiled_boundaries == (plan.entries[0].disposition,)


def test_a_proof_at_a_site_takes_the_constraint_out_of_evaluation() -> None:
    """A constraint the construction enforces is not evaluated again."""
    route = _route(_site(structural_proofs=(_RecordingProof("the grid starts there"),)))

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, ProvedByConstruction)


def test_a_proof_is_consulted_only_at_the_site_that_declares_it() -> None:
    """Proofs are a property of a site, not of the route as a whole."""
    proof = _RecordingProof(verdict_reason=None)
    route = _route(
        _site(stage="discrete_combo", available_names=frozenset()),
        _site(stage="savings_stage", structural_proofs=(proof,)),
    )

    _plan(route)

    assert proof.consulted == ["savings_stage"]


def test_a_boundary_compiler_at_a_site_compiles_the_constraint() -> None:
    """A solver that splits its grid on a boundary says so through the plan."""

    def compiler(
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG001
    ) -> CompileBoundary:
        return CompileBoundary(
            constraint=bound.constraint,
            program=BoundaryProgram(surfaces=(), payload=None),
        )

    route = _route(_site(boundary_compilers=(compiler,)))

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, CompileBoundary)


def test_a_capability_answering_outside_its_contract_is_refused() -> None:
    """Deciding *where* a constraint is evaluated belongs to the planner."""

    def rogue(
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG001
    ) -> Evaluate:
        return Evaluate(constraint=bound.constraint, stage="state_action")

    route = _route(_site(structural_proofs=(rogue,)))

    with pytest.raises(TypeError, match="may only"):
        plan_constraints(constraints=_borrowing(), routes=(route,), context=_context())


def test_a_route_from_another_phase_is_refused() -> None:
    """A route must belong to the phase it is planned in.

    A plan is built per phase, against that phase's own constraints and
    function pool, so a route from the other one would be classified against a
    scope it does not have.
    """
    route = _route(_site(), phase="simulate")

    with pytest.raises(ValueError, match="phase"):
        plan_constraints(constraints=_borrowing(), routes=(route,), context=_context())


def test_a_route_with_no_sites_refuses_every_constraint() -> None:
    """A solver that offers nowhere to meet a constraint refuses it explicitly.

    This is plain EGM: it evaluates no user constraint at all. Declaring an
    empty route says that once, rather than leaving each constraint unmentioned.
    """
    route = _route(path="egm")

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, Reject)


def test_a_refusal_names_what_the_constraint_could_not_read() -> None:
    """The message carries the leaves the route could not supply, per site."""
    route = _route(_site(available_names=frozenset({"wealth"})))

    plan = _plan(route)

    assert "consumption" in plan.entries[0].disposition.reason  # ty: ignore[unresolved-attribute]


def test_a_site_restricting_nothing_evaluates_whatever_it_is_handed() -> None:
    """`None` is no restriction, which is grid search: every name is in scope."""
    route = _route(_site(available_names=None))

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, Evaluate)


def test_an_empty_allow_list_is_not_read_as_no_restriction() -> None:
    """An empty allow-list is the declaration that a site evaluates nothing.

    Restricting nothing and evaluating nothing are opposite statements, and a
    site makes exactly one of them: `None` evaluates every constraint handed to
    it, an empty frozenset evaluates none. Spelling the first as an empty set
    would make the two indistinguishable.
    """
    route = _route(_site(available_names=frozenset()))

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, Reject)


def test_a_proof_that_declines_hands_the_constraint_to_the_next_one() -> None:
    """Returning nothing means 'not mine', so the next proof still gets a turn."""
    route = _route(
        _site(
            structural_proofs=(
                _RecordingProof(verdict_reason=None),
                _RecordingProof(verdict_reason="the grid starts there"),
            )
        )
    )

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, ProvedByConstruction)


def test_a_proof_after_the_one_that_claimed_is_not_consulted() -> None:
    """The first capability to claim a constraint decides it.

    A later proof seeing a constraint an earlier one already disposed of could
    reach a second verdict on the same object. That it is never asked is what
    makes the order meaningful rather than incidental.
    """
    later = _RecordingProof(verdict_reason="a different reason")
    route = _route(
        _site(structural_proofs=(_RecordingProof("the grid starts there"), later))
    )

    _plan(route)

    assert later.consulted == []


def test_a_proof_is_consulted_before_a_boundary_compiler() -> None:
    """Proving a constraint away beats compiling it, so proofs run first."""

    def compiler(
        *,
        bound: BoundConstraint,
        context: ConstraintContext,  # noqa: ARG001
    ) -> CompileBoundary:
        return CompileBoundary(
            constraint=bound.constraint,
            program=BoundaryProgram(surfaces=(), payload=None),
        )

    route = _route(
        _site(
            structural_proofs=(_RecordingProof("already enforced"),),
            boundary_compilers=(compiler,),
        )
    )

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, ProvedByConstruction)


def _affordable(consumption: FloatND, wealth: FloatND) -> FloatND:
    return consumption <= wealth


def test_every_declared_constraint_is_disposed_of_whatever_its_shape() -> None:
    """An opaque predicate, a structured condition and a helper-wrapped one all land.

    The three take different paths through the planner, and the count is what
    says none of them fell out on the way.
    """
    constraints = normalize_constraints(
        constraints={
            "borrowing": ref("spendable") >= 0.0,
            "affordable": _affordable,
            "solvent": ref("wealth") >= 0.0,
        }
    )

    plan = plan_constraints(
        constraints=constraints, routes=(_route(_site()),), context=_context()
    )

    assert {entry.constraint_name for entry in plan.entries} == {
        "borrowing",
        "affordable",
        "solvent",
    }


def test_declaring_no_constraints_plans_to_nothing() -> None:
    """An empty pool is nothing to decide, not an error."""
    plan = plan_constraints(
        constraints=MappingProxyType({}), routes=(_route(_site()),), context=_context()
    )

    assert plan.entries == ()


def test_two_routes_claiming_to_be_the_same_pipeline_are_refused() -> None:
    """A repeated key would let one pipeline's verdict overwrite another's."""
    route = _route(_site())

    with pytest.raises(ValueError, match="share the key"):
        plan_constraints(
            constraints=_borrowing(), routes=(route, route), context=_context()
        )


def test_a_site_that_evaluates_nothing_still_runs_its_proofs() -> None:
    """Evaluating nothing bars calling a constraint, not discharging one.

    An endogenous-grid branch enforces its borrowing limit through the savings
    grid it inverts on while calling no predicate at all, so a site that
    evaluates nothing is still where that proof belongs.
    """
    route = _route(
        _site(
            available_names=frozenset(),
            structural_proofs=(_RecordingProof("the grid starts there"),),
        )
    )

    plan = _plan(route)

    assert isinstance(plan.entries[0].disposition, ProvedByConstruction)


def test_a_dependency_free_constraint_is_not_evaluated_where_nothing_is() -> None:
    """Needing no name is not the same as a site being able to call it.

    A site that evaluates nothing supplies no leaf, and a constraint that asks
    for no leaf is trivially within that. Deriving evaluability from the subset
    test alone would hand such a constraint to a kernel that calls no
    predicate, which is the silent outcome the plan exists to prevent.
    """
    constraints = normalize_constraints(
        constraints={
            "always": Condition(
                expression=Compare(left=Const(1.0), op=">=", right=Const(0.0))
            )
        }
    )

    plan = plan_constraints(
        constraints=constraints,
        routes=(_route(_site(available_names=frozenset())),),
        context=_context(),
    )

    assert isinstance(plan.entries[0].disposition, Reject)


def test_the_dependency_free_witness_really_reads_nothing() -> None:
    """The witness above is only a witness if it needs no name at all.

    A constraint that turned out to have a dependency would be refused at an
    empty allow-list for the ordinary reason, and the test above would pass
    without touching the case it is named for.
    """
    constraints = normalize_constraints(
        constraints={
            "always": Condition(
                expression=Compare(left=Const(1.0), op=">=", right=Const(0.0))
            )
        }
    )

    site = _site()

    assert (
        transitive_arg_names(
            constraint=constraints["always"],
            pool=site.function_pool,
        )
        == frozenset()
    )


def test_the_transitive_walk_reports_the_names_a_constraint_does_read() -> None:
    """The empty answer above is evidence only if the walk can return a full one.

    The pool that makes the control necessary is specifically a small one. With
    a rich pool a broken walk returning `frozenset()` would show up somewhere
    else; here it is indistinguishable from correct behaviour on every input,
    so an empty answer carries no information unless something in the same run
    comes back non-empty.
    """
    site = _site()

    assert transitive_arg_names(
        constraint=_borrowing()["borrowing"],
        pool=site.function_pool,
    ) == frozenset({"wealth", "consumption"})
