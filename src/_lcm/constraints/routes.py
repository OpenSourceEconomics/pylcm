"""One terminal disposition per constraint, per route a solver actually walks.

A *route* is one solver's ordered candidate-production pipeline for one phase:
the sites at which it has candidates in hand, in the order it produces them. A
*site* is one such point, and what distinguishes sites is which names are bound
there and which function pool is in scope — a nesting solver rewrites its pool
between sites, so the same declaration means different work at each.

Disposition is a property of a route, not of a regime. The same borrowing limit
is enforced by construction where an endogenous-grid solve inverts on its
savings grid and evaluated in simulation, where no such grid exists. Asking for
one verdict per `(constraint, regime, phase)` cannot express that, and a nesting
solver has more than one route per phase, so the coarser unit has to drop one of
them — silently, because a constraint that reaches no disposition is neither
honoured nor refused and surfaces as a wrong policy rather than as an error.

The invariant this module exists to hold is therefore:

    for every phase-specialized solver route, every applicable constraint has
    exactly one terminal disposition.

What a constraint needs is read off the site's own pool rather than off the
constraint's surface, so `spendable >= 0` and the same requirement spelled over
`spendable`'s own leaves are disposed of alike.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from _lcm.constraints.dispositions import (
    CompileBoundary,
    ConstraintContext,
    ConstraintDisposition,
    Evaluate,
    EvaluationStage,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.materialize import transitive_arg_names
from _lcm.constraints.processed import ProcessedConstraint
from _lcm.typing import EconFunctionsMapping, FunctionName


@dataclass(frozen=True, kw_only=True)
class ConstraintRouteKey:
    """Which pipeline a disposition belongs to.

    Hashable and compared by value: the key is what a plan's coverage is
    counted over, so two descriptions of the same pipeline must be the same
    key.
    """

    phase: Literal["solve", "simulate"]
    """Phase whose candidates this route produces."""

    period_group: tuple[int, ...] | None
    """Periods sharing one build of this route, or `None` when it does not vary.

    `None` says the route is the same in every period the regime is active, and
    is what a solver whose pool no period resolves differently declares. A
    tuple says the opposite — that this route was built for these periods while
    a sibling route was built for others — and only a solver that genuinely
    rebuilds per group emits one.

    Keeping the two distinguishable is the point. Emitting one route per group
    from a solver that does not vary would put N entries in the plan where
    there is one fact, and a coverage count over those pairs would read a
    constant as evidence of grouping.

    Caching compiled cores by a period-dependent key is a different axis and
    not a reason to emit tuples: what varies there is the program, while a
    route's identity is which names a site can read. A solver may keep one core
    per age signature and still walk one route.
    """

    solver_path: tuple[str, ...]
    """The nest of solvers producing the candidates, outermost first.

    `("grid_search",)` for a flat solve; `("negm", "adjuster")` and
    `("negm", "keeper")` for the two branches a nested solve produces.
    """


@dataclass(frozen=True, kw_only=True, eq=False)
class ConstraintSite:
    """One point along a route at which a constraint could be met."""

    stage: EvaluationStage
    """What kind of candidate is in hand here."""

    function_pool: EconFunctionsMapping
    """The functions in scope at this site.

    The pool a *nesting* solver rewrites going into a branch, not the regime's
    declared pool: a constraint's leaves are resolved through this, so handing
    over the unrewritten pool would classify against a scope the site does not
    have.
    """

    available_names: frozenset[str] | None
    """Leaf names a constraint may be evaluated over here. Three declarations:

    - `None` — no restriction, which is grid search: every name is in scope.
    - an *empty* frozenset — this site evaluates nothing. Every constraint
      falls through to the next site, while the site's proofs and compilers
      still run, so it stays a place a constraint can be discharged.
    - a non-empty frozenset — the allow-list a constraint's leaves must be a
      subset of.

    This is the site's only statement about evaluation, deliberately. A second
    field saying the same thing in another form — a name the site binds, a flag
    — would agree with this one until it did not, and nothing observable would
    say which had rotted. So a name the site binds is listed here or it is not
    readable: forgetting it costs a refusal that names the missing name at
    model build, which is loud, where the other way round the site would claim
    a constraint nothing there evaluates.
    """

    structural_proofs: tuple[StructuralProof, ...] = ()
    """Proofs consulted here, in order."""

    boundary_compilers: tuple[BoundaryCompiler, ...] = ()
    """Boundary compilers consulted here, in order."""


@dataclass(frozen=True, kw_only=True, eq=False)
class ConstraintRoute:
    """One solver's ordered candidate-production pipeline for one phase."""

    key: ConstraintRouteKey
    """Which pipeline this is."""

    sites: tuple[ConstraintSite, ...]
    """The points at which a constraint can be met, in the order they arise.

    Not a description of the pipeline. A solver's candidate production may have
    any number of stages, and only those at which a constraint can be
    evaluated, proved, or compiled belong here — a stage that can decide none
    of the three cannot change a plan, so declaring it would put pipeline shape
    into the field the ledger's decisions are counted over. `solver_path` and
    `stage` are where the pipeline is named.

    An empty tuple is a solver that offers nowhere to meet a constraint and
    refuses every one it is handed. That is a declaration, not an omission.
    """


@dataclass(frozen=True, eq=False)
class BoundConstraint:
    """A constraint resolved against the pool of one site."""

    constraint: ProcessedConstraint
    """The constraint being disposed of."""

    site: ConstraintSite
    """The site it was resolved against."""

    transitive_inputs: frozenset[str]
    """The leaf names it needs once resolved through the site's pool."""


@runtime_checkable
class StructuralProof(Protocol):
    """Decides whether a solver's construction already enforces a constraint."""

    def __call__(
        self, *, bound: BoundConstraint, context: ConstraintContext
    ) -> ProvedByConstruction | Reject | None:
        """Discharge the constraint, refuse it, or decline to judge it.

        Args:
            bound: The constraint resolved against this site.
            context: What the proof may read about the regime and the phase.

        Returns:
            The verdict, or `None` meaning "not mine" — the planner then tries
            the next proof, then the site's compilers, then the next site.

        """
        ...


@runtime_checkable
class BoundaryCompiler(Protocol):
    """Turns a constraint's boundary into a program a solver can act on."""

    def __call__(
        self, *, bound: BoundConstraint, context: ConstraintContext
    ) -> CompileBoundary | Reject | None:
        """Compile the constraint's boundary, refuse it, or decline to judge it.

        Args:
            bound: The constraint resolved against this site.
            context: What the compiler may read about the regime and the phase.

        Returns:
            The verdict, or `None` meaning "not mine".

        """
        ...


@dataclass(frozen=True, kw_only=True, eq=False)
class ConstraintPlanEntry:
    """What one route decided about one constraint."""

    constraint_name: FunctionName
    """The name the constraint was declared under."""

    route: ConstraintRouteKey
    """The route that decided it."""

    disposition: ConstraintDisposition
    """The one terminal verdict."""


@dataclass(frozen=True, eq=False)
class ConstraintPlan:
    """Every constraint's disposition on every route that applies to it."""

    entries: tuple[ConstraintPlanEntry, ...]
    """One entry per (constraint, route) pair."""

    def merge(self, other: ConstraintPlan) -> ConstraintPlan:
        """Return the plan holding both plans' entries.

        Args:
            other: The plan to merge in, over disjoint routes.

        Returns:
            The combined plan.

        """
        return ConstraintPlan(entries=(*self.entries, *other.entries))


def plan_constraints(
    *,
    constraints: Mapping[FunctionName, ProcessedConstraint],
    routes: tuple[ConstraintRoute, ...],
    context: ConstraintContext,
) -> ConstraintPlan:
    """Assign every constraint one terminal disposition on every route.

    Args:
        constraints: The phase's normalized constraints.
        routes: The routes the phase's solver walks, each belonging to the
            phase named by `context`.
        context: What a proof or a compiler may read about regime and phase.

    Returns:
        The plan, holding exactly one entry per (constraint, route) pair.

    Raises:
        ValueError: If a route belongs to another phase, or if two routes share
            one key.
        TypeError: If a proof or a compiler answers with a kind of verdict it
            is not allowed to give.

    """
    _fail_if_a_route_is_out_of_phase(routes=routes, context=context)
    _fail_if_two_routes_share_a_key(routes=routes)
    entries = tuple(
        ConstraintPlanEntry(
            constraint_name=name,
            route=route.key,
            disposition=_disposition_along(
                constraint=constraint, route=route, context=context
            ),
        )
        for route in routes
        for name, constraint in constraints.items()
    )
    _fail_if_coverage_is_incomplete(
        entries=entries, constraints=constraints, routes=routes
    )
    return ConstraintPlan(entries=entries)


def _disposition_along(
    *,
    constraint: ProcessedConstraint,
    route: ConstraintRoute,
    context: ConstraintContext,
) -> ConstraintDisposition:
    """Return the verdict the first site along the route to claim it gives.

    A site is asked in turn to prove the constraint, to compile its boundary,
    and finally to evaluate it; the first that can decides, and a site that can
    do none of the three hands it to the next.

    Evaluating at an earlier site therefore beats proving at a later one. That
    ordering is deliberate: evaluating a constraint some later construction
    also enforces costs a redundant predicate, while the opposite mistake —
    treating a constraint as discharged by a construction that is not actually
    reached — is silent.
    """
    shortfalls: list[str] = []
    for site in route.sites:
        bound = BoundConstraint(
            constraint=constraint,
            site=site,
            transitive_inputs=transitive_arg_names(
                constraint=constraint, pool=site.function_pool
            ),
        )
        for proof in site.structural_proofs:
            verdict = proof(bound=bound, context=context)
            if verdict is not None:
                _fail_if_verdict_is_outside_the_contract(
                    verdict=verdict,
                    allowed=(ProvedByConstruction, Reject),
                    source=proof,
                    allowed_description="prove a constraint or refuse it",
                )
                return verdict
        for boundary_compiler in site.boundary_compilers:
            verdict = boundary_compiler(bound=bound, context=context)
            if verdict is not None:
                _fail_if_verdict_is_outside_the_contract(
                    verdict=verdict,
                    allowed=(CompileBoundary, Reject),
                    source=boundary_compiler,
                    allowed_description="compile a constraint's boundary or refuse it",
                )
                return verdict
        if _site_can_evaluate(bound=bound):
            return Evaluate(constraint=constraint, stage=site.stage)
        shortfalls.append(_shortfall_at(bound=bound))
    return Reject(
        constraint=constraint,
        reason=_unmet_message(
            constraint=constraint, route=route, context=context, shortfalls=shortfalls
        ),
    )


def _site_can_evaluate(*, bound: BoundConstraint) -> bool:
    """Whether the site evaluates constraints, and every leaf this one needs.

    That a site evaluates nothing is a rule here rather than a consequence of
    the subset test, because the subset test gets that case wrong: a constraint
    needing no name at all is trivially within an empty allow-list, so deriving
    the answer would hand it to a kernel that calls no predicate. Degenerate to
    write and perfectly declarable — two literals compared, or a zero-argument
    callable — and silent when it happens.
    """
    available = bound.site.available_names
    if available is None:
        return True
    if not available:
        return False
    return bound.transitive_inputs <= available


def _shortfall_at(*, bound: BoundConstraint) -> str:
    """Say why one site could not claim the constraint."""
    site = bound.site
    if site.available_names is not None and not site.available_names:
        return f"at '{site.stage}' nothing is evaluated"
    missing = sorted(bound.transitive_inputs - (site.available_names or frozenset()))
    return f"at '{site.stage}' it still needs {missing}"


def _unmet_message(
    *,
    constraint: ProcessedConstraint,
    route: ConstraintRoute,
    context: ConstraintContext,
    shortfalls: list[str],
) -> str:
    """Say why a route could meet the constraint at none of its sites."""
    label = _route_label(key=route.key)
    opening = (
        f"The constraint '{constraint.name}' of regime "
        f"'{context.regime_name}' cannot be met on the {label} route"
    )
    if not shortfalls:
        return (
            f"{opening}, which offers no site at which a constraint can be "
            "evaluated. Encode the requirement in the solver's own "
            "construction, or use a solver that evaluates constraints."
        )
    per_site = "; ".join(shortfalls)
    return (
        f"{opening}: {per_site}. Restate it over names one of those sites can "
        "read, or use a solver that evaluates over the whole state-action "
        "product."
    )


def _route_label(*, key: ConstraintRouteKey) -> str:
    """Name a route the way an error message should quote it."""
    return f"`{'/'.join(key.solver_path)} {key.phase}`"


def _fail_if_a_route_is_out_of_phase(
    *, routes: tuple[ConstraintRoute, ...], context: ConstraintContext
) -> None:
    """Refuse a route belonging to a phase other than the one being planned."""
    out_of_phase = [
        _route_label(key=route.key)
        for route in routes
        if route.key.phase != context.phase
    ]
    if not out_of_phase:
        return
    msg = (
        f"Planning the '{context.phase}' phase of regime "
        f"'{context.regime_name}' was handed {out_of_phase}, which belong to "
        "another phase. A plan is built per phase against that phase's own "
        "constraints and function pool, so a route from the other one would be "
        "classified against a scope it does not have."
    )
    raise ValueError(msg)


def _fail_if_two_routes_share_a_key(*, routes: tuple[ConstraintRoute, ...]) -> None:
    """Refuse two routes claiming to be the same pipeline."""
    counts = Counter(route.key for route in routes)
    duplicated = [_route_label(key=key) for key, n in counts.items() if n > 1]
    if not duplicated:
        return
    msg = (
        f"Two routes share the key {duplicated}. A plan holds one entry per "
        "(constraint, route) pair, so a repeated key would let one pipeline's "
        "verdict overwrite another's with nothing to say which survived."
    )
    raise ValueError(msg)


def _fail_if_coverage_is_incomplete(
    *,
    entries: tuple[ConstraintPlanEntry, ...],
    constraints: Mapping[FunctionName, ProcessedConstraint],
    routes: tuple[ConstraintRoute, ...],
) -> None:
    """Refuse a plan that does not decide every constraint on every route."""
    expected = {(name, route.key) for name in constraints for route in routes}
    observed = {(entry.constraint_name, entry.route) for entry in entries}
    if observed == expected:
        return
    missing = sorted((name, _route_label(key=key)) for name, key in expected - observed)
    unasked = sorted((name, _route_label(key=key)) for name, key in observed - expected)
    msg = (
        f"The constraint plan is not exhaustive: {missing} received no "
        f"disposition and {unasked} were decided without being declared. A "
        "constraint with no disposition is neither honoured nor refused, so it "
        "surfaces as a wrong policy rather than as an error."
    )
    raise ValueError(msg)


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
        "constraint is evaluated belongs to the planner, so that no constraint "
        "reaches a site no capability vouched for."
    )
    raise TypeError(msg)
