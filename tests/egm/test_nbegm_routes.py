"""What the case-piece solvers declare can happen to a constraint they are handed.

The NBEGM kernels invert the Euler equation on a savings grid and evaluate no
user constraint at any point, so their declaration is the strongest one the
language allows: no name is readable where a constraint would be called. Every
constraint therefore reaches a disposition of `Reject` unless a proof claims it
first, and the one proof they carry is the borrowing limit their own savings
grid already enforces.

The proof keys on the comparison the declaration stands for, not on the type of
object the user happened to construct, so a bound written out by hand is proved
exactly as the convenience constructor's is.
"""

from dataclasses import fields
from types import MappingProxyType

import jax.numpy as jnp
import pytest

import lcm
from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.dispositions import (
    ConstraintContext,
    Evaluate,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.materialize import transitive_arg_names
from _lcm.constraints.processed import normalize_constraints
from _lcm.constraints.routes import ConstraintRoute, plan_constraints
from _lcm.egm.nbegm_routes import case_piece_routes
from _lcm.engine import VariableInfo, Variables
from _lcm.solution.contract import (
    ConstraintRouteContext,
    _BoundLiquidMargin,
    _BoundOuterContinuousMargin,
    simulation_route,
)
from lcm import LinSpacedGrid
from lcm.consumption_savings_regime import (
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.solvers import NBEGM, NNBEGM, FiniteOuterGrid
from lcm.typing import BoolND, ContinuousAction, ContinuousState

_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=10)
_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)
_VARIABLES = Variables(
    info=MappingProxyType(
        {
            "liquid": VariableInfo(
                kind="state", topology="continuous", is_process=False
            ),
            "consumption": VariableInfo(
                kind="action", topology="continuous", is_process=False
            ),
        }
    )
)


def _route_context(*, phase: str = "solve") -> ConstraintRouteContext:
    """A minimal route context for a one-asset case-piece regime."""
    return ConstraintRouteContext(
        regime_name="alive",
        phase=phase,  # ty: ignore[invalid-argument-type]
        functions=MappingProxyType({}),
        variables=_VARIABLES,
        flat_param_names=frozenset({"crra"}),
        active_periods=(0, 1, 2),
    )


def _constraint_context(*, phase: str = "solve") -> ConstraintContext:
    """A minimal disposition context for the same regime."""
    return ConstraintContext(
        regime_name="alive",
        phase=phase,  # ty: ignore[invalid-argument-type]
        grids=MappingProxyType({"liquid": _SAVINGS_GRID}),
        function_names=frozenset({"resources", "savings", "utility"}),
        param_names=frozenset({"crra"}),
    )


def _bound_nbegm() -> NBEGM:
    """An NBEGM whose liquid margin is resolved, as the engine hands it over."""
    return NBEGM(savings_grid=_SAVINGS_GRID)._with_liquid_margin(
        _BoundLiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        )
    )


def _bound_nnbegm() -> NNBEGM:
    """An NNBEGM whose two margins are resolved, as the engine hands it over."""
    return NNBEGM(
        inner=NBEGM(savings_grid=_SAVINGS_GRID),
        outer_search=FiniteOuterGrid(grid=_SAVINGS_GRID),
    )._with_margins(
        liquid=_BoundLiquidMargin(
            state="liquid",
            action="consumption",
            resources="resources",
            post_decision_state="savings",
        ),
        outer=_BoundOuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="illiquid_post",
            no_adjustment=None,
        ),
    )


def _dispositions(constraints):
    """Plan a constraint pool against the case-piece route, keyed by name."""
    plan = plan_constraints(
        constraints=normalize_constraints(constraints=constraints),
        routes=case_piece_routes(
            context=_route_context(),
            post_decision_function="savings",
            solver_path=("nbegm",),
        ),
        context=_constraint_context(),
    )
    return {entry.constraint_name: entry.disposition for entry in plan.entries}


def _simulate_dispositions(constraints):
    """Plan a constraint pool against the simulate-phase route, keyed by name."""
    plan = plan_constraints(
        constraints=normalize_constraints(constraints=constraints),
        routes=case_piece_routes(
            context=_route_context(phase="simulate"),
            post_decision_function="savings",
            solver_path=("nbegm",),
        ),
        context=_constraint_context(phase="simulate"),
    )
    return {entry.constraint_name: entry.disposition for entry in plan.entries}


def rationing(consumption: ContinuousAction, liquid: ContinuousState) -> BoolND:
    """A feasibility predicate no savings-grid node locates."""
    return jnp.square(consumption) + jnp.square(liquid) <= 400.0


def always() -> BoolND:
    """A predicate that reads nothing at all."""
    return jnp.asarray(0.0) <= 0.0


def test_a_bound_the_savings_grid_enforces_is_proved_by_construction():
    """The borrowing limit the grid already imposes needs no evaluation."""
    got = _dispositions(
        {"borrowing_limit": post_decision_lower_bound(margin=_MARGIN, lower=0.0)}
    )

    assert isinstance(got["borrowing_limit"], ProvedByConstruction)


def test_the_proof_names_the_savings_grid_as_what_enforces_the_bound():
    """The discharged constraint carries a reason a diagnostic can quote."""
    got = _dispositions(
        {"borrowing_limit": post_decision_lower_bound(margin=_MARGIN, lower=0.0)}
    )

    assert "savings grid" in got["borrowing_limit"].proof.reason


def test_a_hand_written_bound_is_proved_exactly_as_the_constructor_is():
    """`ref("savings") >= 0.0` means what `post_decision_lower_bound` means.

    The proof keys on the comparison, so the two spellings cannot be admitted
    and refused respectively — which is what happens the moment a proof keys on
    a marker attribute only one of them carries.
    """
    got = _dispositions({"borrowing_limit": lcm.ref("savings") >= 0.0})

    assert isinstance(got["borrowing_limit"], ProvedByConstruction)


def test_a_bound_on_a_name_the_grid_does_not_span_is_not_proved():
    """Only a bound on the post-decision state the savings grid spans is proved.

    A lower bound on any other name says nothing about that grid, so nothing has
    discharged it and it falls through to be refused. Whether the *number* a
    bound names is the grid's own lowest node is a separate question, asked once
    when the model is built — see
    `test_nbegm_refuses_a_lower_bound_that_disagrees_with_its_savings_grid`.
    """
    got = _dispositions({"borrowing_limit": lcm.ref("wealth") >= 0.0})

    assert isinstance(got["borrowing_limit"], Reject)


def test_a_general_predicate_is_rejected():
    """No name is readable where the kernel would call a constraint."""
    got = _dispositions({"rationing": rationing})

    assert isinstance(got["rationing"], Reject)


def test_the_refusal_names_the_constraint_the_kernel_cannot_evaluate():
    """The diagnostic identifies which declaration was refused."""
    got = _dispositions({"rationing": rationing})

    assert "rationing" in got["rationing"].reason


def test_the_dependency_free_witness_really_reads_nothing():
    """`always` needs no name at the site, which is what gives the refusal meaning.

    A constraint that needs a name is refused for the ordinary reason — it is not
    in the allow-list — so it could not tell an empty allow-list read as a
    statement from one read as a subset test. Only a witness that needs nothing
    separates the two.

    Asserted over the names resolved through the site's own pool, which is what
    the planner consults. What the constraint spells is a different question, and
    the two agree only where the pool produces none of the names it spells.
    """
    site = case_piece_routes(
        context=_route_context(),
        post_decision_function="savings",
        solver_path=("nbegm",),
    )[0].sites[0]
    normalized = normalize_constraints(constraints={"always": always})

    assert (
        transitive_arg_names(constraint=normalized["always"], pool=site.function_pool)
        == frozenset()
    )


def test_a_constraint_reading_nothing_is_refused_like_any_other():
    """A predicate with no dependencies is not evaluated here either.

    A site that evaluates nothing says so with an empty allow-list, and that is
    a statement rather than a subset test that happens to fail. Read as a subset
    test it would admit exactly this constraint, whose dependencies are a subset
    of every set including the empty one — and the kernel would then be handed a
    predicate it calls nowhere.
    """
    got = _dispositions({"always": always})

    assert isinstance(got["always"], Reject)


@pytest.mark.parametrize(
    ("solver", "solver_path"),
    [(_bound_nbegm(), ("nbegm",)), (_bound_nnbegm(), ("nnbegm", "adjuster"))],
)
def test_the_case_piece_solvers_declare_an_empty_allow_list_not_the_default(
    solver, solver_path
):
    """Neither solver leaves its routes undeclared, and neither is permissive.

    An undeclared solver and a permissive one cannot be told apart by any
    verdict — both are the absence of a restriction — so the assertion is on the
    declaration. `None` routes would say the solver has not been written down;
    an unrestricted site would say the kernel reads every name where it
    evaluates a constraint, which is the opposite of true for it.
    """
    routes = solver.build_constraint_routes(context=_route_context())

    assert routes is not None
    keyed = {route.key.solver_path: route for route in routes}
    assert keyed[solver_path].sites[0].available_names == frozenset()


def test_nbegm_walks_one_solve_route_for_every_kernel_it_dispatches():
    """The five kernel variants share one route, because they differ in none.

    NBEGM compiles case pieces, a piecewise-affine schedule, a discrete
    envelope, their composition, or the ride-along kernels. What varies is the
    program; what a route carries — where a constraint can be met, over which
    names, through which pool — is the same for all five.
    """
    routes = _bound_nbegm().build_constraint_routes(context=_route_context())

    assert len(routes) == 1


def test_the_nested_solver_walks_one_route_per_rewritten_pool():
    """The adjuster and the keeper enter the inner solve through different pools.

    The adjuster's pool has the outer post-decision function removed, so the
    node the outer search picked arrives as a bound parameter; the keeper's has
    it replaced by the no-adjustment law. A site carries the pool it is entered
    with, so these are two routes rather than one described twice.
    """
    routes = _bound_nnbegm().build_constraint_routes(context=_route_context())

    assert {route.key.solver_path for route in routes} == {
        ("nnbegm", "adjuster"),
        ("nnbegm", "keeper"),
    }


def _route_shape(route: ConstraintRoute) -> tuple:
    """Every field of a route and of each of its sites, in declaration order.

    Built by reflection rather than by naming the fields, so a field added to
    either dataclass is compared without this having to be updated — which is
    the failure the comparison exists to catch.
    """
    return (
        route.key,
        tuple(
            tuple(_comparable(getattr(site, field.name)) for field in fields(site))
            for site in route.sites
        ),
    )


def _comparable(value: object) -> object:
    """A value that compares by what it is, not by identity.

    A proof is built fresh per call, so two routes carrying the same proof hold
    different objects. Comparing the callable's qualified name says they were
    built by the same factory, which is the claim.
    """
    if isinstance(value, tuple):
        return tuple(_comparable(entry) for entry in value)
    if callable(value):
        return getattr(value, "__qualname__", value)
    return value


@pytest.mark.parametrize(
    ("solver", "solver_path"),
    [(_bound_nbegm(), ("nbegm",)), (_bound_nnbegm(), ("nnbegm",))],
)
def test_the_simulate_route_is_the_shared_one_and_not_a_local_spelling(
    solver, solver_path
):
    """Simulation is the phase's pipeline, so both solvers take the shared route.

    Restating it here would agree with the shared declaration until one of them
    changed, and the disagreement would be a field nobody compared.
    """
    context = _route_context(phase="simulate")

    got = solver.build_constraint_routes(context=context)

    assert [_route_shape(route) for route in got] == [
        _route_shape(
            simulation_route(
                context=context,
                solver_path=solver_path,
                structural_proofs=(
                    proves_the_savings_grids_lower_bound(post_decision="savings"),
                ),
            )
        )
    ]


def test_the_bound_is_proved_on_the_simulate_route_too():
    """Simulation enforces the same limit through the mask built from that node.

    This is the phase where the proof fires for these solvers: the solve route
    evaluates nothing and every accepted model reaches it with no constraint,
    while the simulate route carries the injected budget predicate and whatever
    else the regime declares. Attaching the proof and firing it are different
    claims, and only the second one keeps the bound from being evaluated a
    second time against a mask that already imposes it.
    """
    got = _simulate_dispositions(
        {"borrowing_limit": post_decision_lower_bound(margin=_MARGIN, lower=0.0)}
    )

    assert isinstance(got["borrowing_limit"], ProvedByConstruction)


def test_the_simulate_route_still_evaluates_what_it_can_read():
    """The unrestricted simulate site is not a second refusal in disguise.

    Without this the test above would pass on a route that discharged the bound
    and refused everything else, which is the solve route's behaviour rather
    than simulation's.
    """
    got = _simulate_dispositions({"rationing": rationing})

    assert isinstance(got["rationing"], Evaluate)
