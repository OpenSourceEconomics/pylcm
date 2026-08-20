"""Each shipped solver declares the routes it actually walks.

A route is not a description of a solver family, it is a description of one
solver's candidate production in one phase. Grid search searches the whole
state-action product, so every name is in scope wherever it evaluates. Plain
EGM evaluates no user constraint at all. DC-EGM evaluates per discrete
combination, before the continuous inner stage, so a constraint reading the
continuous action cannot be met there.

A solver that has not declared its routes says so by declaring nothing, and
nothing is planned for it — which is what keeps the attributed refusal a custom
solver already gets from turning into a generic failure.
"""

from types import MappingProxyType
from typing import cast

from _lcm.constraints.dispositions import (
    ConstraintContext,
    Evaluate,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.processed import normalize_constraints
from _lcm.constraints.routes import ConstraintRoute, plan_constraints
from _lcm.engine import VariableInfo, Variables
from _lcm.solution.contract import ConstraintRouteContext, SolutionKernels
from _lcm.typing import EconFunctionsMapping
from lcm import ref
from lcm.solvers import DCEGM, EGM, GridSearch, Solver
from lcm.typing import BoolND, ContinuousState
from tests.test_models import dcegm_paper_twin, negm_kinked_toy

_VARIABLES = Variables(
    info=MappingProxyType(
        {
            "work_choice": VariableInfo(
                kind="state", topology="discrete", is_process=False
            ),
            "wealth": VariableInfo(
                kind="state", topology="continuous", is_process=False
            ),
            "housing": VariableInfo(
                kind="state", topology="continuous", is_process=False
            ),
            "retire": VariableInfo(
                kind="action", topology="discrete", is_process=False
            ),
            "consumption": VariableInfo(
                kind="action", topology="continuous", is_process=False
            ),
        }
    )
)

_NEGM_VARIABLES = Variables(
    info=MappingProxyType(
        {
            "wealth": VariableInfo(
                kind="state", topology="continuous", is_process=False
            ),
            "illiquid": VariableInfo(
                kind="state", topology="continuous", is_process=False
            ),
            "consumption": VariableInfo(
                kind="action", topology="continuous", is_process=False
            ),
            "illiquid_investment": VariableInfo(
                kind="action", topology="continuous", is_process=False
            ),
        }
    )
)
_NEGM_REGIME = negm_kinked_toy.build_alive_regime()


def _route_context(*, phase: str = "solve") -> ConstraintRouteContext:
    return ConstraintRouteContext(
        regime_name="working",
        phase=phase,  # ty: ignore[invalid-argument-type]
        functions=MappingProxyType({}),
        variables=_VARIABLES,
        flat_param_names=frozenset({"interest_rate"}),
        active_periods=(0, 1, 2),
    )


def _constraint_context(*, phase: str = "solve") -> ConstraintContext:
    return ConstraintContext(
        regime_name="working",
        phase=phase,  # ty: ignore[invalid-argument-type]
        grids=MappingProxyType({}),
        function_names=frozenset(),
        param_names=frozenset({"interest_rate"}),
    )


def _negm_route_context(*, phase: str) -> ConstraintRouteContext:
    return ConstraintRouteContext(
        regime_name="alive",
        phase=phase,  # ty: ignore[invalid-argument-type]
        functions=cast(
            "EconFunctionsMapping",
            MappingProxyType(dict(_NEGM_REGIME.functions)),
        ),
        variables=_NEGM_VARIABLES,
        flat_param_names=frozenset(),
        active_periods=(0, 1, 2),
    )


def _negm_constraint_context(*, phase: str) -> ConstraintContext:
    return ConstraintContext(
        regime_name="alive",
        phase=phase,  # ty: ignore[invalid-argument-type]
        grids=MappingProxyType({}),
        function_names=frozenset(_NEGM_REGIME.functions),
        param_names=frozenset(),
    )


def housing_stays_in_bounds(new_durable: ContinuousState) -> BoolND:
    """The outer candidate stays on the durable grid's represented domain."""
    return (new_durable >= 0.0) & (new_durable <= 30.0)


def _bound(solver: Solver) -> Solver:
    """Bind `solver` onto an ordinary consumption-savings regime.

    A solver reads its own margin roles to declare its routes — which state the
    inversion produces, which post-decision state its grid spans — and those
    exist only once a regime has bound them. The engine always calls
    `build_constraint_routes` on the bound object, the same one it hands to
    `build_period_kernels`.
    """
    regime = dcegm_paper_twin._working_life(solver="dcegm")
    return regime.replace(solver=solver).solver


def _dcegm() -> Solver:
    """The bound DC-EGM of an ordinary consumption-savings regime."""
    return _bound(DCEGM(savings_grid=dcegm_paper_twin.SAVINGS_GRID))


def _egm() -> Solver:
    """The bound plain EGM of the same regime."""
    return _bound(EGM(savings_grid=dcegm_paper_twin.SAVINGS_GRID))


def _routes_of(solver: Solver, *, phase: str = "solve") -> tuple[ConstraintRoute, ...]:
    """The routes `solver` declares for one phase, asserted to be declared."""
    routes = solver.build_constraint_routes(context=_route_context(phase=phase))
    assert routes is not None
    return routes


def _disposition(solver, *, constraint, phase="solve"):
    """Plan one constraint over `solver`'s routes for one phase."""
    routes = _routes_of(solver, phase=phase)
    plan = plan_constraints(
        constraints=normalize_constraints(constraints={"c": constraint}),
        routes=routes,
        context=_constraint_context(phase=phase),
    )
    return plan.entries[0].disposition


def test_grid_search_evaluates_over_the_whole_state_action_product() -> None:
    """Every name is in scope where grid search evaluates, so nothing is refused."""
    disposition = _disposition(GridSearch(), constraint=ref("consumption") <= 5.0)

    assert isinstance(disposition, Evaluate)


def test_grid_search_evaluates_at_the_state_action_stage_when_solving() -> None:
    """Grid search has whole candidates in hand, not a discrete combination."""
    disposition = _disposition(GridSearch(), constraint=ref("consumption") <= 5.0)

    assert disposition.stage == "state_action"


def test_grid_search_evaluates_at_the_simulation_stage_when_simulating() -> None:
    """The simulate phase checks feasibility against the subject's own candidate."""
    disposition = _disposition(
        GridSearch(), constraint=ref("consumption") <= 5.0, phase="simulate"
    )

    assert disposition.stage == "simulation"


def test_plain_egm_evaluates_no_user_constraint_when_solving() -> None:
    """The envelope-free kernel calls no predicate, so it refuses rather than drops."""
    disposition = _disposition(
        _egm(),
        constraint=ref("consumption") <= 5.0,
    )

    assert isinstance(disposition, Reject)


def test_plain_egm_evaluates_the_simulate_phase_feasibility_check() -> None:
    """Its simulate phase has whole candidates, including a synthesized budget."""
    disposition = _disposition(
        _egm(),
        constraint=ref("consumption") <= 5.0,
        phase="simulate",
    )

    assert isinstance(disposition, Evaluate)


def test_dcegm_evaluates_a_constraint_over_its_discrete_combination() -> None:
    """A discrete action is bound per combination, so a constraint on it is callable."""
    disposition = _disposition(_dcegm(), constraint=ref("retire") <= 1)

    assert isinstance(disposition, Evaluate)


def test_dcegm_evaluates_at_the_discrete_combo_stage() -> None:
    """DC-EGM's feasibility predicate is built per discrete combination."""
    disposition = _disposition(_dcegm(), constraint=ref("retire") <= 1)

    assert disposition.stage == "discrete_combo"


def test_dcegm_refuses_a_constraint_reading_its_continuous_action() -> None:
    """The consumption the Euler inversion produces is not bound per combination."""
    disposition = _disposition(_dcegm(), constraint=ref("consumption") <= 5.0)

    assert isinstance(disposition, Reject)


def test_dcegm_evaluates_a_constraint_over_a_passive_continuous_state() -> None:
    """Every continuous state other than the Euler state is bound per combination."""
    disposition = _disposition(_dcegm(), constraint=ref("housing") >= 0.0)

    assert isinstance(disposition, Evaluate)


def test_dcegm_refuses_a_constraint_reading_its_euler_state() -> None:
    """The Euler state is the axis the inversion produces, not a combination input."""
    disposition = _disposition(_dcegm(), constraint=ref("wealth") >= 0.0)

    assert isinstance(disposition, Reject)


def test_dcegm_evaluates_a_constraint_over_a_regime_parameter() -> None:
    """A param is a constant wherever the kernel evaluates."""
    disposition = _disposition(_dcegm(), constraint=ref("interest_rate") >= 0.0)

    assert isinstance(disposition, Evaluate)


def test_dcegm_evaluates_the_simulate_phase_feasibility_check() -> None:
    """Simulation holds the realized action, so the continuous one is readable."""
    disposition = _disposition(
        _dcegm(), constraint=ref("consumption") <= 5.0, phase="simulate"
    )

    assert isinstance(disposition, Evaluate)


def test_a_solver_that_does_not_rebuild_per_age_declares_one_route() -> None:
    """A route that is the same in every period says so, rather than repeating.

    One route per period group from a solver that resolves its pool alike at
    every age would put an entry per group in the plan where there is a single
    fact, and a coverage count over those pairs would read a constant as
    evidence of grouping.
    """
    routes = GridSearch().build_constraint_routes(context=_route_context())

    assert [route.key.period_group for route in routes] == [None]


def test_a_solver_that_has_not_declared_its_routes_declares_nothing() -> None:
    """Undeclared is not the same as declaring an unrestricted route.

    A solver whose routes nobody has written down must not be handed a
    permissive default, because that would claim the opposite of the truth for
    any solver that in fact evaluates nothing.
    """
    from _lcm.solution.contract import Solver, SolverBuildContext  # noqa: PLC0415

    class _Undeclared(Solver):
        def build_period_kernels(
            self,
            *,
            context: SolverBuildContext,
        ) -> SolutionKernels:
            raise NotImplementedError

    assert _Undeclared().build_constraint_routes(context=_route_context()) is None


def test_every_solver_declares_the_same_simulate_route() -> None:
    """Simulation is the phase's pipeline, not the solver's.

    It walks the regime's DAG on realized states and the realized action, so a
    whole candidate is in hand whatever the solver did when solving. Six
    separate spellings of that one fact would agree by convention until one
    did not, and the disagreement would be a field nobody compared.
    """
    solvers = (
        GridSearch(),
        _egm(),
        _dcegm(),
    )

    sites = [_routes_of(solver, phase="simulate")[0].sites for solver in solvers]

    assert [(site.stage, site.available_names) for (site,) in sites] == [
        ("simulation", None)
    ] * 3


def test_negm_evaluates_housing_bounds_on_all_three_candidate_routes() -> None:
    """The adjuster, keeper, and simulation each honour the outer-stock bound."""
    solver = _NEGM_REGIME.solver
    normalized = normalize_constraints(
        constraints={"housing_stays_in_bounds": housing_stays_in_bounds}
    )
    observed = {}
    for phase in ("solve", "simulate"):
        routes = solver.build_constraint_routes(
            context=_negm_route_context(phase=phase)
        )
        assert routes is not None
        plan = plan_constraints(
            constraints=normalized,
            routes=routes,
            context=_negm_constraint_context(phase=phase),
        )
        for entry in plan.entries:
            assert isinstance(entry.disposition, Evaluate)
            observed[(entry.route.phase, entry.route.solver_path)] = (
                entry.route.period_group,
                entry.disposition.stage,
            )

    assert observed == {
        ("solve", ("negm", "adjuster")): ((0, 1, 2), "outer_candidate"),
        ("solve", ("negm", "keeper")): ((0, 1, 2), "keeper_candidate"),
        ("simulate", ("negm",)): (None, "simulation"),
    }


def test_dcegm_proves_a_bound_on_the_state_its_savings_grid_spans() -> None:
    """The grid the solve inverts on is what enforces the limit, so it is not called.

    Its lowest node *is* the limit, and the simulate phase enforces the same
    number through the mask built from it. The declaration is still a claim —
    checked against the grid when the model is built — which is what makes
    proving it different from ignoring it.
    """
    disposition = _disposition(_dcegm(), constraint=ref("savings") >= 0.0)

    assert isinstance(disposition, ProvedByConstruction)


def test_the_proof_names_the_savings_grid_as_what_enforces_the_bound() -> None:
    """A diagnostic has to be able to quote what discharged a constraint."""
    disposition = _disposition(_dcegm(), constraint=ref("savings") >= 0.0)

    assert "savings grid" in disposition.proof.reason


def test_a_bound_on_another_name_is_not_proved_by_the_savings_grid() -> None:
    """A lower bound elsewhere says nothing about the grid, so nothing proved it.

    Discharging it on the strength of its shape alone would silently drop a
    constraint the model relies on.
    """
    disposition = _disposition(_dcegm(), constraint=ref("wealth") >= 0.0)

    assert isinstance(disposition, Reject)


def test_the_same_bound_is_proved_on_the_simulate_route_too() -> None:
    """The simulate mask is synthesized from the same lowest node."""
    disposition = _disposition(
        _dcegm(), constraint=ref("savings") >= 0.0, phase="simulate"
    )

    assert isinstance(disposition, ProvedByConstruction)


def test_plain_egm_proves_the_bound_its_savings_grid_enforces() -> None:
    """The envelope-free kernel evaluates nothing, but its grid still enforces this."""
    disposition = _disposition(
        _egm(),
        constraint=ref("savings") >= 0.0,
    )

    assert isinstance(disposition, ProvedByConstruction)
