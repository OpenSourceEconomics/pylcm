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

from _lcm.constraints.dispositions import ConstraintContext, Evaluate, Reject
from _lcm.constraints.processed import normalize_constraints
from _lcm.constraints.routes import plan_constraints
from _lcm.engine import VariableInfo, Variables
from _lcm.solution.contract import ConstraintRouteContext, SolutionKernels
from lcm import ref
from lcm.solvers import DCEGM, EGM, GridSearch
from tests.test_models import dcegm_paper_twin

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


def _dcegm() -> DCEGM:
    """The bound DC-EGM of an ordinary consumption-savings regime."""
    regime = dcegm_paper_twin._working_life(solver="dcegm")
    return regime.solver  # ty: ignore[invalid-return-type]


def _disposition(solver, *, constraint, phase="solve"):
    """Plan one constraint over `solver`'s routes for one phase."""
    routes = solver.build_constraint_routes(context=_route_context(phase=phase))
    assert routes is not None
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
        EGM(savings_grid=dcegm_paper_twin.SAVINGS_GRID),
        constraint=ref("consumption") <= 5.0,
    )

    assert isinstance(disposition, Reject)


def test_plain_egm_evaluates_the_simulate_phase_feasibility_check() -> None:
    """Its simulate phase has whole candidates, including a synthesized budget."""
    disposition = _disposition(
        EGM(savings_grid=dcegm_paper_twin.SAVINGS_GRID),
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
        EGM(savings_grid=dcegm_paper_twin.SAVINGS_GRID),
        _dcegm(),
    )

    sites = [
        solver.build_constraint_routes(context=_route_context(phase="simulate"))[
            0
        ].sites
        for solver in solvers
    ]

    assert [(site.stage, site.available_names) for (site,) in sites] == [
        ("simulation", None)
    ] * 3
