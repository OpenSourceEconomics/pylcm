import functools
from collections import Counter
from collections.abc import Callable
from dataclasses import replace as dataclasses_replace
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import pytest
from beartype import beartype
from numpy.testing import assert_array_equal

from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import build_template_egm_carry
from _lcm.engine import Regime, VariableInfo, Variables
from _lcm.grids import DiscreteGrid, Grid, LinSpacedGrid
from _lcm.regime_building.canonicalize import _canonicalize_phase_transitions
from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.regime_building.processing import (
    _rename_params_to_qnames,
    _wrap_regime_transition_probs,
    process_regimes,
)
from _lcm.solution.contract import SolutionKernels, SolverBuildContext
from _lcm.variables import from_regime, get_grids
from lcm import (
    LinearAggregator,
    LinearExpectation,
    Phased,
    categorical,
)
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.model import Model
from lcm.regime import Regime as UserRegime
from lcm.solver_api import EGM_CONTINUATION
from lcm.solvers import DCEGM, EGM, NBEGM, NEGM, GridSearch, Solver
from lcm.typing import FloatND, ScalarInt
from tests.conftest import build_prepared_structure
from tests.mock_regime import MockRegime
from tests.solution.test_egm_solver import _SAVINGS_GRID as EGM_SAVINGS_GRID
from tests.solution.test_egm_solver import _model as egm_model
from tests.test_models import negm_kinked_toy
from tests.test_models.dcegm_paper_twin import build_dcegm_model
from tests.test_models.deterministic.base import dead, working_life
from tests.test_nbegm_constraint_validation import _build_smooth_model


def test_variables_from_regime_tags_kind_and_topology(binary_category_class):
    def utility(c):
        pass

    def next_c(*, a, b):
        pass

    mock_regime = MockRegime(
        actions={
            "a": DiscreteGrid(category_class=binary_category_class),
        },
        states={
            "c": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"c": next_c},
        functions={"utility": utility},
    )

    got = from_regime(mock_regime)

    assert isinstance(got, Variables)
    assert set(got) == {"a", "c"}
    assert got["a"] == VariableInfo(
        kind="action", topology="discrete", is_process=False
    )
    assert got["c"] == VariableInfo(kind="state", topology="discrete", is_process=False)


def test_get_grids(binary_category_class):
    def next_c(*, a, b):
        pass

    mock_regime = MockRegime(
        actions={
            "a": DiscreteGrid(category_class=binary_category_class),
        },
        states={
            "c": DiscreteGrid(category_class=binary_category_class),
        },
        state_transitions={"c": next_c},
        functions={"utility": lambda _c: None},
    )

    got = get_grids(mock_regime)
    assert isinstance(got["a"], DiscreteGrid)
    assert got["a"].categories == ("cat0", "cat1")
    assert got["a"].codes == (0, 1)

    assert isinstance(got["c"], DiscreteGrid)
    assert got["c"].categories == ("cat0", "cat1")
    assert got["c"].codes == (0, 1)


def test_get_grids_reorder(binary_category_class):
    def next_state(*, a, b):
        pass

    mock_regime = MockRegime(
        actions={
            "a": DiscreteGrid(category_class=binary_category_class),
        },
        states={
            "b": DiscreteGrid(category_class=binary_category_class),
            "c": DiscreteGrid(category_class=binary_category_class, batch_size=1),
            "d": LinSpacedGrid(start=0, stop=1, n_points=5, batch_size=3),
            "e": LinSpacedGrid(start=0, stop=1, n_points=5, batch_size=1),
            "f": LinSpacedGrid(start=0, stop=1, n_points=5),
        },
        state_transitions={
            "b": next_state,
            "c": next_state,
            "d": next_state,
            "e": next_state,
            "f": next_state,
        },
        functions={"utility": lambda _c: None},
    )

    got = get_grids(mock_regime)
    assert list(got.keys()) == ["c", "b", "e", "d", "f", "a"]


def test_process_regimes():
    ages = AgeGrid(start=0, stop=4, step="Y")
    user_regimes = {"working_life": working_life, "dead": dead}
    regime_names_to_ids = MappingProxyType(
        {name: jnp.int32(idx) for idx, name in enumerate(user_regimes.keys())}
    )
    finalized_user_regimes = finalize_regimes(
        user_regimes=user_regimes,
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    regimes = process_regimes(
        user_regimes=finalized_user_regimes,
        ages=ages,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=True,
        prepared_structure=build_prepared_structure(
            user_regimes=finalized_user_regimes, ages=ages
        ),
    )
    working_regime = regimes["working_life"]

    # Variable Info
    variables = working_regime.solution._variables
    assert variables["wealth"] == VariableInfo(
        kind="state", topology="continuous", is_process=False
    )
    assert variables["labor_supply"] == VariableInfo(
        kind="action", topology="discrete", is_process=False
    )
    assert variables["consumption"] == VariableInfo(
        kind="action", topology="continuous", is_process=False
    )

    # Grids — compare the grid objects (which now include transition attributes)
    assert working_regime.solution.grids["wealth"] == working_life.states["wealth"]
    assert (
        working_regime.solution.grids["consumption"]
        == working_life.actions["consumption"]
    )

    assert isinstance(working_regime.solution.grids["labor_supply"], DiscreteGrid)
    assert working_regime.solution.grids["labor_supply"].categories == (
        "work",
        "retire",
    )
    assert working_regime.solution.grids["labor_supply"].codes == (0, 1)

    # Materialized grids
    assert_array_equal(
        working_regime.solution.grids["consumption"].to_jax(),
        working_life.actions["consumption"].to_jax(),
    )
    assert_array_equal(
        working_regime.solution.grids["wealth"].to_jax(),
        cast("Grid", working_life.states["wealth"]).to_jax(),
    )

    assert (
        working_regime.solution.grids["labor_supply"].to_jax() == jnp.array([0, 1])
    ).all()

    # Functions
    assert working_regime.solution.transitions is not None
    assert working_regime.solution.constraints is not None
    assert "utility" in working_regime.solution.functions


def test_variables_excludes_constraint_names():
    """Constraint functions do not appear as variables in the Variables container."""

    def wealth_constraint(wealth):
        return wealth > 200

    working_copy = working_life.replace(
        constraints=dict(working_life.constraints)
        | {"wealth_constraint": wealth_constraint}
    )

    got = from_regime(working_copy)
    assert "wealth_constraint" not in got


@pytest.fixture(name="two_non_terminal_regimes")
def _two_non_terminal_regimes() -> MappingProxyType[str, Regime]:
    """Two non-terminal regimes that share underlying user functions."""

    def next_x(x):
        return x

    def regime_transition(*, age, final_age):
        return jnp.where(age >= final_age, 1, 0)

    @categorical(ordered=False)
    class TwoRegimeId:
        early: ScalarInt
        late: ScalarInt

    early = UserRegime(
        transition=regime_transition,
        states={"x": LinSpacedGrid(start=0, stop=10, n_points=4)},
        state_transitions={"x": next_x},
        functions={"utility": lambda x: x},
        active=lambda age: age < 1,
    )
    late = UserRegime(
        transition=regime_transition,
        states={"x": LinSpacedGrid(start=0, stop=10, n_points=6)},
        state_transitions={"x": next_x},
        functions={"utility": lambda x: x},
        active=lambda age: age >= 1,
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    finalized_user_regimes = finalize_regimes(
        user_regimes={"early": early, "late": late},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )
    return process_regimes(
        user_regimes=finalized_user_regimes,
        ages=ages,
        regime_names_to_ids=MappingProxyType(
            {"early": jnp.int32(0), "late": jnp.int32(1)}
        ),
        enable_jit=True,
        prepared_structure=build_prepared_structure(
            user_regimes=finalized_user_regimes, ages=ages
        ),
    )


@pytest.mark.parametrize(
    "attr",
    ["next_state", "compute_regime_transition_probs"],
)
def test_simulate_functions_use_per_regime_callables(
    *,
    two_non_terminal_regimes: MappingProxyType[str, Regime],
    attr: str,
) -> None:
    """Two regimes built from shared user functions get distinct simulate callables."""
    early_func = getattr(two_non_terminal_regimes["early"].simulation, attr)
    late_func = getattr(two_non_terminal_regimes["late"].simulation, attr)
    assert id(early_func) != id(late_func)


def test_rename_params_to_qnames_with_partial():
    """Regression: dags >=0.5.1 renames bound partial keywords to qualified names."""

    def utility(*, consumption, risk_aversion):
        return consumption ** (1 - risk_aversion)

    func = functools.partial(utility, risk_aversion=2.0)
    regime_params_template = MappingProxyType(
        {"utility": MappingProxyType({"risk_aversion": "float"})}
    )

    result = _rename_params_to_qnames(
        func=func,
        regime_params_template=regime_params_template,
        param_key="utility",
    )
    # 1. The bound default must work under the qualified name. Before dags >=0.5.1,
    #    the manual unwrap/re-wrap rebound keywords under the old name, so this call
    #    would raise TypeError.
    assert result(consumption=5.0) == 5.0 ** (1 - 2.0)

    # 2. The qualified name must be usable to override the default. This fails if
    #    _rename_params_to_qnames is a no-op (no renaming happened).
    assert result(consumption=5.0, utility__risk_aversion=3.0) == 5.0 ** (1 - 3.0)


def test_wrap_regime_transition_probs_return_annotation_accepts_mapping():
    """The regime-transition-probs wrapper returns a regime-name → probability
    mapping, so its return annotation describes that mapping.

    The wrapper turns a `next_regime` function's probability array into a
    `MappingProxyType` keyed by regime name. Its return annotation must match
    that mapping rather than the array type carried by `next_regime`, so a
    beartype check on the wrapper accepts the value it genuinely returns.
    """

    def next_regime() -> FloatND:
        return jnp.array([0.3, 0.7])

    regime_names_to_ids = MappingProxyType(
        {"working": jnp.int32(0), "retired": jnp.int32(1)}
    )
    wrapped = _wrap_regime_transition_probs(
        func=next_regime,  # ty: ignore[invalid-argument-type]
        regime_names_to_ids=regime_names_to_ids,
    )

    result = beartype(wrapped)()

    assert set(result) == {"working", "retired"}


def _pair_handover_regime() -> UserRegime:
    """A regime whose only handover to `retired` is its carried state."""

    def impute_pension_wealth(wealth: float) -> float:
        return wealth * 0.1

    def evolve_pension_wealth(pension_wealth: float) -> float:
        return pension_wealth * 1.03

    def next_wealth(wealth: float) -> float:
        return wealth

    def next_regime(_age: float) -> ScalarInt:
        return jnp.int32(0)

    def utility(wealth: float) -> FloatND:
        return jnp.asarray(wealth)

    return UserRegime(
        transition=next_regime,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3),
            "pension_wealth": Phased(
                solve=impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=5.0, n_points=2),
            ),
        },
        state_transitions={
            "wealth": next_wealth,
            "pension_wealth": evolve_pension_wealth,
        },
        actions={},
        functions={"utility": utility},
    )


def test_carried_law_registered_for_carried_only_target():
    """A target regime sharing only the carried state still receives `next_<name>`.

    A regime can hand over nothing but its carried state (retirement keeps
    pension wealth, drops the working states). The carried law of motion must
    be registered for that target in the simulate phase — otherwise the
    simulation silently freezes the carried value on the crossing.
    """
    working = _pair_handover_regime()
    simulate_states_per_regime = {
        "working": frozenset({"wealth", "pension_wealth"}),
        "retired": frozenset({"pension_wealth"}),
        "dead": frozenset(),
    }
    canonical, _ = _canonicalize_phase_transitions(
        phase_slice=normalize_regime_phases(working).simulation,
        states_per_regime=simulate_states_per_regime,
    )
    assert "retired" in canonical["pension_wealth"]


def test_carried_state_counts_as_covered_for_reachability():
    """A target carrying a carried state stays reachable when per-target
    transitions exist.

    With per-target transitions present, a target not explicitly named in any
    per-target dict is reachable when simple transitions cover its state
    needs. In the simulate phase the carried state's law of motion is an
    ordinary simple transition, so the carried state counts as covered and
    the target receives both the ordinary hand-over and the carried law.
    """

    def impute_pension_wealth(wealth: float) -> float:
        return wealth * 0.1

    def evolve_pension_wealth(pension_wealth: float) -> float:
        return pension_wealth * 1.03

    def next_wealth(wealth: float) -> float:
        return wealth

    def next_health_working(health: float) -> float:
        return health

    def next_regime(_age: float) -> ScalarInt:
        return jnp.int32(0)

    def utility(wealth: float) -> FloatND:
        return jnp.asarray(wealth)

    working = UserRegime(
        transition=next_regime,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3),
            "health": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
            "pension_wealth": Phased(
                solve=impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=5.0, n_points=2),
            ),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": {"working": next_health_working},
            "pension_wealth": evolve_pension_wealth,
        },
        actions={},
        functions={"utility": utility},
    )
    # `retired` is not named in any per-target dict; its ordinary state need
    # (wealth) is covered by a bare law and the carried law covers the
    # carried state, so it must be reachable and receive both laws.
    simulate_states_per_regime = {
        "working": frozenset({"wealth", "health", "pension_wealth"}),
        "retired": frozenset({"wealth", "pension_wealth"}),
        "dead": frozenset(),
    }
    canonical, _ = _canonicalize_phase_transitions(
        phase_slice=normalize_regime_phases(working).simulation,
        states_per_regime=simulate_states_per_regime,
    )
    assert "retired" in canonical["wealth"]
    assert "retired" in canonical["pension_wealth"]


def test_mock_regime_get_all_functions_matches_real_regime():
    """`MockRegime.get_all_functions` exposes the same keys as the real method.

    The mock is the test double for canonical processing; a key set that
    drifts from `lcm.regime.Regime.get_all_functions` (e.g. dropping a carried
    state's `next_<name>` law) would let mock-based tests pass against
    behavior the real regime does not have.
    """

    def impute_pension_wealth(wealth: float) -> float:
        return wealth * 0.1

    def evolve_pension_wealth(pension_wealth: float) -> float:
        return pension_wealth * 1.03

    def next_wealth(wealth: float) -> float:
        return wealth

    def next_regime(_age: float) -> ScalarInt:
        return jnp.int32(0)

    def utility(wealth: float) -> FloatND:
        return jnp.asarray(wealth)

    kwargs: dict = {
        "transition": next_regime,
        "states": {
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3),
            "pension_wealth": Phased(
                solve=impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=5.0, n_points=2),
            ),
        },
        "state_transitions": {
            "wealth": next_wealth,
            "pension_wealth": evolve_pension_wealth,
        },
        "functions": {"utility": utility},
    }
    real = finalize_regimes(
        user_regimes={"regime": UserRegime(**kwargs)},
        derived_categoricals={},
        koopmans_aggregator=LinearAggregator(),
        certainty_equivalent=LinearExpectation(),
    )["regime"]
    mock = MockRegime(**kwargs)
    assert set(mock.get_all_functions()) == set(real.get_all_functions())


def _egm_model() -> Model:
    """A one-liquid consumption-saving model solved by plain EGM."""
    return egm_model(solver=EGM(savings_grid=EGM_SAVINGS_GRID))


def _nbegm_model() -> Model:
    """An unconstrained case-piece model solved by NB-EGM on a dense route."""
    return _build_smooth_model(constraints={})


@pytest.mark.parametrize(
    ("build_model", "solver_class", "expected"),
    [
        (_egm_model, EGM, {"saving": 1}),
        (_nbegm_model, NBEGM, {"alive": 1}),
        (build_dcegm_model, DCEGM, {"working_life": 1, "retirement": 1}),
        (negm_kinked_toy.build_model, NEGM, {"alive": 1}),
    ],
    ids=["egm", "nbegm", "dcegm", "negm"],
)
def test_a_continuation_source_model_builds_each_regimes_kernels_once(
    *,
    monkeypatch: pytest.MonkeyPatch,
    build_model: Callable[[], Model],
    solver_class: type[Solver],
    expected: dict[str, int],
) -> None:
    """Declaring the leaves a solver reads costs no second kernel build.

    A solver's `build_period_kernels` runs the model author's build-time
    consumers — a compiled constraint boundary, a boundary plan — so calling it
    twice for one regime consumes each of them twice.
    """
    built: list[str] = []
    build_period_kernels = solver_class.build_period_kernels

    # keyword-only-exempt: library-callback=types.FunctionType.__get__
    def counting_build_period_kernels(
        self: Solver, *, context: SolverBuildContext
    ) -> SolutionKernels:
        """Record the regime the kernels are built for, then build them."""
        built.append(context.regime_name)
        return build_period_kernels(self, context=context)

    monkeypatch.setattr(
        solver_class, "build_period_kernels", counting_build_period_kernels
    )

    build_model()

    assert Counter(built) == expected


# keyword-only-exempt: library-callback=types.FunctionType.__get__
def _declare_a_different_continuation_spec(
    self: Solver, *, kernels: SolutionKernels, context: SolverBuildContext
) -> SolutionKernels:
    """Return kernels publishing a continuation the engine never handed over."""
    del self, context
    return dataclasses_replace(
        kernels,
        continuation_spec=EGMContinuationSpec(
            template=build_template_egm_carry(n_rows=4)
        ),
    )


def test_a_declaration_that_changes_more_than_the_kernels_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the returned period kernels are honoured, so nothing else may move."""
    monkeypatch.setattr(
        DCEGM, "declare_continuation_reads", _declare_a_different_continuation_spec
    )

    with pytest.raises(
        RegimeInitializationError, match="'continuation_spec'"
    ) as excinfo:
        build_dcegm_model()

    assert "working_life" in str(excinfo.value)


# keyword-only-exempt: library-callback=types.FunctionType.__get__
def _declare_unchanged_kernels(
    self: Solver, *, kernels: SolutionKernels, context: SolverBuildContext
) -> SolutionKernels:
    """Return an equal container, declaring nothing, as a real hook would."""
    del self, context
    return dataclasses_replace(kernels)


def test_a_declaring_solver_whose_kernels_are_engine_decorated_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine-produced carry adapter would swallow a solver's declarations.

    A regime that publishes no continuation of its own is handed one the engine
    produces, wrapped around its adapters. A solver declaring reads on such a
    regime would attach them to programs the wrapper never publishes, so the
    combination is named at build instead.
    """
    monkeypatch.setattr(
        GridSearch,
        "required_continuation_keys",
        property(lambda _self: frozenset({EGM_CONTINUATION})),
    )
    monkeypatch.setattr(
        GridSearch, "declare_continuation_reads", _declare_unchanged_kernels
    )

    with pytest.raises(RegimeInitializationError, match="engine-produced"):
        build_dcegm_model()
