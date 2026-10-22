"""Tests for runtime-supplied grid points (IrregSpacedGrid)."""

import warnings
from collections.abc import Mapping

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.grids import Grid, IrregSpacedGrid
from lcm import (
    AgeGrid,
    DeterministicTransition,
    LinSpacedGrid,
    Model,
    Phased,
    Transition,
    categorical,
)
from lcm.exceptions import InvalidParamsError
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    FunctionName,
    ScalarInt,
    UserFunction,
    UserParams,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility(*, consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return jnp.log(consumption + 1) + 0.01 * wealth


def _next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption)


def _borrowing_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return consumption <= wealth


def _next_regime(period: int) -> FloatND:
    return jnp.where(period >= 1, RegimeId.dead, RegimeId.alive)


_EDGES = {
    "alive": Transition(
        targets={"alive": 0, "dead": (0, 1)},
        law=DeterministicTransition(func=_next_regime),
    )
}


def _make_model(
    *,
    wealth_grid: Grid | Phased | None = None,
    utility: UserFunction | Phased = _utility,
    functions: Mapping[FunctionName, UserFunction] | None = None,
    constraints: Mapping[FunctionName, UserFunction] | None = None,
    fixed_params: UserParams | None = None,
) -> Model:
    """Create a simple 2-regime model for testing."""
    if wealth_grid is None:
        wealth_grid = LinSpacedGrid(start=1, stop=10, n_points=5)

    alive = UserRegime(
        functions={"utility": utility, **(functions or {})},
        states={"wealth": wealth_grid},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5, n_points=5)},
        constraints=(
            constraints
            if constraints is not None
            else {"borrowing_constraint": _borrowing_constraint}
        ),
    )
    dead = UserRegime(
        functions={"utility": lambda: 0.0},
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges=_EDGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
        fixed_params=fixed_params or {},
    )


def test_runtime_grid_creation():
    """IrregSpacedGrid without points, only n_points."""
    grid = IrregSpacedGrid(n_points=5)
    assert grid.pass_points_at_runtime
    assert grid.n_points == 5


def test_runtime_grid_to_jax_raises_before_substitution():
    """`to_jax()` on a runtime grid is a bug — substitution happens at
    solve/simulate time via `regime.solution.state_action_space(regime_params=...)`.
    The error message must point the caller at that API."""
    grid = IrregSpacedGrid(n_points=5)
    with pytest.raises(Exception, match="state_action_space"):
        grid.to_jax()


def test_fixed_grid_not_runtime():
    """IrregSpacedGrid with fixed points should not need runtime points."""
    grid = IrregSpacedGrid(points=np.array([1.0, 2.0, 3.0]))
    assert not grid.pass_points_at_runtime
    assert grid.n_points == 3


def test_runtime_grid_validation():
    """Grid with runtime-supplied points requires n_points >= 2."""
    with pytest.raises(Exception, match="at least 2"):
        IrregSpacedGrid(n_points=1)


def test_runtime_grid_requires_points_or_n_points():
    """Must specify either points or n_points."""
    with pytest.raises(Exception, match="Either points or n_points"):
        IrregSpacedGrid()


def test_get_params_template_lists_each_parameter_with_its_annotation():
    """Each regime maps its functions to their parameters' annotation names."""
    assert _make_model().get_params_template() == {
        "alive": {
            "utility": {},
            "borrowing_constraint": {},
            "next_wealth": {"interest_rate": "float"},
            "koopmans_aggregator": {"discount_factor": "FloatND"},
            "certainty_equivalent": {},
        },
        "dead": {"utility": {}},
    }


def test_get_params_template_is_built_from_plain_dicts():
    """Every branch of the template is a plain `dict` a user can fill in."""
    template = _make_model().get_params_template()
    branch_types = {type(template)} | {
        type(branch)
        for regime in template.values()
        for branch in (regime, *regime.values())
    }
    assert branch_types == {dict}


def test_get_params_template_returns_a_fresh_copy_on_each_call():
    """Filling in one returned template leaves the next call's template as is."""
    model = _make_model()
    model.get_params_template()["alive"]["next_wealth"]["interest_rate"] = "filled"
    template = model.get_params_template()
    assert template["alive"]["next_wealth"]["interest_rate"] == "float"


def _template_resources(wealth: ContinuousState) -> FloatND:
    return wealth


def _template_utility(
    *,
    consumption: ContinuousAction,
    resources: FloatND,
    consumptioz: FloatND,
    wealt: FloatND,
    resourcz: FloatND,
    coefficient: FloatND,
) -> FloatND:
    return consumption + resources + consumptioz + wealt + resourcz + coefficient


@pytest.fixture
def near_miss_model() -> Model:
    """Build a utility whose unresolved arguments resemble model names."""
    return _make_model(
        utility=_template_utility,
        functions={"resources": _template_resources},
    )


def test_near_miss_names_do_not_warn_at_build() -> None:
    """Unresolved names remain runtime parameters without build warnings."""
    with warnings.catch_warnings(record=True) as recorded:
        _make_model(
            utility=_template_utility,
            functions={"resources": _template_resources},
        )
    assert not recorded


def test_get_params_template_notes_near_miss_names(near_miss_model: Model) -> None:
    """The returned template suggests nearby state, action and function names."""
    assert near_miss_model.get_params_template()["alive"]["utility"] == {
        "consumptioz": "FloatND  [close to action 'consumption'; typo?]",
        "wealt": "FloatND  [close to state 'wealth'; typo?]",
        "resourcz": "FloatND  [close to function 'resources'; typo?]",
        "coefficient": "FloatND",
    }


def test_near_miss_names_leave_the_internal_template_plain(
    near_miss_model: Model,
) -> None:
    """Display suggestions leave the internal runtime parameter template intact."""
    assert near_miss_model._params_template["alive"]["utility"] == {
        "consumptioz": "FloatND",
        "wealt": "FloatND",
        "resourcz": "FloatND",
        "coefficient": "FloatND",
    }


def test_near_miss_template_is_a_fresh_copy(near_miss_model: Model) -> None:
    """Filling a displayed suggestion leaves the next template copy intact."""
    near_miss_model.get_params_template()["alive"]["utility"]["consumptioz"] = "filled"
    assert near_miss_model.get_params_template()["alive"]["utility"]["consumptioz"] == (
        "FloatND  [close to action 'consumption'; typo?]"
    )


def test_near_miss_parameters_are_still_required_at_solve(
    near_miss_model: Model,
) -> None:
    """Hand-written params must supply unresolved near-miss arguments."""
    with pytest.raises(InvalidParamsError, match="Missing required parameter"):
        near_miss_model.solve(
            params={"discount_factor": 0.95, "interest_rate": 0.05},
            log_level="debug",
        )


def _template_constraint(
    *,
    consumption: ContinuousAction,
    consumptioz: FloatND,
) -> FloatND:
    return consumption <= consumptioz


def test_get_params_template_notes_constraint_arguments() -> None:
    """A constraint's unresolved argument receives the same near-miss note."""
    model = _make_model(constraints={"limit": _template_constraint})
    assert model.get_params_template()["alive"]["limit"] == {
        "consumptioz": "FloatND  [close to action 'consumption'; typo?]"
    }


def _template_phase_utility(
    *,
    consumption: ContinuousAction,
    consumptioz: FloatND,
) -> FloatND:
    return consumption + consumptioz


def test_get_params_template_notes_both_phase_variants() -> None:
    """Both phases contribute arguments to the annotated user-facing template."""
    model = _make_model(
        utility=Phased(solve=_utility, simulate=_template_phase_utility)
    )
    assert model.get_params_template()["alive"]["utility"] == {
        "consumptioz": "FloatND  [close to action 'consumption'; typo?]"
    }


def test_get_params_template_omits_fixed_near_miss_parameters() -> None:
    """A bound near-miss argument needs no runtime template entry."""
    model = _make_model(
        utility=_template_phase_utility,
        fixed_params={"consumptioz": 0.5},
    )
    assert model.get_params_template()["alive"].get("utility", {}) == {}


def _template_threshold_utility(
    *, abcde: FloatND, abxde: FloatND, abxyf: FloatND
) -> FloatND:
    return abcde + abxde + abxyf


def test_get_params_template_uses_the_sequence_matcher_threshold() -> None:
    """A match at the similarity threshold is noted; a weaker match stays plain."""
    model = Model(
        regimes={
            "alive": UserRegime(
                functions={"utility": _template_threshold_utility},
                states={"abcde": LinSpacedGrid(start=0, stop=1, n_points=2)},
            ),
            "dead": UserRegime(functions={"utility": lambda: 0.0}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={},
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
    )
    assert model.get_params_template()["alive"]["utility"] == {
        "abxde": "FloatND  [close to state 'abcde'; typo?]",
        "abxyf": "FloatND",
    }


def _template_unrelated_utility(
    resourcz: FloatND,
) -> FloatND:
    return resourcz


def test_get_params_template_ignores_masked_and_other_regime_names() -> None:
    """A masked model helper available in another regime supplies no suggestion."""
    model = Model(
        regimes={
            "alive": UserRegime(
                functions={"utility": _template_unrelated_utility, "resources": None}
            ),
            "dead": UserRegime(functions={"utility": lambda resources: resources}),
        },
        functions={"resources": lambda: 1.0},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={},
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
    )
    assert model.get_params_template()["alive"]["utility"] == {"resourcz": "FloatND"}


def _template_grid_utility(*, consumption: ContinuousAction, point: FloatND) -> FloatND:
    return consumption + point


def test_get_params_template_leaves_runtime_grid_entries_plain() -> None:
    """Runtime-grid metadata is not an unresolved function argument."""
    model = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
        utility=_template_grid_utility,
        functions={"point": _template_resources},
    )
    assert model.get_params_template()["alive"]["wealth"]["points"] == "Float1D"


def _template_pruned_utility(
    consumptioz: FloatND,
) -> FloatND:
    return consumptioz


def _make_pruned_template_model(variable_kind: str) -> Model:
    """Build a model with a misspelled broadcast-variable argument."""
    variables = {"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)}
    return Model(
        regimes={
            "alive": UserRegime(functions={"utility": _template_pruned_utility}),
            "dead": UserRegime(functions={"utility": lambda: 0.0}),
        },
        states=variables if variable_kind == "state" else {},
        actions=variables if variable_kind == "action" else {},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        edges={},
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
    )


@pytest.mark.parametrize("variable_kind", ["state", "action"])
def test_near_miss_names_do_not_retain_broadcast_variables(variable_kind: str) -> None:
    """A misspelled argument does not consume the broadcast variable it resembles."""
    model = _make_pruned_template_model(variable_kind)
    assert "consumption" in model.pruned_variables["alive"]


@pytest.mark.parametrize("variable_kind", ["state", "action"])
def test_get_params_template_notes_pruned_broadcast_variables(
    variable_kind: str,
) -> None:
    """A misspelling retains its hint when the unread broadcast variable is pruned."""
    model = _make_pruned_template_model(variable_kind)
    assert model.get_params_template()["alive"]["utility"] == {
        "consumptioz": f"FloatND  [close to {variable_kind} 'consumption'; typo?]"
    }


@pytest.mark.parametrize("variable_kind", ["state", "action"])
def test_pruned_broadcast_variable_hints_survive_pickle(variable_kind: str) -> None:
    """Model persistence preserves suggestions for pruned broadcast variables."""
    model = _make_pruned_template_model(variable_kind)
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert restored.get_params_template() == model.get_params_template()


def _template_composed_resources_utility(
    *,
    consumption: ContinuousAction,
    resourcz: FloatND,
) -> FloatND:
    return -1.0 / consumption + resourcz


def test_get_params_template_notes_composed_function_names() -> None:
    """A composed resource function supplies its declared name to typo notes."""
    from tests.test_models import n_nbegm_toy  # noqa: PLC0415

    model = n_nbegm_toy.build_model(
        variant="negm",
        utility_function=_template_composed_resources_utility,
    )
    assert model.get_params_template()["alive"]["utility"] == {
        "resourcz": "FloatND  [close to function 'resources'; typo?]"
    }


def _template_constraint_reference_utility(
    *, consumption: ContinuousAction, borrowing_contraint: FloatND
) -> FloatND:
    return consumption + borrowing_contraint


def test_get_params_template_notes_near_constraint_names() -> None:
    """An unresolved argument can suggest a nearby constraint function's name."""
    model = _make_model(utility=_template_constraint_reference_utility)
    assert model.get_params_template()["alive"]["utility"] == {
        "borrowing_contraint": (
            "FloatND  [close to function 'borrowing_constraint'; typo?]"
        )
    }


def _template_impute_wealth(wealt: FloatND) -> ContinuousState:
    return wealt


def test_get_params_template_notes_carried_state_function_arguments() -> None:
    """A carried state's solve function receives notes for its unresolved inputs."""
    model = _make_model(
        wealth_grid=Phased(
            solve=_template_impute_wealth,
            simulate=LinSpacedGrid(start=1, stop=10, n_points=5),
        )
    )
    assert model.get_params_template()["alive"]["wealth"] == {
        "wealt": "FloatND  [close to state 'wealth'; typo?]"
    }


def test_runtime_grid_in_params_template():
    """IrregSpacedGrid with runtime-supplied points adds 'points' to template."""
    model = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
    )
    alive_template = model._params_template["alive"]
    assert "wealth" in alive_template
    assert "points" in alive_template["wealth"]


def test_solve_with_runtime_grid():
    """Solve should work when grid points are provided via params."""
    model = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"wealth": {"points": jnp.linspace(1, 10, 5)}},
    }
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug").values
    assert len(period_to_regime_to_V_arr) > 0


def test_runtime_grid_matches_fixed():
    """Runtime-supplied grid with same points should give same results as fixed."""
    points = jnp.linspace(1, 10, 5)

    # Fixed-points model
    model_fixed = _make_model(
        wealth_grid=IrregSpacedGrid(points=np.array(list(points.tolist()))),
    )
    params_fixed = {"discount_factor": 0.95, "interest_rate": 0.05}
    V_fixed = model_fixed.solve(params=params_fixed, log_level="debug").values

    # Runtime-points model
    model_runtime = _make_model(
        wealth_grid=IrregSpacedGrid(n_points=5),
    )
    params_runtime = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"wealth": {"points": points}},
    }
    V_runtime = model_runtime.solve(params=params_runtime, log_level="debug").values

    # V_arr for period 0, regime "alive" should match
    for period in V_fixed:
        if "alive" in V_fixed[period] and "alive" in V_runtime[period]:
            aaae(V_fixed[period]["alive"], V_runtime[period]["alive"])


def _make_action_grid_model(*, consumption_grid: IrregSpacedGrid) -> Model:
    """Create a 2-regime model where consumption is the runtime-points action grid."""
    alive = UserRegime(
        functions={"utility": _utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": consumption_grid},
        constraints={"borrowing_constraint": _borrowing_constraint},
    )
    dead = UserRegime(
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges=_EDGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
    )


def test_runtime_action_grid_in_params_template():
    """IrregSpacedGrid action with runtime-supplied points adds 'points' to template."""
    model = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    alive_template = model._params_template["alive"]
    assert "consumption" in alive_template
    assert "points" in alive_template["consumption"]


def test_solve_with_runtime_action_grid():
    """Solve should work when action grid points are provided via params."""
    model = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"consumption": {"points": jnp.linspace(0.1, 5.0, 5)}},
    }
    period_to_regime_to_V_arr = model.solve(params=params, log_level="debug").values
    assert len(period_to_regime_to_V_arr) > 0


def test_runtime_action_grid_matches_fixed():
    """Runtime action grid with same points gives same V as a fixed action grid."""
    points = jnp.linspace(0.1, 5.0, 5)

    model_fixed = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(points=np.array(list(points.tolist()))),
    )
    params_fixed = {"discount_factor": 0.95, "interest_rate": 0.05}
    V_fixed = model_fixed.solve(params=params_fixed, log_level="debug").values

    model_runtime = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    params_runtime = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"consumption": {"points": points}},
    }
    V_runtime = model_runtime.solve(params=params_runtime, log_level="debug").values

    for period in V_fixed:
        if "alive" in V_fixed[period] and "alive" in V_runtime[period]:
            aaae(V_fixed[period]["alive"], V_runtime[period]["alive"])


def test_runtime_action_grid_changes_solution():
    """Different runtime action points should yield different V (sanity check)."""
    model = _make_action_grid_model(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    base = {"discount_factor": 0.95, "interest_rate": 0.05}
    V_low = model.solve(
        params=base | {"alive": {"consumption": {"points": jnp.linspace(0.1, 1.0, 5)}}},
        log_level="debug",
    ).values
    V_high = model.solve(
        params=base | {"alive": {"consumption": {"points": jnp.linspace(0.1, 5.0, 5)}}},
        log_level="debug",
    ).values
    # Period 0 alive value should differ when the action support differs
    assert not jnp.allclose(V_low[0]["alive"], V_high[0]["alive"])


def _make_action_grid_model_with_stateful_dead(
    *, consumption_grid: IrregSpacedGrid
) -> Model:
    """Variant where `dead` has a `wealth` state so its utility depends on it.

    Used to surface NaN propagation when the simulate path forgets to
    substitute runtime-supplied action gridpoints.
    """

    def _alive_utility(
        *, consumption: ContinuousAction, wealth: ContinuousState
    ) -> FloatND:
        return jnp.log(consumption + 1) + 0.01 * wealth

    def _dead_utility(wealth: ContinuousState) -> FloatND:
        return jnp.log(wealth + 1)

    alive = UserRegime(
        functions={"utility": _alive_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
        state_transitions={
            "wealth": {
                "alive": _next_wealth,
                "dead": _next_wealth,
            },
        },
        actions={"consumption": consumption_grid},
        constraints={"borrowing_constraint": _borrowing_constraint},
    )
    dead = UserRegime(
        functions={"utility": _dead_utility},
        states={"wealth": LinSpacedGrid(start=1, stop=10, n_points=5)},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges=_EDGES,
        regime_id_class=RegimeId,
        initial_nodes={0: "alive"},
    )


def test_simulate_with_runtime_action_grid_no_nan() -> None:
    """Simulate must substitute runtime-supplied action gridpoints into the
    state-action space; otherwise the action grid is filled with NaN
    placeholders, optimal_actions become NaN, next_states propagate NaN to
    the dead regime, and `validate_V` raises.
    """
    model = _make_action_grid_model_with_stateful_dead(
        consumption_grid=IrregSpacedGrid(n_points=5),
    )
    params = {
        "discount_factor": 0.95,
        "interest_rate": 0.05,
        "alive": {"consumption": {"points": jnp.linspace(0.1, 5.0, 5)}},
    }
    initial_conditions = {
        "regime_id": jnp.array([RegimeId.alive, RegimeId.alive, RegimeId.alive]),
        "age": jnp.array([0.0, 0.0, 0.0]),
        "wealth": jnp.array([2.0, 5.0, 9.0]),
    }
    result = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="debug",
    )
    df = result.to_dataframe()
    assert not df["value"].isna().any()
