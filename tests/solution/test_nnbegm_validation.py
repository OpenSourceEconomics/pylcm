"""Validation tests for the regime-owned NNBEGM nesting contract.

`NestedConsumptionSavingsRegime` owns the two structural margins.  Its three
validation tiers establish their kind, existence, and disjointness before the
model-stage NNBEGM validator runs.  The solver-side validator therefore owns
only the dynamic condition that the outer carried-state law must not depend on
the inner liquid post-decision margin, directly or through a sibling law.
"""

import contextlib
import logging
from collections.abc import Iterator
from typing import cast

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import _lcm.solution.nnbegm as nnbegm_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.nnbegm_validation import validate_nnbegm_regimes
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.solution.nnbegm import (
    _fail_if_the_solve_grid_cannot_reconstruct_a_candidate,
)
from lcm import AgeGrid, LinSpacedGrid, Model
from lcm.consumption_savings_regime import (
    LiquidMargin,
    NestedConsumptionSavingsRegime,
    OuterContinuousMargin,
    outer_unchanged,
)
from lcm.exceptions import (
    ModelInitializationError,
    RegimeInitializationError,
    UnrepresentableOuterCandidateError,
)
from lcm.regime import Regime
from lcm.solvers import AdaptiveOuterMesh, FiniteOuterGrid
from lcm.typing import ContinuousAction, ContinuousState, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.test_models import n_nbegm_toy


def _valid_regime() -> NestedConsumptionSavingsRegime:
    return NestedConsumptionSavingsRegime(
        active=lambda age: age <= 20,
        states={
            "wealth": n_nbegm_toy.WEALTH_GRID,
            "illiquid": n_nbegm_toy.ILLIQUID_GRID,
        },
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": n_nbegm_toy.durable_transition,
        },
        actions={
            "consumption": n_nbegm_toy.CONSUMPTION_GRID,
            "illiquid_investment": n_nbegm_toy.ILLIQUID_INVESTMENT_GRID,
        },
        transition=n_nbegm_toy.next_regime,
        functions={
            "utility": n_nbegm_toy.utility,
            "new_illiquid": n_nbegm_toy.new_illiquid,
            "resources": n_nbegm_toy.resources,
            "liquid_savings": n_nbegm_toy.liquid_savings,
            "credited": n_nbegm_toy.credited,
        },
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="resources",
            post_decision_state="liquid_savings",
        ),
        outer_continuous=OuterContinuousMargin(
            state="illiquid",
            action="illiquid_investment",
            post_decision_state="new_illiquid",
            no_adjustment=outer_unchanged,
        ),
        solver=n_nbegm_toy.build_solver(variant="n_nbegm"),
    )


_VALID = _valid_regime()


def _validate(regime: Regime) -> None:
    """Run the NNBEGM dynamic contract check on a single-regime mapping."""
    validate_nnbegm_regimes(user_regimes={"alive": regime})


def test_valid_two_asset_toy_nnbegm_regime_passes_validation() -> None:
    """The smooth two-asset toy satisfies the nesting contract."""
    _validate(_VALID)


def test_margin_collision_is_rejected_before_solver_validation() -> None:
    """Tier one rejects a role name shared by the liquid and outer margins."""
    with pytest.raises(RegimeInitializationError, match="must not collide"):
        _VALID.replace(
            outer_continuous=OuterContinuousMargin(
                state="illiquid",
                action="consumption",
                post_decision_state="new_illiquid",
                no_adjustment=outer_unchanged,
            )
        )


def test_explicitly_masked_outer_function_is_rejected_before_model_build() -> None:
    """Tier two rejects an explicitly masked outer post-decision function."""
    with pytest.raises(RegimeInitializationError, match="explicitly masked"):
        _VALID.replace(
            functions={**dict(_VALID.functions), "new_illiquid": None},
        )


def _euler_law_reading_outer_margin(
    liquid_savings: FloatND, new_illiquid: ContinuousState
) -> ContinuousState:
    """A liquid Euler law whose return depends on the chosen durable stock."""
    return (1.0 + n_nbegm_toy.LIQUID_RATE) * liquid_savings + 0.01 * new_illiquid


def test_an_euler_law_reading_the_outer_margin_is_accepted() -> None:
    """The outer node is fixed while the inner solve runs, so this is valid."""
    regime = _VALID.replace(
        state_transitions={
            "wealth": _euler_law_reading_outer_margin,
            "illiquid": n_nbegm_toy.durable_transition,
        },
    )
    _validate(regime)


def _utility_coupling_consumption_and_durable_move(
    consumption: ContinuousAction, new_illiquid: ContinuousState
) -> FloatND:
    """A Cobb-Douglas composite of consumption and chosen durable service."""
    composite = consumption**0.8 * new_illiquid**0.2
    return composite ** (1.0 - n_nbegm_toy.RISK_AVERSION) / (
        1.0 - n_nbegm_toy.RISK_AVERSION
    )


def test_a_utility_composite_of_consumption_and_the_durable_is_accepted() -> None:
    """Conditional on an outer node, utility remains a 1-D inner problem."""
    regime = _VALID.replace(
        functions={
            **dict(_VALID.functions),
            "utility": _utility_coupling_consumption_and_durable_move,
        },
    )
    _validate(regime)


def test_a_regime_with_a_non_nested_solver_is_left_alone() -> None:
    """The dynamic NNBEGM check ignores regimes not bound to NNBEGM."""
    regime = Regime(
        active=_VALID.active,
        states=_VALID.states,
        state_transitions=_VALID.state_transitions,
        actions={"illiquid_investment": n_nbegm_toy.ILLIQUID_INVESTMENT_GRID},
        transition=_VALID.transition,
        functions=_VALID.functions,
        solver=n_nbegm_toy.build_solver(variant="brute"),
    )
    _validate(regime)


def _outer_law_reading_the_inner_savings(
    new_illiquid: ContinuousState, liquid_savings: FloatND
) -> ContinuousState:
    """A durable law whose carried stock depends on the inner savings choice."""
    return new_illiquid + 0.01 * liquid_savings


def _outer_law_reading_the_removed_action(
    illiquid_investment: ContinuousAction,
) -> ContinuousState:
    """A carried stock law that still asks for the enumerated action."""
    return illiquid_investment


def test_a_law_reading_the_removed_outer_action_is_rejected() -> None:
    """Inner solves bind the post-decision stock, never the generating action."""
    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_the_removed_action,
        },
    )

    with pytest.raises(ModelInitializationError, match="illiquid_investment"):
        _validate(regime)


def test_an_outer_law_reading_the_inner_savings_margin_is_rejected() -> None:
    """Direct dependence on the inner post-decision axis breaks nesting."""
    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_the_inner_savings,
        },
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        _validate(regime)


def _outer_law_reading_a_sibling_law(
    new_illiquid: ContinuousState, next_wealth: ContinuousState
) -> ContinuousState:
    """A durable law reaching the inner margin through the Euler-state law."""
    return new_illiquid + 0.01 * next_wealth


def test_an_outer_law_reaching_the_inner_margin_through_a_sibling_is_rejected() -> None:
    """The dependency traversal follows sibling state-transition laws."""
    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_a_sibling_law,
        },
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        _validate(regime)


def test_a_depreciating_outer_law_is_accepted() -> None:
    """A law reading only the chosen outer stock stays in scope."""

    def depreciating(new_illiquid: ContinuousState) -> ContinuousState:
        return 0.7 * new_illiquid

    regime = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": depreciating,
        },
    )
    _validate(regime)


def test_model_build_runs_the_dynamic_nnbegm_contract_check() -> None:
    """`Model(...)` invokes the same dynamic coupling guard."""
    alive = _VALID.replace(
        state_transitions={
            "wealth": n_nbegm_toy.next_wealth,
            "illiquid": _outer_law_reading_the_inner_savings,
        }
    )
    dead = Regime(
        transition=None,
        active=lambda age: age > 20,
        states={
            "wealth": n_nbegm_toy.WEALTH_GRID,
            "illiquid": n_nbegm_toy.ILLIQUID_GRID,
        },
        functions={"utility": n_nbegm_toy.terminal_utility},
    )
    with pytest.raises(ModelInitializationError, match="belongs to the inner margin"):
        Model(
            regimes={"alive": alive, "dead": dead},
            regime_id_class=n_nbegm_toy.RegimeId,
            ages=AgeGrid(start=20, stop=25, step="5Y"),
            fixed_params={"final_age_alive": 20},
        )


@pytest.mark.parametrize(
    ("outer_start", "outer_stop", "offending"),
    [
        pytest.param(-5.0, 20.0, "below", id="node-below-the-state-floor"),
        pytest.param(0.0, 30.0, "above", id="node-above-the-state-ceiling"),
        pytest.param(-5.0, 30.0, "below", id="nodes-outside-both-ends"),
    ],
)
def test_a_finite_outer_grid_reaching_outside_the_outer_state_domain_is_refused(
    outer_start: float, outer_stop: float, offending: str
) -> None:
    """Every finite outer node must name a value the outer state can hold.

    The finite search treats its nodes as post-decision targets for the outer
    state, so a node outside that state's own grid asks the solve to retain a
    stock the state cannot represent. The value function is undefined there and
    the read past the edge extrapolates silently, so the mismatch is refused
    where both grids are declared rather than discovered as an extrapolated
    continuation value.
    """
    with pytest.raises(
        (ModelInitializationError, RegimeInitializationError),
        match=r"outer (grid|search).*(outside|domain)|domain.*outer",
    ) as refusal:
        n_nbegm_toy.build_model(
            variant="n_nbegm",
            n_periods=3,
            outer_search=FiniteOuterGrid(
                grid=LinSpacedGrid(start=outer_start, stop=outer_stop, n_points=15)
            ),
        )
    # The message must name which edge was crossed: a refusal that cannot say
    # whether the search ran off the floor or the ceiling does not locate it.
    assert offending in str(refusal.value)


def test_a_finite_outer_grid_inside_the_outer_state_domain_is_accepted() -> None:
    """Narrowing the outer search inside the state's domain stays legal.

    The refusal is about leaving the declared domain, not about the two grids
    being identical, so a strictly interior search must still build.
    """
    n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        outer_search=FiniteOuterGrid(
            grid=LinSpacedGrid(start=2.0, stop=18.0, n_points=9)
        ),
    )


@pytest.mark.parametrize(
    "level",
    [logging.CRITICAL, logging.WARNING, logging.INFO, logging.DEBUG],
    ids=["off", "warning", "progress", "debug"],
)
def test_a_declared_node_the_solve_cannot_reconstruct_stops_the_solve(level) -> None:
    """A solve-grid node the solve cannot reconstruct refuses at every log level.

    The candidate is a declared node, so an inverse landing outside the outer
    state's domain means the declaration and the grids disagree -- a defect in
    the model, known before anything is published. Dropping it quietly leaves a
    policy bank whose contents depend on the diagnostic setting, so the same
    model would publish different policies at `log_level="off"` and `"debug"`.
    """
    logger = logging.getLogger(f"lcm.test.declared_node.{level}")
    logger.setLevel(level)

    with pytest.raises(
        UnrepresentableOuterCandidateError, match="could not reconstruct"
    ):
        _fail_if_the_solve_grid_cannot_reconstruct_a_candidate(
            logger=logger,
            dropped=jnp.asarray([True]),
            n_live=jnp.asarray([True]),
            regime_name="working",
            period=0,
        )


_MESH = AdaptiveOuterMesh(
    initial_grid=n_nbegm_toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)


def _lossy_new_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """`s' = Z + Iz`, with the outer action round-tripped through float16.

    Structurally affine with unit slope, so the coefficient certificate reads a
    clean map -- while the round trip has quantized the action the solve would
    later have to recover.
    """
    hop = jnp.asarray(illiquid_investment).astype(jnp.float16).astype(jnp.float64)
    return illiquid + hop


@pytest.mark.parametrize("outer_search", [None, _MESH], ids=["finite", "adaptive"])
def test_a_lossy_outer_map_is_refused_before_any_policy_is_published(
    outer_search: AdaptiveOuterMesh | None,
) -> None:
    """No outer search may publish a replay policy for a map it cannot invert.

    The refusal belongs to the declared map, not to the search that ranks it, so
    it must reach the finite grid and the adaptive mesh identically. A route that
    publishes instead hands simulation a policy whose outer action cannot be
    recovered, and replay then selects a different decision without saying so.
    """
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        outer_post_decision_function=_lossy_new_illiquid,
        outer_search=outer_search,
    )
    with pytest.raises(RegimeInitializationError, match="cannot invert"):
        model.solve(
            params={"discount_factor": 0.95},
            log_level="off",
            return_simulation_policy=True,
        )


def _doubled_new_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    """`s' = Z + 2 * Iz`: affine in the outer action, at two units per unit."""
    return illiquid + 2.0 * illiquid_investment


def _doubled_model(*, outer_search: AdaptiveOuterMesh | None) -> Model:
    return n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        outer_post_decision_function=_doubled_new_illiquid,
        outer_search=outer_search,
    )


_REPLAY_INITIAL = {
    "wealth": jnp.array([4.3, 11.7, 19.9, 8.1]),
    "illiquid": jnp.array([1.37, 6.6, 13.2, 17.5]),
    "age": jnp.full(4, 20.0),
    "regime_id": jnp.zeros(4, dtype=jnp.int32),
}


def test_the_continuous_outer_mesh_publishes_a_policy_for_a_one_for_one_map() -> None:
    """The mesh publishes a replay policy for the map its replay can invert.

    The control for the refusal below: continuous-outer replay recovers the
    outer action by subtracting the map's offset from the chosen stock, so a map
    that moves one unit of stock per unit of action is exactly what it inverts.
    """
    model = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=3, outer_search=_MESH)

    _, policies = model.solve(
        params={"discount_factor": 0.95},
        log_level="off",
        return_simulation_policy=True,
    )

    assert isinstance(policies[0]["alive"], NestedEGMSimPolicy)


def test_the_continuous_outer_mesh_refuses_a_map_its_replay_cannot_invert() -> None:
    """A map moving more than one unit of stock per unit of action is refused.

    Continuous-outer replay recovers the outer action by subtracting the map's
    offset from the chosen stock, which is the inverse only at one unit per
    unit. Publishing for any other coefficient hands simulation a decision it
    cannot reproduce, and the pair it emits is then the generic action-grid
    winner rather than the solved one.
    """
    model = _doubled_model(outer_search=_MESH)

    with pytest.raises(RegimeInitializationError, match="Continuous-outer replay"):
        model.solve(
            params={"discount_factor": 0.95},
            log_level="off",
            return_simulation_policy=True,
        )


def test_the_uninvertible_map_refusal_is_identical_on_both_replay_routes() -> None:
    """Split and automatic replay report the same refusal for the same map.

    Both routes read one certificate of the declared map, so a map neither can
    invert must stop them the same way rather than one refusing and the other
    proceeding on a policy the first declined to publish.
    """
    model = _doubled_model(outer_search=_MESH)

    with pytest.raises(RegimeInitializationError) as split:
        model.solve(
            params={"discount_factor": 0.95},
            log_level="off",
            return_simulation_policy=True,
        )
    with pytest.raises(RegimeInitializationError) as automatic:
        model.simulate(
            params={"discount_factor": 0.95},
            initial_conditions=dict(_REPLAY_INITIAL),
            period_to_regime_to_V_arr=None,
            log_level="off",
            seed=42,
        )

    assert str(split.value) == str(automatic.value)


def test_the_finite_outer_grid_recovers_the_action_a_doubled_map_needs() -> None:
    """The finite outer search inverts a doubled map instead of refusing it.

    It recovers the action from the retained target by the map's own certified
    coefficient, so reaching the same stock through a map that moves two units
    per unit of action takes exactly half the action.
    """
    params = {"discount_factor": 0.95}
    one_for_one = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=3)

    unit_investment = (
        one_for_one.simulate(
            params=params,
            initial_conditions=dict(_REPLAY_INITIAL),
            period_to_regime_to_V_arr=None,
            log_level="off",
            seed=42,
        )
        .to_dataframe()["illiquid_investment"]
        .to_numpy()
    )
    doubled_investment = (
        _doubled_model(outer_search=None)
        .simulate(
            params=params,
            initial_conditions=dict(_REPLAY_INITIAL),
            period_to_regime_to_V_arr=None,
            log_level="off",
            seed=42,
        )
        .to_dataframe()["illiquid_investment"]
        .to_numpy()
    )

    aaae(doubled_investment, unit_investment / 2.0, decimal=DECIMAL_PRECISION)


def test_the_continuous_outer_mesh_refuses_a_second_passive_continuous_state() -> None:
    """Two passive continuous stocks in one regime stop the continuous solve.

    Continuous-outer replay indexes the published branch rows at the subject's
    own passive stock, and it brackets exactly one such axis. A regime carrying
    two of them has rows the replay cannot address, so the solve refuses rather
    than publishing a policy simulation would answer with the action-grid winner.
    """
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=_MESH,
        second_passive_state=True,
    )

    with pytest.raises(RegimeInitializationError, match="passive continuous state"):
        model.solve(
            params={"discount_factor": 0.95},
            log_level="off",
            return_simulation_policy=True,
        )


def _new_illiquid_reading_consumption(
    illiquid: ContinuousState,
    illiquid_investment: ContinuousAction,
    consumption: ContinuousAction,
) -> ContinuousState:
    """`s' = Z + Iz + 0.01 c`: affine in the outer action, reads the inner one."""
    return illiquid + illiquid_investment + 0.01 * consumption


def _published_capability(policy: object) -> OuterReplayCapability:
    """The replay capability the published NNBEGM policy carries."""
    return cast("NNBEGMSimPolicy | NestedEGMSimPolicy", policy).replay_capability


def test_the_continuous_outer_mesh_refuses_an_outer_map_replay_cannot_bind() -> None:
    """An outer map reading the inner action stops the continuous solve.

    Continuous-outer replay evaluates the declared map at each subject from
    that subject's states, the regime's parameters, `period` and `age`. It has
    no inner action to supply — the inner action is what the replay is solving
    for — so a map that reads one cannot be evaluated where the decision is
    rebuilt, and publishing for it would hand simulation the action-grid winner.
    """
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=_MESH,
        outer_post_decision_function=_new_illiquid_reading_consumption,
    )

    with pytest.raises(RegimeInitializationError, match="consumption"):
        model.solve(
            params={"discount_factor": 0.95},
            log_level="off",
            return_simulation_policy=True,
        )


def test_both_outer_searches_publish_the_same_certified_outer_inverse() -> None:
    """Finite and adaptive solves settle one answer to the inversion question.

    The two searches differ in how they explore the outer margin, never in what
    they conclude about the declared map. Publishing different inverses would
    let one route replay a stock the other refused.
    """
    params = {"discount_factor": 0.95}
    _, finite = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params=params, log_level="off", return_simulation_policy=True
    )
    _, adaptive = n_nbegm_toy.build_model(
        variant="n_nbegm", n_periods=2, outer_search=_MESH
    ).solve(params=params, log_level="off", return_simulation_policy=True)

    assert (
        _published_capability(finite[0]["alive"]).inverse
        == _published_capability(adaptive[0]["alive"]).inverse
    )


def test_the_published_capability_records_the_outer_state_domain() -> None:
    """The published verdict carries the stock domain the solve admitted."""
    _, policies = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=2).solve(
        params={"discount_factor": 0.95},
        log_level="off",
        return_simulation_policy=True,
    )

    inverse = _published_capability(policies[0]["alive"]).inverse
    assert (inverse.low, inverse.high) == (0.0, 20.0)


@contextlib.contextmanager
def _recorded_published_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[object]]:
    """Record every replay policy object either outer search constructs."""
    constructed: list[object] = []
    for name in ("NNBEGMSimPolicy", "NestedEGMSimPolicy"):
        original = getattr(nnbegm_module, name)

        def record(*, _original=original, **kwargs: object) -> object:
            policy = _original(**kwargs)
            constructed.append(policy)
            return policy

        monkeypatch.setattr(nnbegm_module, name, record)
    yield constructed


_SOLVE_ARGS = {
    "params": {"discount_factor": 0.95},
    "log_level": "off",
}

_UNSUPPORTED_ROUTES = {
    "adaptive_solve": lambda model: model.solve(
        **_SOLVE_ARGS, return_simulation_policy=True
    ),
    "automatic_simulation": lambda model: model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions=dict(_REPLAY_INITIAL),
        period_to_regime_to_V_arr=None,
        log_level="off",
    ),
    "finite_solve": lambda model: model.solve(
        **_SOLVE_ARGS, return_simulation_policy=True
    ),
    "split_replay": lambda model: model.solve(
        **_SOLVE_ARGS, return_simulation_policy=True
    ),
}

_ADAPTIVE_ROUTES = frozenset({"adaptive_solve", "automatic_simulation"})


@pytest.mark.parametrize("route", sorted(_UNSUPPORTED_ROUTES))
def test_no_route_returns_a_replay_policy_for_an_uninvertible_map(
    route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsupported declaration leaves every route with no policy object at all.

    Refusing is not enough on its own: a route that built a replay policy and
    then raised would still have to be trusted never to hand it on. Recording
    every construction shows the refusal lands before any policy exists — on the
    finite grid and the adaptive mesh, whether simulation solves for itself or
    the caller solves first and supplies the pair. The split route is covered by
    the solve it must run first: with no policy to supply, nothing can be
    replayed.
    """
    model = n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        outer_post_decision_function=_lossy_new_illiquid,
        outer_search=_MESH if route in _ADAPTIVE_ROUTES else None,
    )

    with (
        _recorded_published_policies(monkeypatch) as constructed,
        pytest.raises(RegimeInitializationError),
    ):
        _UNSUPPORTED_ROUTES[route](model)

    assert constructed == []
