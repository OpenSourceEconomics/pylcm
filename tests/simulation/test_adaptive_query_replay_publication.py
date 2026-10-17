"""Adaptive N-NB-EGM replay with a canonical callable borrowing floor.

A model whose liquid margin carries a callable floor keeps its nested replay
policy. The composite kernel declares typed producers for it, model authority
binds the floor replay evaluates to the installed canonical runtime bound,
simulation evaluates that floor at each subject's current state, and mapped
canonical scoring folds every evaluated row's mandatory integrity record into
one host check.
"""

import dataclasses
from types import MappingProxyType, SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _lcm.simulation.simulate as simulation_module
from _lcm.continuation import EGMQuerySpec
from _lcm.egm.budget import DCEGM_BUDGET_CONSTRAINT_NAME
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.engine import NNBEGMPolicyRead, Regime
from _lcm.execution.core_program import ProgramScope, core_program_graph
from _lcm.params.edges import regime_kernel_params
from _lcm.post_decision_bound import PostDecisionBoundSpec
from _lcm.solution.feasibility import BoundedResult, BoundInputStatus
from _lcm.solution.model_authority import (
    _validate_adaptive_nnbegm_policy_query_authority,
)
from _lcm.typing import ConstraintFunction
from lcm import ExecutionConfig, Model, post_decision_lower_bound
from lcm.consumption_savings_regime import NestedConsumptionSavingsRegime
from lcm.exceptions import InvalidValueFunctionError
from lcm.solver_api import SIMULATION_POLICY
from lcm.typing import BoolND, ContinuousState, FloatND
from tests.simulation import test_simulate_n_nbegm_continuous as adaptive
from tests.test_models import n_nbegm_toy as toy

_PARAMS = {"discount_factor": 0.95}


def _illiquid_floor(illiquid: ContinuousState) -> FloatND:
    """A current-state floor that differs across subjects."""
    return 0.1 * illiquid


def _model(*, lower) -> Model:
    """Return the adaptive toy, with `lower` as its callable floor when given."""
    config = ExecutionConfig(device_memory_bytes=None)
    seed = toy.build_model(
        variant="n_nbegm",
        n_periods=3,
        outer_search=adaptive._MESH,
        execution_config=config,
    )
    if lower is None:
        return seed
    alive = seed.user_regimes["alive"]
    assert isinstance(alive, NestedConsumptionSavingsRegime)
    feature = alive.replace(
        constraints={
            "borrowing": post_decision_lower_bound(margin=alive.liquid, lower=lower)
        }
    )
    return Model(
        regimes={**seed.user_regimes, "alive": feature},
        ages=seed.ages,
        edges=seed.edges,
        fixed_params=seed.fixed_params,
        initial_nodes=seed.initial_nodes,
        regime_id_class=toy.RegimeId,
        execution_config=config,
    )


@pytest.fixture(scope="module")
def callable_model() -> Model:
    return _model(lower=adaptive._adaptive_zero_floor)


@pytest.fixture(scope="module")
def state_floor_model() -> Model:
    return _model(lower=_illiquid_floor)


def test_callable_floor_kernel_declares_typed_nested_policy_producers(
    callable_model: Model,
) -> None:
    """Both roles retain one replay-scoped producer of the nested policy type."""
    regime = callable_model._regimes["alive"]
    graph = core_program_graph(kernel=regime.solution.period_kernels[0])
    producers = {
        name: program.retained_artifact_payload_types[SIMULATION_POLICY]
        for name, program in graph.items()
        if SIMULATION_POLICY in program.retained_artifact_keys
        and program.scope is ProgramScope.REPLAY
    }
    assert regime.simulation.replay_route.consumer_route == "nnbegm_nested"
    assert producers == {
        "keeper:query-replay": NestedEGMSimPolicy,
        "adjuster:query-replay": NestedEGMSimPolicy,
    }


def test_model_authority_binds_the_callable_floor_adapter(
    callable_model: Model,
) -> None:
    """Retained adaptive replay authenticates against the canonical floor DAG."""
    regime = callable_model._regimes["alive"]
    assert _validate_adaptive_nnbegm_policy_query_authority(
        regime=regime, expected=_query_authority(regime)
    )


def _query_authority(regime: Regime) -> EGMQuerySpec:
    expected = regime.solution.query_authority
    assert expected is not None
    return expected


def _always_feasible(**_inputs: FloatND) -> BoolND:
    """A budget constraint carrying no installed model bound."""
    return jnp.asarray(a=True)


def _mutated(*, regime: Regime, mutation: str) -> tuple[Regime, EGMQuerySpec]:
    """Return a regime and model query spec with one authority input substituted."""
    expected = _query_authority(regime)
    adaptive_outer = expected.adaptive_outer
    assert adaptive_outer is not None
    simulation = regime.simulation
    read = simulation.egm_policy_read
    assert isinstance(read, NNBEGMPolicyRead)
    if mutation == "outer_action_role":
        simulation = dataclasses.replace(
            simulation,
            egm_policy_read=dataclasses.replace(
                read, outer_action_name=read.inner_action_name
            ),
        )
    elif mutation == "inner_action_role":
        simulation = dataclasses.replace(
            simulation,
            egm_policy_read=dataclasses.replace(
                read, inner_action_name=read.outer_action_name
            ),
        )
    elif mutation == "runtime_bound_removed":
        simulation = dataclasses.replace(simulation, post_decision_bounds=())
    elif mutation == "intrinsic_budget_unbound":
        pools = {}
        for period, (
            functions,
            constraints,
        ) in simulation.feasibility_pools_by_period.items():
            unbound = dict(constraints)
            unbound[DCEGM_BUDGET_CONSTRAINT_NAME] = cast(
                "ConstraintFunction", _always_feasible
            )
            pools[period] = (functions, MappingProxyType(unbound))
        simulation = dataclasses.replace(
            simulation, feasibility_pools_by_period=MappingProxyType(pools)
        )
    elif mutation == "floor_reads_liquid_state":
        expected = dataclasses.replace(
            expected,
            evaluator_inputs=tuple(
                (purpose, (*inputs, expected.bound.liquid_state))
                if purpose == "lower_bound"
                else (purpose, inputs)
                for purpose, inputs in expected.evaluator_inputs
            ),
        )
    elif mutation == "floor_reads_outer_action":
        expected = dataclasses.replace(
            expected,
            evaluator_inputs=tuple(
                (purpose, (*inputs, adaptive_outer.outer_action))
                if purpose == "lower_bound"
                else (purpose, inputs)
                for purpose, inputs in expected.evaluator_inputs
            ),
        )
    else:
        raise ValueError(mutation)
    return dataclasses.replace(regime, simulation=simulation), expected


@pytest.mark.parametrize(
    ("mutation", "refusal"),
    [
        ("outer_action_role", "Adaptive policy query roles differ"),
        ("inner_action_role", "Adaptive policy query roles differ"),
        ("runtime_bound_removed", "omits its canonical runtime bound"),
        ("intrinsic_budget_unbound", "intrinsic budget omits its model bound"),
        ("floor_reads_liquid_state", "floor inputs differ from fixed-state"),
        ("floor_reads_outer_action", "floor inputs differ from fixed-state"),
    ],
)
def test_model_authority_refuses_a_substituted_floor_adapter(
    *, callable_model: Model, mutation: str, refusal: str
) -> None:
    """Each substituted role, bound or floor input is refused before solving."""
    regime, expected = _mutated(
        regime=callable_model._regimes["alive"], mutation=mutation
    )
    with pytest.raises(TypeError, match=refusal):
        _validate_adaptive_nnbegm_policy_query_authority(
            regime=regime, expected=expected
        )


def _floor_payload(*, savings_lower_bound: float) -> NestedEGMSimPolicy:
    """Return a nested payload carrying only the fields the floor reads."""
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "liquid_state_name", "wealth")
    object.__setattr__(payload, "savings_lower_bound", savings_lower_bound)
    return payload


def _current_floor(*, model: Model, payload: NestedEGMSimPolicy) -> np.ndarray:
    regime = model._regimes["alive"]
    assert model.ages is not None
    return np.asarray(
        simulation_module._nested_current_floor(
            payload=payload,
            regime=regime,
            states={
                "wealth": jnp.asarray([4.3, 11.7, 19.9]),
                "illiquid": jnp.asarray([1.37, 6.6, 13.2]),
            },
            flat_params=regime_kernel_params(
                model._process_params(_PARAMS, runtime_checks=True),
                regime_name="alive",
            ),
            period=0,
            age=model.ages.values[0],
            n_subjects=3,
        )
    )


def test_nested_floor_is_the_callable_at_each_current_state(
    state_floor_model: Model,
) -> None:
    """Replay's budget uses the callable at each subject, not the scalar descriptor."""
    floor = _current_floor(
        model=state_floor_model, payload=_floor_payload(savings_lower_bound=-7.0)
    )
    np.testing.assert_allclose(floor, 0.1 * np.asarray([1.37, 6.6, 13.2]), rtol=1e-6)


def test_nested_floor_of_a_constant_callable_is_unchanged(
    callable_model: Model,
) -> None:
    """A constant callable floor reproduces the toy's zero floor everywhere."""
    floor = _current_floor(
        model=callable_model, payload=_floor_payload(savings_lower_bound=-7.0)
    )
    np.testing.assert_array_equal(floor, [0.0, 0.0, 0.0])


def test_nested_floor_without_a_callable_keeps_the_solver_floor() -> None:
    """Without a canonical callable floor, replay keeps the scalar savings floor."""
    floor = _current_floor(
        model=_model(lower=None), payload=_floor_payload(savings_lower_bound=0.25)
    )
    np.testing.assert_array_equal(floor, [0.25, 0.25, 0.25])


class _StubRegime(Regime):
    """Engine regime carrying only the fields the mapped scoring reaches."""

    def __init__(self, *, simulation: SimpleNamespace) -> None:
        object.__setattr__(self, "name", "alive")
        object.__setattr__(self, "simulation", simulation)


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def enable_x64(request: pytest.FixtureRequest):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield request.param
    jax.config.update("jax_enable_x64", previous)


def _mapped_baseline(
    *,
    monkeypatch: pytest.MonkeyPatch,
    bound: PostDecisionBoundSpec,
    nonfinite_outer: float,
) -> tuple[MappingProxyType[str, FloatND], FloatND, BoolND]:
    """Score a two-row bank whose row at `nonfinite_outer` reports a bad floor."""

    def evaluate(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        return BoundedResult(
            output=(outer * 1.0, jnp.ones_like(outer, dtype=bool)),
            bound_status=(
                BoundInputStatus(
                    nonfinite_floor=jnp.any(outer == nonfinite_outer),
                    invalid_shifted_grid=jnp.asarray(a=False),
                ),
            ),
        )

    def grid_q(*, candidate_actions, **_):
        outer = jnp.asarray(candidate_actions["outer"])
        return jnp.zeros_like(outer), jnp.ones_like(outer, dtype=bool)

    monkeypatch.setattr(simulation_module, "_evaluate_canonical_Q_at_actions", evaluate)
    monkeypatch.setattr(simulation_module, "_canonical_Q_at_actions", grid_q)
    monkeypatch.setattr(
        simulation_module,
        "_nested_actions_are_intrinsically_admissible",
        lambda *, actions, **_: jnp.ones_like(
            jnp.asarray(actions["outer"]), dtype=bool
        ),
    )
    payload = object.__new__(NestedEGMSimPolicy)
    object.__setattr__(payload, "outer_action_name", "outer")
    object.__setattr__(payload, "inner_action_name", "inner")
    return simulation_module._nested_grid_baseline(
        payload=payload,
        grid_actions=MappingProxyType(
            {"outer": jnp.asarray([0.5]), "inner": jnp.asarray([1.0])}
        ),
        regime=_StubRegime(simulation=SimpleNamespace(post_decision_bounds=(bound,))),
        states={"state": jnp.asarray([0.0])},
        canonical_states={"state": jnp.asarray([0.0])},
        action_names=("outer", "inner"),
        next_regime_to_V_arr=MappingProxyType({}),
        flat_params=MappingProxyType({}),
        period=0,
        age=jnp.asarray(0.0),
        replay_candidate=(
            jnp.asarray([[1.0], [2.0]]),
            jnp.asarray([[1.0], [1.0]]),
            jnp.asarray([[True], [True]]),
        ),
    )


def _callable_bound(model: Model) -> PostDecisionBoundSpec:
    (bound,) = (
        bound
        for bound in model._regimes["alive"].simulation.post_decision_bounds
        if bound.kind == "dag"
    )
    return bound


def test_mapped_bank_scoring_with_finite_floors_selects_the_best_row(
    *, monkeypatch: pytest.MonkeyPatch, callable_model: Model, enable_x64: bool
) -> None:
    """A clean bank passes the single host check and keeps canonical ranking."""
    del enable_x64
    actions, value, admissible = _mapped_baseline(
        monkeypatch=monkeypatch,
        bound=_callable_bound(callable_model),
        nonfinite_outer=99.0,
    )
    np.testing.assert_array_equal(np.asarray(actions["outer"]), [2.0])
    np.testing.assert_array_equal(np.asarray(value), [2.0])
    np.testing.assert_array_equal(np.asarray(admissible), [True])


@pytest.mark.parametrize("nonfinite_outer", [1.0, 2.0], ids=["losing", "winning"])
def test_mapped_bank_scoring_refuses_a_nonfinite_floor_on_any_row(
    *,
    monkeypatch: pytest.MonkeyPatch,
    callable_model: Model,
    enable_x64: bool,
    nonfinite_outer: float,
) -> None:
    """A nonfinite floor on a losing or winning mapped row fails at the host."""
    del enable_x64
    with pytest.raises(InvalidValueFunctionError, match="nonfinite borrowing floor"):
        _mapped_baseline(
            monkeypatch=monkeypatch,
            bound=_callable_bound(callable_model),
            nonfinite_outer=nonfinite_outer,
        )
