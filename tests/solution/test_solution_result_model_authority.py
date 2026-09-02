"""Regressions for the model authority over labelled solution results."""

from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import pytest

import lcm.model as model_module
from _lcm.egm.published_policy import NNBEGMSimPolicy
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import SIMULATION_POLICY, ArtifactStore
from tests.solution.test_solution_result import _small_grid_search_inputs
from tests.test_models import n_nbegm_discrete_toy as discrete_toy
from tests.test_models.n_nbegm_toy import RegimeId

_PARAMS = {"discount_factor": 0.95, "alive": {"premium": 1.0}}
_INITIAL = {
    "wealth": jnp.asarray([4.0, 15.0, 24.0]),
    "illiquid": jnp.asarray([0.0, 12.0, 20.0]),
    "age": jnp.full(3, 20.0),
    "regime_id": jnp.full(3, RegimeId.alive, dtype=jnp.int32),
}


def _forward_loop_must_not_run(**_kwargs: object) -> None:
    raise AssertionError("forward simulation ran before model-authority preflight")


def test_co_mutated_value_and_schema_dtype_is_rejected_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    coordinate = next(iter(solution.metadata.value_schemas))
    period, regime_name = coordinate
    original = solution.values[period][regime_name]
    replacement = original.astype(jnp.float16)

    values = {
        p: dict(regime_to_value) for p, regime_to_value in solution.values.items()
    }
    values[period][regime_name] = replacement
    schemas = dict(solution.metadata.value_schemas)
    schemas[coordinate] = replace(schemas[coordinate], dtype=str(replacement.dtype))
    malformed = replace(
        solution,
        values=values,
        metadata=replace(solution.metadata, value_schemas=schemas),
    )

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="dtype"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def _finite_discrete_solution():
    model = discrete_toy.build_model(variant="n_nbegm", n_periods=2)
    solution = model.solve(params=_PARAMS, log_level="off")
    ref = next(ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY)
    policy = cast("NNBEGMSimPolicy", solution.replay_artifacts[ref])
    return model, solution, ref, policy


def test_nnbegm_unknown_discrete_code_is_rejected_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = _finite_discrete_solution()
    codes = policy.candidate_discrete_actions
    assert codes is not None
    malformed_codes = codes.at[0, 0].set(jnp.int32(99))
    malformed_policy = replace(policy, candidate_discrete_actions=malformed_codes)
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: malformed_policy}
        ),
    )

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match=r"code|domain"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_nnbegm_noncanonical_discrete_dtype_is_rejected_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = _finite_discrete_solution()
    codes = policy.candidate_discrete_actions
    assert codes is not None
    malformed_policy = replace(policy)
    # Construct an object that could arrive from a corrupted or older serialized
    # result without asking the current constructor to authenticate it first.
    object.__setattr__(
        malformed_policy,
        "candidate_discrete_actions",
        codes.astype(jnp.int16),
    )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: malformed_policy}
        ),
    )

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="dtype"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_nnbegm_self_consistent_omitted_state_axis_is_rejected_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = _finite_discrete_solution()
    assert policy.state_names == ("wealth", "illiquid")
    malformed_policy = replace(
        policy,
        state_names=("wealth",),
        candidate_inner_action=policy.candidate_inner_action[..., 0],
        candidate_outer_target=policy.candidate_outer_target[..., 0],
        candidate_value=policy.candidate_value[..., 0],
    )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: malformed_policy}
        ),
    )

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"state_names|axis|state/action roles",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_nnbegm_co_mutated_float_payload_dtype_is_rejected_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, solution, ref, policy = _finite_discrete_solution()
    malformed_policy = replace(
        policy,
        candidate_inner_action=policy.candidate_inner_action.astype(jnp.float16),
        candidate_outer_target=policy.candidate_outer_target.astype(jnp.float16),
        candidate_value=policy.candidate_value.astype(jnp.float16),
        outer_grid_values=policy.outer_grid_values.astype(jnp.float16),
    )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {ref: malformed_policy}
        ),
    )

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="dtype"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )
