"""Real finite-policy leaves retain their addresses and remain dynamic operands."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.published_policy import NNBEGMSimPolicy
from _lcm.execution.core_program import (
    CoreProgram,
    _value_read_argument_leaf,
    materialize_core_program,
)
from _lcm.execution.value_transfer import ValueArtifactKind, ValueInputChannel
from _lcm.simulation import compile as simulation_compile
from _lcm.simulation.policy_programs import (
    POLICY_PREPARE,
    POLICY_RANK,
    ReplayPayload,
    _Rank,
)
from _lcm.simulation.programs import _SubjectTiled
from _lcm.simulation.runtime import (
    SimulationDispatchContext,
    SimulationRuntime,
    _build_context,
)
from lcm import ExecutionConfig, Model
from lcm.solver_api import SIMULATION_POLICY
from tests.test_models import n_nbegm_discrete_toy as discrete_toy
from tests.test_models import n_nbegm_toy as smooth_toy

type CandidateBank = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
type RankedBank = tuple[Mapping[str, jax.Array], jax.Array, jax.Array]


@dataclass(frozen=True, kw_only=True)
class _Call:
    program: CoreProgram
    arguments: Mapping[str, object]
    result: object
    n_subjects: int


@dataclass(frozen=True, kw_only=True, eq=False)
class _RecordingRuntime(SimulationRuntime):
    calls: list[_Call] = field(default_factory=list)

    def dispatch(
        self,
        *,
        program: CoreProgram,
        arguments: Mapping[str, object],
        period: int,
        n_subjects: int,
        residency: SimulationDispatchContext | None = None,
    ) -> object:
        result = super().dispatch(
            program=program,
            arguments=arguments,
            period=period,
            n_subjects=n_subjects,
            residency=residency,
        )
        if program.name in (POLICY_PREPARE, POLICY_RANK):
            self.calls.append(
                _Call(
                    program=program,
                    arguments=arguments,
                    result=result,
                    n_subjects=n_subjects,
                )
            )
        return result


@dataclass(frozen=True, kw_only=True)
class _ProducedCase:
    policy: NNBEGMSimPolicy
    runtime: _RecordingRuntime
    preparation: _Call
    ranking: _Call


@pytest.fixture(
    scope="module",
    params=((False, 0), (False, 1), (True, 0), (True, 1)),
    ids=("smooth-row0", "smooth-row1", "discrete-row0", "discrete-row1"),
)
def produced_case(request: pytest.FixtureRequest) -> _ProducedCase:
    """Obtain real producer arrays and arguments from actual public dispatch."""
    discrete, row = request.param
    factory = discrete_toy if discrete else smooth_toy
    base = factory.build_model(variant="n_nbegm", n_periods=2)
    model = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=smooth_toy.RegimeId,
        fixed_params=base.fixed_params,
        execution_config=ExecutionConfig(axis_widths={"subject": 1}),
    )
    params = {"discount_factor": 0.95}
    if discrete:
        params["premium"] = 0.25
    solution = model.solve(params=params, log_level="off")
    policy = solution.replay_artifacts.project(SIMULATION_POLICY)[0]["alive"]
    assert isinstance(policy, NNBEGMSimPolicy)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(simulation_compile, "SimulationRuntime", _RecordingRuntime)
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "wealth": np.array([4.3, 11.7]),
                "illiquid": np.array([1.37, 6.6]),
                "age": np.array([20.0, 20.0]),
                "regime_id": np.zeros(2, dtype=np.int32),
            },
            log_level="off",
            seed=17,
        )
    runtime = model._runtime_regimes_for_shape(compile_batch_size=1)[
        "alive"
    ].simulation.programs.executor
    assert isinstance(runtime, _RecordingRuntime)
    assert [call.program.name for call in runtime.calls] == [
        POLICY_PREPARE,
        POLICY_RANK,
        POLICY_PREPARE,
        POLICY_RANK,
    ]
    assert [call.n_subjects for call in runtime.calls] == [1, 1, 1, 1]
    preparation, ranking = runtime.calls[2 * row : 2 * row + 2]
    runtime.calls.clear()
    return _ProducedCase(
        policy=policy, runtime=runtime, preparation=preparation, ranking=ranking
    )


def _assert_correspondence(*, case: _ProducedCase, call: _Call) -> None:
    wrapped = ReplayPayload.from_policy(case.policy)
    producer_leaves, producer_structure = jax.tree_util.tree_flatten_with_path(
        case.policy
    )
    restored = wrapped.restore()
    assert type(restored) is NNBEGMSimPolicy
    assert wrapped.structure == producer_structure
    assert restored.state_names == case.policy.state_names
    assert restored.inner_action_name == case.policy.inner_action_name
    assert restored.outer_action_name == case.policy.outer_action_name
    assert restored.discrete_action_names == case.policy.discrete_action_names
    assert restored.n_keeper_candidates == case.policy.n_keeper_candidates
    assert restored.replay_capability is case.policy.replay_capability
    assert jax.tree_util.tree_structure(restored) == producer_structure
    assert all(
        actual is original
        for actual, (_, original) in zip(
            jax.tree_util.tree_leaves(restored), producer_leaves, strict=True
        )
    )
    materialized = materialize_core_program(
        program=call.program,
        context=_build_context(
            arguments={**call.arguments, "payload": wrapped}, period=0
        ),
    )
    reads = tuple(
        read
        for read in materialized.requirements.value_reads
        if read.target.kind is ValueArtifactKind.REPLAY_ARTIFACT_LEAF
    )
    expected = {
        tuple(f"{type(step).__name__}:{step}" for step in path): value
        for path, value in producer_leaves
    }
    assert len(reads) == len(expected)
    assert {read.target.leaf_path for read in reads} == set(expected)
    for read in reads:
        assert read.target.period == 0
        assert read.target.regime == "alive"
        assert read.target.artifact_key == SIMULATION_POLICY
        assert read.source.source_period == 0
        assert read.source.source_regime == "alive"
        assert read.source.core_key == call.program.name
        assert read.source.channel is ValueInputChannel.CURRENT_REPLAY_ARTIFACT
        assert (
            _value_read_argument_leaf(program=materialized, read=read)
            is expected[read.target.leaf_path]
        )


def test_real_producer_leaves_match_every_declared_locator(
    produced_case: _ProducedCase,
) -> None:
    """Both stages name every actual producer leaf without converting its metadata."""
    for call in (produced_case.preparation, produced_case.ranking):
        _assert_correspondence(case=produced_case, call=call)


def test_correspondence_rejects_a_same_shaped_source_leaf_swap(
    produced_case: _ProducedCase,
) -> None:
    """A wrong logical binding cannot hide behind identical dtype and shape."""
    for call in (produced_case.preparation, produced_case.ranking):
        reads = call.program.requirements.value_reads
        first, second = [
            i
            for i, read in enumerate(reads)
            if read.target.kind is ValueArtifactKind.REPLAY_ARTIFACT_LEAF
        ][:2]
        changed_reads = tuple(
            replace(read, source=reads[second].source)
            if i == first
            else replace(read, source=reads[first].source)
            if i == second
            else read
            for i, read in enumerate(reads)
        )
        changed = replace(
            call,
            program=replace(
                call.program,
                requirements=replace(
                    call.program.requirements, value_reads=changed_reads
                ),
            ),
        )
        with pytest.raises(AssertionError):
            _assert_correspondence(case=produced_case, call=changed)


def _dispatch_same_executable(
    *, case: _ProducedCase, call: _Call, arguments: Mapping[str, object]
) -> object:
    original = case.runtime.prepare(
        program=call.program,
        arguments=call.arguments,
        period=0,
        n_subjects=call.n_subjects,
    )
    replacement = case.runtime.prepare(
        program=call.program,
        arguments=arguments,
        period=0,
        n_subjects=call.n_subjects,
    )
    assert replacement is original
    return case.runtime.dispatch(
        program=call.program,
        arguments=arguments,
        period=0,
        n_subjects=call.n_subjects,
    )


def test_replacement_payload_changes_cached_prepare_and_rank(
    produced_case: _ProducedCase,
) -> None:
    """One compiled pair reads a replacement bank, then the original bank again."""
    case = produced_case
    original_payload = cast("ReplayPayload", case.preparation.arguments["payload"])
    replacement = ReplayPayload.from_policy(
        replace(
            original_payload.restore(),
            candidate_value=jnp.full_like(case.policy.candidate_value, jnp.nan),
        )
    )
    assert replacement.structure == original_payload.structure
    bank = cast(
        "CandidateBank",
        _dispatch_same_executable(
            case=case,
            call=case.preparation,
            arguments={**case.preparation.arguments, "payload": replacement},
        ),
    )
    original_bank = cast("CandidateBank", case.preparation.result)
    assert np.any(np.asarray(original_bank[2]))
    np.testing.assert_array_equal(bank[0], original_bank[0])
    np.testing.assert_array_equal(bank[1], original_bank[1])
    assert not np.any(np.asarray(bank[2]))
    assert not np.any(np.asarray(bank[3]))
    ranked = cast(
        "RankedBank",
        _dispatch_same_executable(
            case=case,
            call=case.ranking,
            arguments={**case.ranking.arguments, "payload": replacement, "bank": bank},
        ),
    )
    assert np.all(np.isneginf(np.asarray(ranked[1])))
    for name, action in ranked[0].items():
        if name in case.policy.discrete_action_names:
            np.testing.assert_array_equal(action, -jnp.ones_like(action))
        else:
            assert np.all(np.isnan(np.asarray(action)))
    np.testing.assert_array_equal(
        ranked[2], np.zeros(case.ranking.n_subjects, dtype=bool)
    )
    for call in (case.preparation, case.ranking):
        restored = _dispatch_same_executable(
            case=case, call=call, arguments=call.arguments
        )
        assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
            call.result
        )
        for actual, expected in zip(
            jax.tree_util.tree_leaves(restored),
            jax.tree_util.tree_leaves(call.result),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "produced_case",
    [(True, 0), (True, 1)],
    indirect=True,
    ids=("discrete-row0", "discrete-row1"),
)
def test_replacement_discrete_codes_remain_dynamic_in_cached_rank(
    produced_case: _ProducedCase,
) -> None:
    """The optional fifth payload leaf is consumed by the same compiled ranker."""
    case = produced_case
    assert case.policy.discrete_action_names
    call = case.ranking
    original = cast("RankedBank", call.result)
    (name,) = case.policy.discrete_action_names
    replacement_code = 1 - int(original[0][name][0])
    payload = cast("ReplayPayload", call.arguments["payload"])
    codes = payload.restore().candidate_discrete_actions
    assert codes is not None
    replacement = ReplayPayload.from_policy(
        replace(
            payload.restore(),
            candidate_discrete_actions=jnp.full_like(codes, replacement_code),
        )
    )
    ranked = cast(
        "RankedBank",
        _dispatch_same_executable(
            case=case, call=call, arguments={**call.arguments, "payload": replacement}
        ),
    )
    assert np.all(np.isfinite(np.asarray(ranked[1])))
    np.testing.assert_array_equal(
        ranked[0][name], np.full(call.n_subjects, replacement_code)
    )
    assert int(ranked[0][name][0]) != int(original[0][name][0])


def _tied_q_and_f(
    *, consumption: jax.Array, **arguments: object
) -> tuple[jax.Array, jax.Array]:
    del arguments
    return jnp.ones_like(consumption), jnp.ones_like(consumption, dtype=bool)


def test_actual_rank_program_keeps_first_tie_and_ignores_dropped_candidates(
    produced_case: _ProducedCase,
) -> None:
    """The real tiled rank body chooses the first live member of an exact Q tie."""
    case = produced_case
    wrapper = case.ranking.program.function
    assert isinstance(wrapper, _SubjectTiled)
    body = wrapper.func
    assert isinstance(body, _Rank)
    regime = replace(
        body.regime,
        simulation=replace(
            body.regime.simulation,
            Q_and_F=MappingProxyType({0: _tied_q_and_f}),
        ),
    )
    call = replace(
        case.ranking,
        program=replace(
            case.ranking.program,
            function=replace(wrapper, func=replace(body, regime=regime)),
        ),
    )
    original_bank = cast("CandidateBank", case.preparation.result)
    next_candidate = case.policy.n_keeper_candidates
    assert np.all(
        np.asarray(original_bank[1][:, 0])
        != np.asarray(original_bank[1][:, next_candidate])
    )
    for first_live in (True, False):
        represented = jnp.zeros_like(original_bank[3]).at[:, next_candidate].set(True)
        represented = represented.at[:, 0].set(first_live)
        bank = (
            jnp.ones_like(original_bank[0]),
            original_bank[1],
            represented,
            represented,
        )
        ranked = cast(
            "RankedBank",
            _dispatch_same_executable(
                case=case, call=call, arguments={**call.arguments, "bank": bank}
            ),
        )
        winner = 0 if first_live else next_candidate
        np.testing.assert_array_equal(
            ranked[0][case.policy.inner_action_name], np.ones(call.n_subjects)
        )
        np.testing.assert_array_equal(
            ranked[0][case.policy.outer_action_name], original_bank[1][:, winner]
        )
        np.testing.assert_array_equal(ranked[1], np.ones(call.n_subjects))
        if case.policy.candidate_discrete_actions is not None:
            for index, name in enumerate(case.policy.discrete_action_names):
                np.testing.assert_array_equal(
                    ranked[0][name],
                    jnp.broadcast_to(
                        case.policy.candidate_discrete_actions[winner, index],
                        (call.n_subjects,),
                    ),
                )
