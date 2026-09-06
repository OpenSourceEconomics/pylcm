"""The public kernel-output envelope and the solve loop's consumer of it.

Every period kernel returns a `KernelOutput`: a value plus four keyed artifact
channels. The solve loop reads each channel by the artifact's declared key
through one consumer, and refuses, naming the regime and period, anything it
has no reader for: an unknown key, a known key with a payload of another type,
a missing required continuation, or a continuation published under another
schema version. No second result type exists between a kernel and the loop.
"""

import ast
import dataclasses
import inspect
import itertools
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, get_args, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float

from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.solution import backward_induction, period_replay
from _lcm.solution import egm as egm_module
from _lcm.solution.contract import (
    GENERATED_REPLAY_AUTHORITY,
    GeneratedReplayAuthority,
    PeriodKernel,
)
from _lcm.solution.kernel_output import ConsumedKernelOutput, consume_kernel_output
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import RegimeName
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactKey,
    KernelOutput,
    ResultRetention,
)
from lcm.solvers import EGM
from lcm.typing import FloatND
from tests.solution.test_dissolution_flag_retention import _make_dissolution_model
from tests.solution.test_egm_solver import _SAVINGS_GRID, _model, _params
from tests.solution.test_n_nbegm_fixed_cost import _MESH
from tests.test_models import n_nbegm_toy, nbegm_ride_along_toy


def _carry() -> EGMCarry:
    row = jnp.asarray([[0.0, 1.0]])
    return EGMCarry(
        endog_grid=row,
        value=row,
        marginal_utility=jnp.ones_like(row),
        taste_shock_scale=jnp.asarray(0.0),
    )


def _diagnostics() -> SolverDiagnostics:
    scalar = jnp.asarray(0.0)
    flag = jnp.zeros((), dtype=jnp.bool_)
    return SolverDiagnostics(
        max_outer_interpolation_error=scalar,
        max_outer_bracket_width=scalar,
        outer_nodes_used=jnp.asarray(1, dtype=jnp.int32),
        outer_at_lower_bound=flag,
        outer_at_upper_bound=flag,
        keeper_adjuster_margin=scalar,
        best_second_best_margin=scalar,
        policy_fallback_mask=flag,
        unresolved_mask=flag,
        n_outer_all_invalid_cells=jnp.asarray(0, dtype=jnp.int32),
    )


def _policy() -> EGMSimPolicy:
    return EGMSimPolicy(
        endog_grid=jnp.asarray([[0.0, 1.0]]),
        policy=jnp.asarray([[0.5, 0.5]]),
        value=jnp.asarray([[0.0, 1.0]]),
        marginal_utility=jnp.asarray([[1.0, 1.0]]),
    )


def _consume(
    *, output: object, continuation_key: ArtifactKey | None = EGM_CONTINUATION
) -> ConsumedKernelOutput:
    return consume_kernel_output(
        output=output,
        continuation_key=continuation_key,
        regime_name="saving",
        period=2,
    )


def _run_unwrapped_kernel_output_post_init(*, value: object) -> None:
    """Exercise pylcm's own guard without the package-claw type wrapper."""
    output = object.__new__(KernelOutput)
    object.__setattr__(output, "value", value)
    for field_name in (
        "continuations",
        "solve_time_artifacts",
        "replay",
        "auxiliary",
    ):
        object.__setattr__(output, field_name, {})
    inspect.unwrap(KernelOutput.__post_init__)(output)


def test_kernel_output_is_public_dependency_safe_and_defensively_immutable() -> None:
    """An installed solver can publish artifacts without importing engine code."""
    module = inspect.getmodule(KernelOutput)
    assert module is not None
    tree = ast.parse(inspect.getsource(module))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(name == "_lcm" or name.startswith("_lcm.") for name in imports)

    key = ArtifactKey(type_id="example.continuation")
    source = {key: "payload"}
    output = KernelOutput(value=jnp.asarray([1.0]), continuations=source)
    source.clear()

    assert output.continuations == {key: "payload"}
    assert isinstance(output.continuations, MappingProxyType)
    assert isinstance(output.solve_time_artifacts, MappingProxyType)
    assert isinstance(output.replay, MappingProxyType)
    assert isinstance(output.auxiliary, MappingProxyType)
    with pytest.raises(TypeError):
        output.continuations[key] = "changed"  # ty: ignore[invalid-assignment]

    numpy_output = KernelOutput(value=np.asarray([2.0], dtype=np.float32))
    assert isinstance(numpy_output.value, np.ndarray)


def test_kernel_output_value_annotation_matches_runtime_float_contract() -> None:
    """The public hint excludes array dtypes that construction rejects."""
    annotation = get_type_hints(KernelOutput)["value"]

    expected = {repr(FloatND), repr(Float[np.ndarray, "*shape"])}
    assert {repr(member) for member in get_args(annotation)} == expected


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0],
        np.asarray([1, 2], dtype=np.int32),
        jnp.asarray([True, False]),
        np.asarray([1 + 2j], dtype=np.complex64),
    ],
    ids=["python-sequence", "integer", "boolean", "complex"],
)
def test_kernel_output_rejects_non_float_array_values(value: object) -> None:
    """A kernel value is one floating NumPy/JAX array leaf, not an implicit cast."""
    with pytest.raises(TypeError, match=r"KernelOutput\.value.*floating.*array"):
        _run_unwrapped_kernel_output_post_init(value=value)


@pytest.mark.parametrize(
    ("left_channel", "right_channel"),
    tuple(
        itertools.combinations(
            ("continuations", "solve_time_artifacts", "replay", "auxiliary"), 2
        )
    ),
)
def test_kernel_output_rejects_one_artifact_identity_in_multiple_channels(
    *,
    left_channel: str,
    right_channel: str,
) -> None:
    """One artifact identity has exactly one semantic retention channel."""
    key = ArtifactKey(type_id="example.ambiguous")

    with pytest.raises(
        ValueError,
        match=rf"example\.ambiguous.*{left_channel}.*{right_channel}",
    ):
        KernelOutput(
            value=jnp.asarray([1.0]),
            **{
                left_channel: {key: "left"},
                right_channel: {key: "right"},
            },
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"type_id": 7}, "type_id.*str"),
        ({"type_id": "example", "schema_version": True}, "schema_version.*int"),
        ({"type_id": "example", "schema_version": "1"}, "schema_version.*int"),
    ],
)
def test_artifact_key_rejects_runtime_types_that_cannot_name_a_schema(
    *, kwargs: dict[str, object], match: str
) -> None:
    key = object.__new__(ArtifactKey)
    object.__setattr__(key, "type_id", kwargs["type_id"])
    object.__setattr__(key, "schema_version", kwargs.get("schema_version", 1))

    with pytest.raises(TypeError, match=match):
        inspect.unwrap(ArtifactKey.__post_init__)(key)


def test_egm_continuation_spec_declares_the_public_artifact_identity() -> None:
    spec = EGMContinuationSpec(template=_carry())

    assert spec.artifact_key is EGM_CONTINUATION


def test_raw_egm_kernel_publishes_the_exact_public_continuation_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[KernelOutput] = []
    original = egm_module._EGMPeriodKernel.__call__

    def recording_call(kernel: object, **kwargs: object) -> KernelOutput:
        output = original(kernel, **kwargs)  # ty: ignore[invalid-argument-type]
        seen.append(output)
        return output

    monkeypatch.setattr(egm_module._EGMPeriodKernel, "__call__", recording_call)

    _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=_params(), log_level="off"
    )

    assert seen
    assert all(tuple(output.continuations) == (EGM_CONTINUATION,) for output in seen)
    assert all(not output.replay for output in seen)


def test_the_consumer_reads_the_declared_continuation() -> None:
    carry = _carry()
    output = KernelOutput(
        value=jnp.asarray([3.0]),
        continuations={EGM_CONTINUATION: carry},
    )

    consumed = _consume(output=output)

    assert consumed.value is output.value
    assert consumed.continuation is carry
    assert consumed.simulation_policy is None
    assert consumed.generated_replay_authority is None
    assert consumed.dissolution is None
    assert consumed.diagnostics is None


def test_the_consumer_normalizes_an_accepted_numpy_value_to_jax() -> None:
    output = KernelOutput(
        value=np.asarray([3.0], dtype=np.float32),
        continuations={EGM_CONTINUATION: _carry()},
    )

    consumed = _consume(output=output)

    assert isinstance(consumed.value, jax.Array)
    np.testing.assert_array_equal(consumed.value, output.value)


@pytest.mark.parametrize(
    ("output", "match"),
    [
        (
            KernelOutput(value=jnp.asarray([1.0])),
            "Regime 'saving'.*period 2.*missing.*pylcm.egm.continuation",
        ),
        (
            KernelOutput(
                value=jnp.asarray([1.0]),
                continuations={
                    ArtifactKey(type_id="example.unknown", schema_version=1): object()
                },
            ),
            "Regime 'saving'.*period 2.*unconsumed.*example.unknown",
        ),
        (
            KernelOutput(
                value=jnp.asarray([1.0]),
                continuations={
                    ArtifactKey(
                        type_id=EGM_CONTINUATION.type_id,
                        schema_version=2,
                    ): object()
                },
            ),
            "Regime 'saving'.*period 2.*version.*2.*expected.*1",
        ),
    ],
    ids=["missing", "unknown-key", "other-version"],
)
def test_the_consumer_fails_closed_before_the_roll_with_cell_coordinates(
    *, output: KernelOutput, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _consume(output=output)


@pytest.mark.parametrize(
    "channel",
    ["solve_time_artifacts", "replay", "auxiliary"],
)
def test_the_consumer_refuses_every_unconsumed_artifact_channel(channel: str) -> None:
    key = ArtifactKey(type_id=f"example.{channel}")
    output = KernelOutput(
        value=jnp.asarray([1.0]),
        continuations={EGM_CONTINUATION: _carry()},
        **{channel: {key: object()}},
    )

    with pytest.raises(
        RuntimeError,
        match=f"Regime 'saving'.*period 2.*unconsumed.*example.{channel}",
    ):
        _consume(output=output)


def test_the_consumer_refuses_a_wrong_payload_under_the_continuation_key() -> None:
    output = KernelOutput(
        value=jnp.asarray([1.0]),
        continuations={EGM_CONTINUATION: object()},
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"pylcm.egm.continuation.*unsupported payload type object"
            r".*ContinuationArtifact"
        ),
    ):
        _consume(output=output)


def test_the_consumer_reads_the_policy_flag_diagnostics_and_generated_authority() -> (
    None
):
    policy = _policy()
    flag = jnp.asarray([True, False])
    diagnostics = _diagnostics()
    authority = GeneratedReplayAuthority(adaptive_outer_nodes=(0.0, 1.0))
    output = KernelOutput(
        value=jnp.asarray([1.0, 2.0]),
        solve_time_artifacts={DISSOLUTION_FLAG: flag},
        replay={SIMULATION_POLICY: policy},
        auxiliary={
            SOLVER_DIAGNOSTICS: diagnostics,
            GENERATED_REPLAY_AUTHORITY: authority,
        },
    )

    consumed = _consume(output=output, continuation_key=None)

    assert consumed.simulation_policy is policy
    assert consumed.dissolution is flag
    assert consumed.diagnostics is diagnostics
    assert consumed.generated_replay_authority is authority


def test_the_consumer_requires_a_bool_dissolution_flag() -> None:
    output = KernelOutput(
        value=jnp.asarray([1.0, 2.0]),
        solve_time_artifacts={DISSOLUTION_FLAG: jnp.asarray([1.0, 0.0])},
    )

    with pytest.raises(RuntimeError, match=r"dissolution_flag.*dtype.*expected bool"):
        _consume(output=output, continuation_key=None)


def test_the_consumer_refuses_a_generated_authority_without_a_policy() -> None:
    output = KernelOutput(
        value=jnp.asarray([1.0]),
        auxiliary={
            GENERATED_REPLAY_AUTHORITY: GeneratedReplayAuthority(
                adaptive_outer_nodes=(0.0,)
            )
        },
    )

    with pytest.raises(RuntimeError, match=r"'saving'.*period 2.*no matching.*policy"):
        _consume(output=output, continuation_key=None)


@pytest.mark.parametrize(
    ("channel", "key"),
    [
        ("replay", SIMULATION_POLICY),
        ("solve_time_artifacts", DISSOLUTION_FLAG),
        ("auxiliary", SOLVER_DIAGNOSTICS),
        ("auxiliary", GENERATED_REPLAY_AUTHORITY),
    ],
    ids=["policy", "flag", "diagnostics", "generated-authority"],
)
def test_the_consumer_refuses_a_known_key_with_the_wrong_payload_type(
    *, channel: str, key: ArtifactKey
) -> None:
    output = KernelOutput(value=jnp.asarray([1.0]), **{channel: {key: object()}})

    with pytest.raises(RuntimeError, match=f"'saving'.*period 2.*{key.type_id}"):
        _consume(output=output, continuation_key=None)


def test_the_consumer_refuses_anything_but_a_kernel_output() -> None:
    with pytest.raises(TypeError, match=r"'saving'.*period 2.*unsupported.*object"):
        _consume(output=object(), continuation_key=None)


def test_a_period_kernel_returns_a_kernel_output_and_replay_carries_it() -> None:
    """The producer contract is `KernelOutput`, and replay holds one on `output`."""
    assert get_type_hints(PeriodKernel.__call__)["return"] is KernelOutput
    assert "output" in period_replay.PeriodReplay.__dataclass_fields__


def test_the_loop_retains_policies_by_declared_route_not_by_caller_flags() -> None:
    """Replay retention has one switch; publication is read off the declarations."""
    parameters = inspect.signature(backward_induction.solve).parameters

    assert "retain_replay" in parameters


def test_values_only_result_does_not_suppress_solve_time_continuation() -> None:
    model = _model(solver=EGM(savings_grid=_SAVINGS_GRID))

    result = model.solve(
        params=_params(),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    assert set(result.values) == set(range(model.n_periods))
    assert all(result.values[period] for period in result.values)
    assert not result.retained_continuations


def _record_kernel_outputs(
    *, solve: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> dict[tuple[int, RegimeName], object]:
    """Run one solve and return what every period kernel returned, by cell."""
    recorded: dict[tuple[int, RegimeName], object] = {}
    original = backward_induction._run_period_kernel

    def recording(**kwargs: Any) -> Any:
        output = original(**kwargs)
        recorded[(kwargs["period"], kwargs["regime_name"])] = output
        return output

    monkeypatch.setattr(backward_induction, "_run_period_kernel", recording)
    solve()
    assert recorded, "no period kernel ran; the sweep is inert"
    return recorded


def _channel_keys(output: KernelOutput) -> dict[str, frozenset[ArtifactKey]]:
    return {
        channel: frozenset(getattr(output, channel))
        for channel in ("continuations", "solve_time_artifacts", "replay", "auxiliary")
    }


def _solve_collective() -> None:
    model, params = _make_dissolution_model()
    model.solve(params=params, log_level="off")


def _solve_egm() -> None:
    _model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=_params(), log_level="off"
    )


def _solve_nbegm() -> None:
    nbegm_ride_along_toy.build_model(variant="nbegm", n_periods=2).solve(
        params=nbegm_ride_along_toy.build_params(), log_level="off"
    )


def _solve_nnbegm(*, outer_search: object = None) -> None:
    n_nbegm_toy.build_model(
        variant="n_nbegm",
        n_periods=2,
        outer_search=outer_search,  # ty: ignore[invalid-argument-type]
    ).solve(params={"discount_factor": 0.95}, log_level="off")


_SHIPPED_KERNELS: dict[str, tuple[Callable[[], None], RegimeName, dict[str, set]]] = {
    "grid_search_singleton": (
        _solve_collective,
        "single_f",
        {"continuations": set(), "solve_time_artifacts": set(), "replay": set()},
    ),
    "grid_search_collective": (
        _solve_collective,
        "married",
        {"continuations": set(), "solve_time_artifacts": {DISSOLUTION_FLAG}},
    ),
    "egm": (
        _solve_egm,
        "saving",
        {"continuations": {EGM_CONTINUATION}, "replay": set()},
    ),
    "terminal_carry": (
        _solve_egm,
        "done",
        {"continuations": {EGM_CONTINUATION}, "replay": set()},
    ),
    "nbegm": (
        _solve_nbegm,
        "alive",
        {"continuations": {EGM_CONTINUATION}},
    ),
    "nnbegm_finite": (
        _solve_nnbegm,
        "alive",
        {"continuations": {EGM_CONTINUATION}, "replay": {SIMULATION_POLICY}},
    ),
    "nnbegm_adaptive": (
        lambda: _solve_nnbegm(outer_search=_MESH),
        "alive",
        {
            "continuations": {EGM_CONTINUATION},
            "replay": {SIMULATION_POLICY},
            "auxiliary": {SOLVER_DIAGNOSTICS, GENERATED_REPLAY_AUTHORITY},
        },
    ),
}


@pytest.mark.parametrize("case", list(_SHIPPED_KERNELS))
def test_every_shipped_kernel_returns_a_kernel_output_on_the_declared_channels(
    *, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each built-in kernel publishes on the channels its artifacts belong to."""
    solve, regime_name, expected_channels = _SHIPPED_KERNELS[case]
    recorded = _record_kernel_outputs(solve=solve, monkeypatch=monkeypatch)
    outputs = [output for (_, name), output in recorded.items() if name == regime_name]

    assert outputs, f"regime {regime_name!r} never ran"
    assert all(isinstance(output, KernelOutput) for output in outputs)
    for output in outputs:
        keys = _channel_keys(output)  # ty: ignore[invalid-argument-type]
        for channel, expected in expected_channels.items():
            assert keys[channel] == frozenset(expected), (case, channel)


def test_a_values_only_solve_publishes_no_replay_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = _record_kernel_outputs(
        solve=lambda: n_nbegm_toy.build_model(variant="n_nbegm", n_periods=2).solve(
            params={"discount_factor": 0.95},
            log_level="off",
            retention=ResultRetention.VALUES,
        ),
        monkeypatch=monkeypatch,
    )

    assert all(
        not output.replay  # ty: ignore[unresolved-attribute]
        for output in recorded.values()
    )


def test_a_kernel_output_survives_a_dataclass_replace_of_its_value() -> None:
    """The loop's placement seam may re-place the value without touching channels."""
    carry = _carry()
    output = KernelOutput(
        value=jnp.asarray([1.0]), continuations={EGM_CONTINUATION: carry}
    )

    replaced = dataclasses.replace(output, value=jnp.asarray([2.0]))

    assert replaced.continuations[EGM_CONTINUATION] is carry
    assert isinstance(replaced.continuations, Mapping)
