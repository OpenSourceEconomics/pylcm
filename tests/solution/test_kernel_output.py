"""The public kernel-output envelope and its legacy bridge."""

import ast
import inspect
import itertools
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.solution import egm as egm_module
from _lcm.solution.contract import KernelResult
from _lcm.solution.kernel_output import (
    normalize_kernel_output,
    require_legacy_kernel_result,
)
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from lcm.solver_api import (
    EGM_CONTINUATION,
    ArtifactKey,
    KernelOutput,
    ResultRetention,
)
from lcm.solvers import EGM
from tests.solution.test_egm_solver import _SAVINGS_GRID, _model, _params


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
    assert "diagnostics" not in KernelOutput.__dataclass_fields__
    with pytest.raises(TypeError):
        output.continuations[key] = "changed"  # ty: ignore[invalid-assignment]

    numpy_output = KernelOutput(value=np.asarray([2.0], dtype=np.float32))
    assert isinstance(numpy_output.value, np.ndarray)


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


def test_bridge_extracts_the_declared_continuation() -> None:
    carry = _carry()
    output = KernelOutput(
        value=jnp.asarray([3.0]),
        continuations={EGM_CONTINUATION: carry},
    )

    result = normalize_kernel_output(
        output=output,
        continuation_key=EGM_CONTINUATION,
        regime_name="saving",
        period=2,
    )

    assert result.V_arr is output.value
    assert result.continuation is carry
    assert result.simulation_policy is None
    assert result.dissolution is None
    assert result.diagnostics is None


def test_bridge_normalizes_an_accepted_numpy_value_to_jax() -> None:
    output = KernelOutput(
        value=np.asarray([3.0], dtype=np.float32),
        continuations={EGM_CONTINUATION: _carry()},
    )

    result = normalize_kernel_output(
        output=output,
        continuation_key=EGM_CONTINUATION,
        regime_name="saving",
        period=2,
    )

    assert isinstance(result.V_arr, jax.Array)
    np.testing.assert_array_equal(result.V_arr, output.value)


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
)
def test_bridge_fails_closed_before_roll_with_cell_coordinates(
    *, output: KernelOutput, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        normalize_kernel_output(
            output=output,
            continuation_key=EGM_CONTINUATION,
            regime_name="saving",
            period=2,
        )


@pytest.mark.parametrize(
    "field",
    ["solve_time_artifacts", "replay", "auxiliary"],
)
def test_bridge_rejects_every_unconsumed_artifact_channel(field: str) -> None:
    key = ArtifactKey(type_id=f"example.{field}")
    output = KernelOutput(
        value=jnp.asarray([1.0]),
        continuations={EGM_CONTINUATION: _carry()},
        **{field: {key: object()}},
    )

    with pytest.raises(
        RuntimeError,
        match=f"Regime 'saving'.*period 2.*unconsumed.*example.{field}",
    ):
        normalize_kernel_output(
            output=output,
            continuation_key=EGM_CONTINUATION,
            regime_name="saving",
            period=2,
        )


def test_bridge_passes_legacy_result_and_diagnostics_through_by_identity() -> None:
    diagnostics = _diagnostics()
    legacy = KernelResult(V_arr=jnp.asarray([1.0]), diagnostics=diagnostics)

    result = normalize_kernel_output(
        output=legacy,
        continuation_key=None,
        regime_name="legacy",
        period=0,
    )

    assert result is legacy
    assert result.diagnostics is diagnostics


def test_legacy_composite_bridge_passes_kernel_result_through_by_identity() -> None:
    legacy = KernelResult(V_arr=jnp.asarray([1.0]))

    result = require_legacy_kernel_result(
        output=legacy,
        consumer="test composite",
    )

    assert result is legacy


def test_legacy_composite_bridge_refuses_a_migrated_child() -> None:
    output = KernelOutput(value=jnp.asarray([1.0]))

    with pytest.raises(
        RuntimeError,
        match=r"test composite cannot yet consume KernelOutput.*migrate this composite",
    ):
        require_legacy_kernel_result(
            output=output,
            consumer="test composite",
        )


def test_bridge_refuses_wrong_payload_under_the_expected_continuation_key() -> None:
    output = KernelOutput(
        value=jnp.asarray([1.0]),
        continuations={EGM_CONTINUATION: object()},
    )

    with pytest.raises(
        RuntimeError,
        match=r"pylcm.egm.continuation.*unsupported payload type object.*EGMCarry",
    ):
        normalize_kernel_output(
            output=output,
            continuation_key=EGM_CONTINUATION,
            regime_name="saving",
            period=2,
        )


def test_values_only_result_does_not_suppress_solve_time_continuation() -> None:
    model = _model(solver=EGM(savings_grid=_SAVINGS_GRID))

    result = model.solve_result(
        params=_params(),
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    assert set(result.values) == set(range(model.n_periods))
    assert all(result.values[period] for period in result.values)
    assert not result.retained_continuations
