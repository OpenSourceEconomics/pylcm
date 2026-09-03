"""The NB-EGM replay harness can inspect one fused compiler lifetime.

This is deliberately a replay experiment, not a production-kernel change.  The
fused callable receives the captured economic inputs directly, constructs the
continuation stacks inside the same compiled boundary as the envelope, and exposes
compiler memory without running the resulting executable.
"""

import inspect
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.solution import period_replay
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.solution.period_capture import _PAYLOAD_NAME
from tests.conftest import invariance_tolerances
from tests.solution.test_nbegm_ride_along_split_compile import _ride_along_kernel
from tests.test_models import nbegm_jump_ride_along_toy, nbegm_ride_along_toy


def _assert_same_fused_result(*, actual: object, expected: object) -> None:
    """Check exact structure and working-dtype numerical invariance separately."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        actual_arr = np.asarray(actual_leaf)
        expected_arr = np.asarray(expected_leaf)
        assert actual_arr.shape == expected_arr.shape
        assert actual_arr.dtype == expected_arr.dtype
        if np.issubdtype(expected_arr.dtype, np.floating):
            rtol, atol = invariance_tolerances(expected_arr)
            np.testing.assert_allclose(
                actual_arr,
                expected_arr,
                rtol=rtol,
                atol=atol,
            )
        else:
            np.testing.assert_array_equal(actual_arr, expected_arr)


def _toy_ride_along_kernel(
    *, cliff_candidates: bool = False
) -> _RideAlongNBEGMPeriodKernel:
    """Build a tiny two-core adapter with the real ride-along boundary."""

    def continuation_core(
        *,
        next_regime_to_continuation,
        next_regime_to_V_arr,
        liquid,
        kind,
        slope,
        period,
        age,
    ):
        child = next_regime_to_continuation["child"].value
        child_value = next_regime_to_V_arr["child"]
        base = slope * liquid + kind + child + child_value + period + age
        stacks = (base, 2 * base)
        return (*stacks, base - 1) if cliff_candidates else stacks

    def envelope_core(
        *,
        cont_value_stack,
        cont_marginal_stack,
        liquid,
        kind,
        slope,
        period,
        age,
        cliff_savings_stack=None,
    ):
        value = cont_value_stack + cont_marginal_stack + liquid + kind
        if cliff_savings_stack is not None:
            value = value + 3 * cliff_savings_stack
        carry = cont_value_stack - slope
        action = cont_marginal_stack + period + age
        return value, carry, action

    # Only the adapter behavior is under test; constructing the solver's large
    # schedule statics would make this a model-building test. Populate the frozen
    # dataclass directly with the two static fields the fused boundary reads.
    kernel = object.__new__(_RideAlongNBEGMPeriodKernel)
    fields = {
        "continuation_core": continuation_core,
        "envelope_core": envelope_core,
        "statics": SimpleNamespace(
            state_names=("liquid", "kind"),
            n_action_branches=0,
        ),
        "cliff_candidates": cliff_candidates,
        "regime_name": "parent",
        "stateful_targets": frozenset({"child"}),
        "transition_target_names": ("child",),
    }
    for name, value in fields.items():
        object.__setattr__(kernel, name, value)
    return kernel


def _fused_inputs():
    child_value = jnp.array([3.0, 4.0])
    return {
        "next_regime_to_continuation": {
            "child": EGMCarry(
                endog_grid=jnp.array([0.0, 1.0]),
                value=child_value,
                marginal_utility=jnp.ones_like(child_value),
                taste_shock_scale=jnp.array(0.0),
            )
        },
        "next_regime_to_V_arr": {"child": jnp.array([0.5, 1.5])},
        "liquid": jnp.array([1.0, 2.0]),
        "kind": jnp.array([0.0, 1.0]),
        "slope": jnp.array(0.25),
        "period": jnp.int32(2),
        "age": jnp.array(42.0),
    }


def test_fused_ride_along_core_matches_the_current_split() -> None:
    """Fusion changes the lifetime boundary, not the numerical calculation."""
    kernel = _toy_ride_along_kernel()
    inputs = _fused_inputs()

    continuation = kernel.continuation_core(**inputs)
    expected = kernel.envelope_core(
        **{
            key: value
            for key, value in inputs.items()
            if not key.startswith("next_regime_to_")
        },
        cont_value_stack=continuation[0],
        cont_marginal_stack=continuation[1],
    )

    actual = jax.jit(kernel.build_fused_replay_core())(**inputs)

    jax.tree.map(
        lambda left, right: np.testing.assert_array_equal(
            np.asarray(left), np.asarray(right)
        ),
        actual,
        expected,
    )


def test_fused_cliff_candidate_core_matches_three_stack_split() -> None:
    """The replay-only fusion threads the optional cliff-savings stack exactly."""
    kernel = _toy_ride_along_kernel(cliff_candidates=True)
    inputs = _fused_inputs()

    continuation = kernel.continuation_core(**inputs)
    expected = kernel.envelope_core(
        **{
            key: value
            for key, value in inputs.items()
            if not key.startswith("next_regime_to_")
        },
        cont_value_stack=continuation[0],
        cont_marginal_stack=continuation[1],
        cliff_savings_stack=continuation[2],
    )

    fused = kernel.build_fused_replay_core()
    names = inspect.signature(fused).parameters
    actual = jax.jit(fused)(**inputs)

    jax.tree.map(
        lambda left, right: np.testing.assert_array_equal(
            np.asarray(left), np.asarray(right)
        ),
        actual,
        expected,
    )
    assert "cont_value_stack" not in names
    assert "cont_marginal_stack" not in names
    assert "cliff_savings_stack" not in names


def test_real_ride_along_fused_core_matches_the_current_split() -> None:
    """The generated NB-EGM cores retain their result under diagnostic fusion."""
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_periods=3, n_liquid=12, n_savings=16
    )
    kernel, context = _ride_along_kernel(
        model=model, params=nbegm_ride_along_toy.build_params()
    )
    continuation_args = kernel.build_lower_args(core_key="continuation", **context)
    continuation_stacks = jax.jit(kernel.continuation_core)(**continuation_args)
    envelope_args = dict(kernel.build_lower_args(core_key="envelope", **context))
    envelope_args["cont_value_stack"] = continuation_stacks[0]
    envelope_args["cont_marginal_stack"] = continuation_stacks[1]

    expected = jax.jit(kernel.envelope_core)(**envelope_args)
    expected_envelope_inputs = frozenset(envelope_args)
    observed_envelope_inputs = []

    def strict_envelope_core(**kwargs):
        observed_envelope_inputs.append(frozenset(kwargs))
        assert frozenset(kwargs) == expected_envelope_inputs
        return kernel.envelope_core(**kwargs)

    guarded_kernel = replace(kernel, envelope_core=strict_envelope_core)
    actual = jax.jit(guarded_kernel.build_fused_replay_core())(**continuation_args)

    _assert_same_fused_result(actual=actual, expected=expected)
    assert observed_envelope_inputs


def test_real_cliff_candidate_fused_core_matches_the_current_split() -> None:
    """A generated jump schedule keeps all three continuation stacks internal."""
    model = nbegm_jump_ride_along_toy.build_model(
        variant="nbegm", n_periods=4, n_liquid=12, n_savings=16
    )
    kernel, context = _ride_along_kernel(
        model=model, params=nbegm_jump_ride_along_toy.build_params()
    )
    assert kernel.cliff_candidates

    continuation_args = kernel.build_lower_args(core_key="continuation", **context)
    continuation_stacks = jax.jit(kernel.continuation_core)(**continuation_args)
    assert len(continuation_stacks) == 3
    envelope_args = dict(kernel.build_lower_args(core_key="envelope", **context))
    envelope_args["cont_value_stack"] = continuation_stacks[0]
    envelope_args["cont_marginal_stack"] = continuation_stacks[1]
    envelope_args["cliff_savings_stack"] = continuation_stacks[2]

    expected = jax.jit(kernel.envelope_core)(**envelope_args)
    fused = kernel.build_fused_replay_core()
    names = inspect.signature(fused).parameters
    actual = jax.jit(fused)(**continuation_args)

    _assert_same_fused_result(actual=actual, expected=expected)
    assert "cont_value_stack" not in names
    assert "cont_marginal_stack" not in names
    assert "cliff_savings_stack" not in names


def test_fused_core_signature_has_no_materialized_continuation_stacks() -> None:
    """Continuation stacks are compiler-local values, never fused-core inputs."""
    fused = _toy_ride_along_kernel().build_fused_replay_core()
    names = inspect.signature(fused).parameters

    assert "next_regime_to_continuation" in names
    assert "next_regime_to_V_arr" in names
    assert "cont_value_stack" not in names
    assert "cont_marginal_stack" not in names
    assert "cliff_savings_stack" not in names


def test_fused_memory_analyzer_compiles_but_never_executes(
    *, monkeypatch, tmp_path: Path
) -> None:
    """A captured-shape probe may compile an over-budget core but must not run it."""

    class CompileOnlyExecutable:
        def __init__(self, *, offset: int) -> None:
            self.executed = False
            self.stats = SimpleNamespace(
                generated_code_size_in_bytes=offset + 1,
                argument_size_in_bytes=offset + 2,
                output_size_in_bytes=offset + 3,
                alias_size_in_bytes=offset + 4,
                temp_size_in_bytes=offset + 5,
                peak_memory_in_bytes=offset + 6,
                host_generated_code_size_in_bytes=offset + 7,
                host_argument_size_in_bytes=offset + 8,
                host_output_size_in_bytes=offset + 9,
                host_alias_size_in_bytes=offset + 10,
                host_temp_size_in_bytes=None,
            )

        def memory_analysis(self):
            return self.stats

        def __call__(self, **_kwargs):
            self.executed = True
            raise AssertionError("the memory analyzer executed the compiled core")

    fused_executable = CompileOnlyExecutable(offset=100)
    split_executables = {
        "continuation": CompileOnlyExecutable(offset=200),
        "envelope": CompileOnlyExecutable(offset=300),
    }
    fused_compile_calls = []
    split_compile_calls = []

    def fake_fused_compile(*, regime, period, kernel_kwargs):
        fused_compile_calls.append((regime, period, kernel_kwargs))
        return fused_executable

    def fake_split_compile(*, regime, period, kernel_kwargs):
        split_compile_calls.append((regime, period, kernel_kwargs))
        return split_executables

    monkeypatch.setattr(
        period_replay,
        "_compile_fused_nbegm_core_for_one_period",
        fake_fused_compile,
    )
    monkeypatch.setattr(
        period_replay,
        "_compile_cores_for_one_period",
        fake_split_compile,
    )
    payload = {
        "regime": object(),
        "period": 1,
        "kernel_kwargs": {
            "regime_name": "parent",
            "ages": SimpleNamespace(values=jnp.array([40.0, 41.0])),
        },
    }
    with (tmp_path / _PAYLOAD_NAME).open("wb") as stream:
        cloudpickle.dump(payload, stream)

    analysis = period_replay.analyze_fused_nbegm_memory(directory=tmp_path)

    assert len(fused_compile_calls) == len(split_compile_calls) == 1
    assert analysis.fused_memory_bytes is not None
    assert asdict(analysis.fused_memory_bytes) == vars(fused_executable.stats)
    split_memory_bytes = {}
    for name, report in analysis.split_memory_bytes.items():
        assert report is not None
        split_memory_bytes[name] = asdict(report)
    assert split_memory_bytes == {
        name: vars(executable.stats) for name, executable in split_executables.items()
    }
    assert (analysis.regime_name, analysis.period, analysis.age) == (
        "parent",
        1,
        41.0,
    )
    assert analysis.experiment_scope == "existing_full_stack_one_jit_lifetime"
    assert analysis.input_shape_provenance == "captured_production_shapes"
    assert analysis.input_layout_fidelity == "default_backend_after_capture_roundtrip"
    assert analysis.preserves_production_sharding is False
    assert not fused_executable.executed
    assert all(not executable.executed for executable in split_executables.values())


def test_compiler_memory_bytes_normalizes_real_and_unsupported_backends() -> None:
    """Backend-specific stats become integer-or-None fields, or no report."""
    compiled = jax.jit(lambda values: values + 1).lower(jnp.ones(2)).compile()

    report = period_replay._compiler_memory_bytes(compiled=compiled)

    raw = compiled.memory_analysis()
    if raw is None:
        assert report is None
    else:
        assert report is not None
        assert all(
            value is None or isinstance(value, int) for value in asdict(report).values()
        )

    class UnsupportedExecutable:
        def __init__(self, *, raises: bool) -> None:
            self.raises = raises

        def memory_analysis(self):
            if self.raises:
                raise RuntimeError("backend has no memory analysis")

    assert (
        period_replay._compiler_memory_bytes(
            compiled=UnsupportedExecutable(raises=False)
        )
        is None
    )
    assert (
        period_replay._compiler_memory_bytes(
            compiled=UnsupportedExecutable(raises=True)
        )
        is None
    )
