"""Budgeted simulate validation reuses its executables across calls of one Model."""

import functools

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.simulation.host_operations import ProfiledSimulationOperations, _operation_key
from _lcm.simulation.initial_conditions import _PREFLIGHT_OPERATIONS
from _lcm.solution.backward_induction import _trace_settings_key
from benchmarks.asv._compile_counters import count_compile_requests
from lcm.execution import ExecutionConfig
from lcm_examples import precautionary_savings
from tests.simulation import (
    test_joint_transition_entry_admission as joint_transition_entry,
)
from tests.simulation import (
    test_state_transition_entry_admission as state_transition_entry,
)

_N_SUBJECTS = 20


def _savings_inputs():
    """Constraint feasibility and a regime transition law."""
    model = precautionary_savings.create_model(
        n_periods=3,
        shock_type="rouwenhorst",
        wealth_grid_type="lin",
        wealth_n_points=6,
        consumption_n_points=6,
        execution_config=ExecutionConfig(
            device_memory_bytes=2 * 1024**3,
            axis_widths={"subject": _N_SUBJECTS},
        ),
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst", sigma=0.1, rho=0.5
    )
    initial_conditions = {
        "age": jnp.full(_N_SUBJECTS, 20.0),
        "wealth": jnp.linspace(1.0, 9.0, _N_SUBJECTS),
        "income": jnp.full(_N_SUBJECTS, 0.0),
        "regime_id": jnp.zeros(_N_SUBJECTS, dtype="int32"),
    }
    return model, params, initial_conditions


def _state_law_inputs():
    """A stochastic state transition law."""
    return state_transition_entry._inputs(budget=2**28)


def _joint_law_inputs():
    """A joint transition whose weights are composed per target."""
    return joint_transition_entry._inputs(
        probabilities=joint_transition_entry._joint_probabilities,
        support=joint_transition_entry._joint_support,
        budget=2**28,
    )


_INPUTS = pytest.mark.parametrize(
    "inputs", [_savings_inputs, _state_law_inputs, _joint_law_inputs]
)


def _simulate(*, model, params, initial_conditions) -> pd.DataFrame:
    return model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="warning",
        seed=0,
    ).to_dataframe(use_labels=False)


def _reciprocal(*, x: jax.Array, denominator: float) -> jax.Array:
    return x / denominator


def _integer_range(*, x: jax.Array) -> jax.Array:
    return jnp.arange(3) + x.astype(int)


def _compile(*, owner: ProfiledSimulationOperations, function, x: jax.Array):
    arguments = {"x": jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)}
    key = _operation_key(
        function=function,
        arguments=arguments,
        static_arguments={},
        subject_outputs=False,
        devices=tuple(x.devices()),
    )
    return owner.compile_candidate(
        key=key, function=function, arguments=arguments, static_arguments={}
    ).executable


@_INPUTS
def test_warm_budgeted_simulate_with_validation_compiles_nothing(inputs) -> None:
    model, params, initial_conditions = inputs()
    with count_compile_requests() as cold:
        _simulate(model=model, params=params, initial_conditions=initial_conditions)
    with count_compile_requests() as warm:
        _simulate(model=model, params=params, initial_conditions=initial_conditions)
    assert (cold.compile_requests > 0, warm.compile_requests) == (True, 0)


@_INPUTS
def test_budgeted_simulate_output_is_identical_on_warm_calls(inputs) -> None:
    model, params, initial_conditions = inputs()
    cold = _simulate(model=model, params=params, initial_conditions=initial_conditions)
    warm = _simulate(model=model, params=params, initial_conditions=initial_conditions)
    pd.testing.assert_frame_equal(warm, cold, check_exact=True)


@_INPUTS
def test_two_models_share_no_validation_executable(inputs) -> None:
    first, params, initial_conditions = inputs()
    second, _, _ = inputs()
    _simulate(model=first, params=params, initial_conditions=initial_conditions)
    _simulate(model=second, params=params, initial_conditions=initial_conditions)
    first_cache = first._simulate_entry_operations.cache
    second_cache = second._simulate_entry_operations.cache
    first_executables = {id(entry.executable) for entry in first_cache.values()}
    second_executables = {id(entry.executable) for entry in second_cache.values()}
    shared = {id(entry.executable) for entry in _PREFLIGHT_OPERATIONS.cache.values()}
    assert first_executables
    assert not first_executables & (second_executables | shared)


def test_partial_keywords_of_signed_zero_get_separate_executables() -> None:
    owner = ProfiledSimulationOperations()
    x = jnp.ones(2)
    positive = _compile(
        owner=owner, function=functools.partial(_reciprocal, denominator=0.0), x=x
    )
    negative = _compile(
        owner=owner, function=functools.partial(_reciprocal, denominator=-0.0), x=x
    )
    assert jnp.array_equal(negative(x=x), jnp.full(2, -jnp.inf))
    assert jnp.array_equal(positive(x=x), jnp.full(2, jnp.inf))


def test_same_program_traced_with_and_without_x64_gets_separate_executables() -> None:
    owner = ProfiledSimulationOperations()
    x = jnp.ones(3, dtype=jnp.float32)
    with jax.enable_x64(new_val=True):
        wide = _compile(owner=owner, function=_integer_range, x=x)
    with jax.enable_x64(new_val=False):
        narrow = _compile(owner=owner, function=_integer_range, x=x)
    assert (wide.out_info.dtype, narrow.out_info.dtype) == (jnp.int64, jnp.int32)


def _add_integer_range(*, x: jax.Array) -> jax.Array:
    return x + jnp.arange(3)


def _constrain_to_mesh_axis(*, x: jax.Array) -> jax.Array:
    return jax.lax.with_sharding_constraint(x, jax.sharding.PartitionSpec("d"))


def _single_device_mesh() -> jax.sharding.Mesh:
    return jax.make_mesh(
        (1,),
        ("d",),
        axis_types=(jax.sharding.AxisType.Auto,),
        devices=jax.devices()[:1],
    )


def _add_first_keyword(*, x: jax.Array, **offsets: float) -> jax.Array:
    return x + next(iter(offsets.values()))


def test_strict_dtype_promotion_gets_its_own_executable() -> None:
    owner = ProfiledSimulationOperations()
    x = jnp.ones(3, dtype=jnp.float32)
    _compile(owner=owner, function=_add_integer_range, x=x)
    with (
        jax.numpy_dtype_promotion("strict"),
        pytest.raises(jax.dtypes.TypePromotionError),
    ):
        _compile(owner=owner, function=_add_integer_range, x=x)


def test_lowering_outside_a_mesh_gets_its_own_executable() -> None:
    owner = ProfiledSimulationOperations()
    x = jnp.ones(3)
    with jax.set_mesh(_single_device_mesh()):
        _compile(owner=owner, function=_constrain_to_mesh_axis, x=x)
    with pytest.raises(RuntimeError):
        _compile(owner=owner, function=_constrain_to_mesh_axis, x=x)


def test_partial_keyword_order_gets_separate_executables() -> None:
    owner = ProfiledSimulationOperations()
    x = jnp.zeros(2)
    first_a = _compile(
        owner=owner,
        function=functools.partial(_add_first_keyword, a=1.0, b=2.0),
        x=x,
    )
    first_b = _compile(
        owner=owner,
        function=functools.partial(_add_first_keyword, b=2.0, a=1.0),
        x=x,
    )
    assert (float(first_a(x=x)[0]), float(first_b(x=x)[0])) == (1.0, 2.0)


@pytest.mark.parametrize(
    "setting",
    [
        lambda: jax.enable_x64(new_val=not jax.config.jax_enable_x64),
        lambda: jax.numpy_dtype_promotion("strict"),
        lambda: jax.default_matmul_precision(
            "bfloat16" if jax.config.jax_default_matmul_precision else "highest"
        ),
        lambda: jax.set_mesh(_single_device_mesh()),
    ],
    ids=["x64", "dtype_promotion", "matmul_precision", "mesh"],
)
def test_trace_settings_key_changes_with_each_trace_setting(setting) -> None:
    outside = _trace_settings_key()
    with setting():
        inside = _trace_settings_key()
    assert inside != outside
