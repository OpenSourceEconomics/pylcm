import jax
import pandas as pd
import pytest
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from _lcm.grids import categorical
from _lcm.grids.continuous import LinSpacedGrid
from _lcm.grids.discrete import DiscreteGrid
from _lcm.utils.logging import v_array_has_inf, v_array_has_nan
from lcm.ages import AgeGrid
from lcm.exceptions import PyLCMError, RegimeInitializationError
from lcm.model import Model
from lcm.regime import Regime as UserRegime
from lcm.result import SimulationResult
from lcm.typing import ScalarInt

# Run these tests on the CPU for parallelization, does not work if pytest runs
# multiple workers, because jax will be initialized already
try:
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_num_cpu_devices", 4)
    _PYTEST_PARALLEL = False
except RuntimeError:
    _PYTEST_PARALLEL = True

_skip_pytest_parallel = pytest.mark.skipif(
    _PYTEST_PARALLEL, reason="Can't set num cpus in pytest paralellel"
)


def _make_correct_distributed_model(*, n_subjects: int | None = None) -> Model:
    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=True)
    class Type:
        low: ScalarInt
        high: ScalarInt

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption, type1, type2: (
                (jnp.log(consumption) + wealth * 0.001) * type1 * type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
            ),
            "type1": DiscreteGrid(Type, distributed=True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
            "type1": None,
            "type2": None,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )

    retirement = UserRegime(
        transition=None,
        functions={
            "utility": lambda wealth, type1, type2: (wealth * 0.5) * type1 * type2
        },
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "type1": DiscreteGrid(Type, distributed=True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        active=lambda age: age >= 5,
    )

    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
        n_subjects=n_subjects,
    )


@pytest.fixture
def correct_distributed_model():
    return _make_correct_distributed_model()


@pytest.fixture
def wrong_distributed_model():
    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=True)
    class Type:
        low: ScalarInt
        medium: ScalarInt
        high: ScalarInt

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption, type1, type2: (
                (jnp.log(consumption) + wealth * 0.001) * type1 * type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
            ),
            "type1": DiscreteGrid(Type, distributed=True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
            "type1": None,
            "type2": None,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )

    retirement = UserRegime(
        transition=None,
        functions={
            "utility": lambda wealth, type1, type2: (wealth * 0.5) * type1 * type2
        },
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "type1": DiscreteGrid(Type, distributed=True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        active=lambda age: age >= 5,
    )

    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
    )


@_skip_pytest_parallel
def test_solution_running_on_multiple_cpus(correct_distributed_model):
    """Test that distribution over multiple CPU's works for solution."""

    period_to_regime_to_V_arr = correct_distributed_model.solve(
        log_level="debug",
        params={"discount_factor": 0.95},
    )

    assert period_to_regime_to_V_arr[0]["working_life"].sharding.num_devices == 4


@_skip_pytest_parallel
def test_solve_returns_eagerly_materialised_V_arrs(correct_distributed_model):
    """Every V_arr shard is materialised before `solve()` returns.

    Backward induction must drain the device-side compute graph before
    the simulate phase consumes the V_arrs, so V stays sharded but no
    pending kernels leak from solve to simulate.
    """
    period_to_regime_to_V_arr = correct_distributed_model.solve(
        log_level="off",
        params={"discount_factor": 0.95},
    )
    for regime_to_V_arr in period_to_regime_to_V_arr.values():
        for V_arr in regime_to_V_arr.values():
            for shard in V_arr.addressable_shards:
                assert shard.data.is_ready()


@_skip_pytest_parallel
def test_simulate_returns_eagerly_materialised_V_arrs(correct_distributed_model):
    """Every V_arr in the `SimulationResult` is materialised before `simulate()`
    returns.

    Forward simulation must drain its lazy compute graph before returning so
    downstream consumers (`to_dataframe`, `save`, anything that reads from
    `raw_results`) start with concrete arrays rather than pending kernels.
    """
    res = correct_distributed_model.simulate(
        log_level="off",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )
    for regime_period_data in res._raw_results.values():
        for period_data in regime_period_data.values():
            for shard in period_data.V_arr.addressable_shards:
                assert shard.data.is_ready()


@_skip_pytest_parallel
def test_simulation_running_on_multiple_cpus(correct_distributed_model):
    """Test that distribution over multiple CPU's works for simulation."""

    res = correct_distributed_model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert res._raw_results["working_life"][2].states["type1"].sharding.num_devices == 4
    assert res._raw_results["working_life"][2].states["type2"].sharding.num_devices == 4
    assert (
        res._raw_results["working_life"][2].states["wealth"].sharding.num_devices == 4
    )


@_skip_pytest_parallel
def test_save_load_preserves_sharding_and_dataframe(
    correct_distributed_model, tmp_path
):
    """`save` / `load` round-trip preserves per-shard data and DataFrame output.

    Arrays must travel through the on-disk format without an implicit
    gather: each shard is written and restored on the same device mesh,
    and the `to_dataframe()` projection is byte-identical to the
    in-memory result.
    """
    res = correct_distributed_model.simulate(
        log_level="off",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    save_dir = tmp_path / "result"
    res.save(directory=save_dir)
    loaded = SimulationResult.load(directory=save_dir)

    for period, regime_dict in res._period_to_regime_to_V_arr.items():
        for regime_name, V_arr in regime_dict.items():
            loaded_V = loaded._period_to_regime_to_V_arr[period][regime_name]
            assert loaded_V.sharding.num_devices == V_arr.sharding.num_devices
            for original_shard, loaded_shard in zip(
                V_arr.addressable_shards,
                loaded_V.addressable_shards,
                strict=True,
            ):
                assert loaded_shard.data.shape == original_shard.data.shape

    pd.testing.assert_frame_equal(loaded.to_dataframe(), res.to_dataframe())


@_skip_pytest_parallel
def test_aot_compiled_simulation_running_on_multiple_cpus():
    """AOT-compiled simulate functions run on multi-device-sharded inputs.

    Setting `n_subjects` makes the first matching `simulate(...)` AOT-compile
    every simulate function for that batch shape. With distributed grids the
    runtime state and value-function arrays are device-sharded, so the
    compiled programs must be lowered with shardings matching what runtime
    dispatches rather than single-device defaults.
    """
    model = _make_correct_distributed_model(n_subjects=36)

    res = model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime_id": jnp.zeros(36, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert (
        res._raw_results["working_life"][2].states["wealth"].sharding.num_devices == 4
    )


@_skip_pytest_parallel
def test_solution_error_if_grid_product_exceeds_devices(wrong_distributed_model):
    """Solve raises when the product of distributed grid sizes exceeds devices."""

    with pytest.raises(PyLCMError, match="must equal the number"):
        wrong_distributed_model.solve(
            log_level="debug",
            params={"discount_factor": 0.95},
        )


@_skip_pytest_parallel
def test_simulation_pads_non_device_multiple_subject_count(correct_distributed_model):
    """A subject count that is not a multiple of the device count simulates cleanly.

    Distributed grids shard subjects across devices, which needs the leading axis to
    divide evenly. pylcm pads internally (duplicating the last subject up to the next
    device multiple) and trims the pad rows back out, so 5 subjects on 4 devices
    yields a result with exactly 5 subjects.
    """
    result = correct_distributed_model.simulate(
        log_level="debug",
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(5, 0),
            "wealth": jnp.full(5, 100.0),
            "type1": jnp.full(5, 1),
            "type2": jnp.full(5, 1),
            "regime_id": jnp.zeros(5, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert result.n_subjects == 5
    assert result.to_dataframe()["subject_id"].nunique() == 5


@_skip_pytest_parallel
@pytest.mark.parametrize("subject_batch_size", [3, 4])
def test_distributed_simulation_rejects_subject_batching(
    correct_distributed_model,
    subject_batch_size,
):
    """Subject-batching is rejected under multi-device distribution.

    The value-function array is sharded across the devices and cannot be gathered
    onto one, so chunking the subject axis (a single-device operation) cannot be
    combined with distributed grids on more than one device — rejected even at a
    batch size that divides the device count (4), not only a non-multiple (3).
    """
    with pytest.raises(PyLCMError, match="distributed grids"):
        correct_distributed_model.simulate(
            log_level="debug",
            params={"discount_factor": 0.95},
            initial_conditions={
                "age": jnp.full(8, 0),
                "wealth": jnp.full(8, 100.0),
                "type1": jnp.ones(8, dtype=jnp.int32),
                "type2": jnp.ones(8, dtype=jnp.int32),
                "regime_id": jnp.zeros(8, dtype=jnp.int32),
            },
            period_to_regime_to_V_arr=None,
            seed=12345,
            subject_batch_size=subject_batch_size,
        )


@_skip_pytest_parallel
def test_distributed_simulation_auto_subject_batch_size_runs_one_pass(
    correct_distributed_model,
):
    """`subject_batch_size="auto"` is a one-pass no-op under multi-device distribution.

    The value-function array is sharded across the devices, so the subject axis
    can't be chunked; `"auto"` falls back to a single sharded pass rather than
    raising, reproducing the explicit single-pass result.
    """
    initial_conditions = {
        "age": jnp.full(8, 0),
        "wealth": jnp.linspace(50.0, 120.0, 8),
        "type1": jnp.ones(8, dtype=jnp.int32),
        "type2": jnp.ones(8, dtype=jnp.int32),
        "regime_id": jnp.zeros(8, dtype=jnp.int32),
    }

    def _simulate(subject_batch_size):
        return (
            correct_distributed_model.simulate(
                log_level="debug",
                params={"discount_factor": 0.95},
                initial_conditions=initial_conditions,
                period_to_regime_to_V_arr=None,
                seed=12345,
                subject_batch_size=subject_batch_size,
            )
            .to_dataframe()
            .sort_values(["subject_id", "period"])
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(_simulate(0), _simulate("auto"))


@pytest.fixture
def partially_distributed_model():
    """Model where one regime has distributed grids and the other does not."""

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=True)
    class Type:
        low: ScalarInt
        high: ScalarInt

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption, type1, type2: (
                (jnp.log(consumption) + wealth * 0.001) * type1 * type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "type1": DiscreteGrid(Type, distributed=True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
            "type1": None,
            "type2": None,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )

    retirement = UserRegime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
        active=lambda age: age >= 5,
    )

    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
    )


@_skip_pytest_parallel
def test_solve_with_partial_distribution_returns_correct_shardings(
    partially_distributed_model,
):
    """Mixed-regime models solve cleanly: distributed regimes get sharded V-arrays.

    The distributed regime's V-array is sharded across all devices; the
    undistributed regime's V-array carries no per-axis sharding (single device).
    """
    period_to_regime_to_V_arr = partially_distributed_model.solve(
        log_level="debug",
        params={"discount_factor": 0.95},
    )
    assert period_to_regime_to_V_arr[0]["working_life"].sharding.num_devices == 4
    assert period_to_regime_to_V_arr[5]["retirement"].sharding.num_devices == 1


def test_distributed_action_grid_raises_at_regime_init():
    """Action grids cannot be distributed; constructing a `Regime` with one raises.

    Distribution is a property of state axes (which form the V-array shape).
    Marking an action grid as distributed has no consistent meaning under the
    current sharding model, so it is rejected at construction time. (Continuous
    action grids never reach this check — they are rejected at grid init by
    `_fail_if_continuous_grid_distributed`.)
    """

    @categorical(ordered=False)
    class Choice:
        a: ScalarInt
        b: ScalarInt

    with pytest.raises(RegimeInitializationError, match="distributed=True"):
        UserRegime(
            functions={"utility": jnp.log},
            states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
            state_transitions={
                "wealth": lambda wealth, choice: wealth - choice,
            },
            actions={
                "choice": DiscreteGrid(Choice, distributed=True),
            },
            transition=lambda age: age,
        )


@_skip_pytest_parallel
def test_v_array_has_nan_keeps_reduction_sharded_on_distributed_input():
    """`v_array_has_nan` returns a mesh-replicated scalar, not a single-device one.

    The reduction stays inside `@jax.jit` so XLA partitions it across the V-array's
    devices (per-device any → all-reduce → replicated scalar) instead of gathering
    the full V-array onto the default device first.
    """
    mesh = jax.make_mesh((4,), ("dev",))
    sharded = jax.device_put(
        jnp.zeros((8,), dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec("dev")),
    )

    result = v_array_has_nan(sharded)

    assert bool(result) is False
    assert result.sharding.num_devices == 4
    assert result.sharding.is_fully_replicated


@_skip_pytest_parallel
def test_v_array_has_inf_keeps_reduction_sharded_on_distributed_input():
    """`v_array_has_inf` returns a mesh-replicated scalar, not a single-device one."""
    mesh = jax.make_mesh((4,), ("dev",))
    sharded = jax.device_put(
        jnp.array([0.0, jnp.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        NamedSharding(mesh, PartitionSpec("dev")),
    )

    result = v_array_has_inf(sharded)

    assert bool(result) is True
    assert result.sharding.num_devices == 4
    assert result.sharding.is_fully_replicated
