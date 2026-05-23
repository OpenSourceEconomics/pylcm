import jax
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


def _make_distributed_batched_model() -> Model:
    """A model whose distributed state grid also sets `batch_size`."""

    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    def wealth():
        return LinSpacedGrid(
            start=1, stop=100, n_points=8, distributed=True, batch_size=1
        )

    working_life = UserRegime(
        functions={
            "utility": lambda wealth, consumption: jnp.log(consumption) + wealth * 0.001
        },
        states={"wealth": wealth()},
        state_transitions={"wealth": lambda wealth, consumption: wealth - consumption},
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 4, RegimeId.retirement, RegimeId.working_life
        ),
        active=lambda age: age < 5,
    )
    retirement = UserRegime(
        transition=None,
        functions={"utility": lambda wealth: wealth * 0.5},
        states={"wealth": wealth()},
        active=lambda age: age >= 5,
    )
    return Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
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
def test_solution_with_distributed_and_batched_grid():
    """Solve emits sharded V-arrays when a distributed grid also batches.

    A grid that is both `distributed=True` and `batch_size > 0` must still
    produce V-arrays carrying the regime's declared device sharding, so they
    match the `next_regime_to_V_arr` template the AOT-compiled consumers were
    lowered against.
    """
    model = _make_distributed_batched_model()

    period_to_regime_to_V_arr = model.solve(
        log_level="debug",
        params={"discount_factor": 0.95},
    )

    assert period_to_regime_to_V_arr[0]["working_life"].sharding.num_devices == 4


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
def test_simulation_error_if_not_multiple(correct_distributed_model):
    """Test that simulation throws error if too many subjects for num cpus."""

    with pytest.raises(PyLCMError, match="multiple"):
        correct_distributed_model.simulate(
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


@_skip_pytest_parallel
def test_v_array_has_nan_keeps_reduction_sharded_on_distributed_input():
    """`v_array_has_nan` returns a mesh-replicated scalar, not a single-device one.

    The reduction stays inside `@jax.jit` so XLA partitions it across the
    V-array's devices (per-device any → all-reduce → replicated scalar)
    instead of gathering the full V-array onto the default device first.
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


def test_distributed_action_grid_raises_at_regime_init():
    """Action grids cannot be distributed; constructing a `Regime` with one raises.

    Distribution is a property of state axes (which form the V-array shape).
    Marking an action grid as distributed has no consistent meaning under the
    current sharding model, so it is rejected at construction time.
    """
    with pytest.raises(RegimeInitializationError, match="distributed=True"):
        UserRegime(
            functions={"utility": jnp.log},
            states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=10)},
            state_transitions={
                "wealth": lambda wealth, consumption: wealth - consumption,
            },
            actions={
                "consumption": LinSpacedGrid(
                    start=1, stop=50, n_points=10, distributed=True
                ),
            },
            transition=lambda age: age,
        )
