import jax
import pytest
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.exceptions import PyLCMError
from lcm.grids import categorical
from lcm.grids.continuous import LinSpacedGrid
from lcm.grids.discrete import DiscreteGrid
from lcm.model import Model
from lcm.regime import Regime
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


@pytest.fixture
def correct_distributed_model():
    @categorical(ordered=False)
    class RegimeId:
        working_life: ScalarInt
        retirement: ScalarInt

    @categorical(ordered=True)
    class Type:
        low: ScalarInt
        high: ScalarInt

    working_life = Regime(
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

    retirement = Regime(
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

    working_life = Regime(
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

    retirement = Regime(
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
        params={"discount_factor": 0.95},
    )

    assert period_to_regime_to_V_arr[0]["working_life"].sharding.num_devices == 4


@_skip_pytest_parallel
def test_simulation_running_on_multiple_cpus(correct_distributed_model):
    """Test that distribution over multiple CPU's works for simulation."""

    res = correct_distributed_model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(36, 0),
            "wealth": jnp.full(36, 100.0),
            "type1": jnp.full(36, 1),
            "type2": jnp.full(36, 1),
            "regime": jnp.zeros(36, dtype=jnp.int32),
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
def test_solution_error_if_not_multiple(wrong_distributed_model):
    """Test that solution throws error if too many states for num cpus."""

    with pytest.raises(PyLCMError, match="smaller than the number"):
        wrong_distributed_model.solve(
            params={"discount_factor": 0.95},
        )


@_skip_pytest_parallel
def test_simulation_error_if_not_multiple(correct_distributed_model):
    """Test that simulation throws error if too many subjects for num cpus."""

    with pytest.raises(PyLCMError, match="multiple"):
        correct_distributed_model.simulate(
            params={"discount_factor": 0.95},
            initial_conditions={
                "age": jnp.full(5, 0),
                "wealth": jnp.full(5, 100.0),
                "type1": jnp.full(5, 1),
                "type2": jnp.full(5, 1),
                "regime": jnp.zeros(5, dtype=jnp.int32),
            },
            period_to_regime_to_V_arr=None,
            seed=12345,
        )
