import jax
import pytest
from jax import numpy as jnp

from lcm.ages import AgeGrid
from lcm.grids import categorical
from lcm.grids.continuous import LinSpacedGrid
from lcm.grids.discrete import DiscreteGrid
from lcm.model import Model
from lcm.regime import Regime

try:
    jax.config.update("jax_num_cpu_devices", 4)
    _HAS_4_CPU = len(jax.devices()) >= 4
except jax.errors.JaxRuntimeError:
    _HAS_4_CPU = False

_skip_no_4_cpu = pytest.mark.skipif(not _HAS_4_CPU, reason="requires 4 CPU's")


@pytest.fixture
def distributed_model():
    @categorical(ordered=False)
    class RegimeId:
        working_life: int
        retirement: int

    @categorical(ordered=True)
    class Type:
        low: int
        high: int

    # Define a regime where 'unused_state' is not used in any function
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


@_skip_no_4_cpu
def test_solution_running_on_multiple_cpus(distributed_model):
    """Test that distribution over multiple CPU's works."""

    period_to_regime_to_V_arr = distributed_model.solve(
        params={"discount_factor": 0.95},
    )

    assert period_to_regime_to_V_arr[0]["working_life"].sharding.num_devices == 4


@_skip_no_4_cpu
def test_simulation_running_on_multiple_cpus(distributed_model):
    """Test that distribution over multiple CPU's works."""

    res = distributed_model.simulate(
        params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(4, 0),
            "wealth": jnp.full(4, 100.0),
            "type1": jnp.full(4, 1),
            "type2": jnp.full(4, 1),
            "regime": jnp.zeros(4, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,
    )

    assert res._raw_results["working_life"][2].states["type1"].sharding.num_devices == 4
    assert res._raw_results["working_life"][2].states["type2"].sharding.num_devices == 4
    assert (
        res._raw_results["working_life"][2].states["wealth"].sharding.num_devices == 4
    )
