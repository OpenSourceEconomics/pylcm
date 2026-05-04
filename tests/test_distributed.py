from lcm.ages import AgeGrid
from lcm.grids import categorical
from lcm.grids.continuous import LinSpacedGrid
from lcm.grids.discrete import DiscreteGrid
from lcm.model import Model
from lcm.regime import MarkovTransition, Regime
from jax import numpy as jnp

def test_unused_state_raises_error():
    """Model raises error when a state is defined but never used."""

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
                (jnp.log(consumption) + wealth * 0.001)*type1*type2
            ),
        },
        states={
            "wealth": LinSpacedGrid(
                start=1,
                stop=100,
                n_points=10,
            ),
            "type1": DiscreteGrid(Type,distributed= True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        state_transitions={
            "wealth": lambda wealth, consumption: wealth - consumption,
            "type1": None,
            "type2": None,
        },
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(age >=4, RegimeId.retirement, RegimeId.working_life),
        active=lambda age: age < 5,
    )

    retirement = Regime(
        transition=None,
        functions={"utility": lambda wealth, type1, type2: (wealth * 0.5)*type1*type2},
        states={
            "wealth": LinSpacedGrid(start=1, stop=100, n_points=10),
            "type1": DiscreteGrid(Type, distributed=True),
            "type2": DiscreteGrid(Type, distributed=True),
        },
        active=lambda age: age >= 5,
    )

    model = Model(
        regimes={"working_life": working_life, "retirement": retirement},
        ages=AgeGrid(start=0, stop=5, step="Y"),
        regime_id_class=RegimeId,
    )
    res = model.simulate(params={"discount_factor": 0.95},
        initial_conditions={
            "age": jnp.full(5, 0),
            "wealth": jnp.full(5, 100.0),
            "type1": jnp.full(5, 1),
            "type2": jnp.full(5, 1),
            "regime": jnp.zeros(5, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        seed=12345,)
