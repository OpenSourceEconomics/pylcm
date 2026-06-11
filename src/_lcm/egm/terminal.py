"""Carry producers for terminal regimes targeted by DC-EGM regimes.

A terminal regime's value is its utility, so its carry rows are closed-form
rather than the output of an EGM step. Two cases:

- *Terminal with a wealth state* (e.g. a bequest motive): the carry holds the
  terminal utility and its `jax.grad` with respect to the wealth state on the
  regime's own wealth grid. The grid lives in M-space with $R \\equiv M$; the
  parent's composed-gradient factor $\\partial R'/\\partial A$ handles the
  space uniformly.
- *Stateless terminal* (e.g. `dead` with `utility=lambda: 0.0`): there is no
  wealth grid and the marginal value of resources is identically zero. The
  carry holds constant-value, zero-marginal-utility broadcast rows; a parent
  whose entire continuation is such a regime hits the degenerate-inversion
  guard and correctly produces the consume-everything policy.
"""

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EgmCarry
from _lcm.typing import EconFunctionsMapping, EgmCarryProducer, StateName
from _lcm.utils.functools import get_union_of_args
from lcm.typing import FloatND, IntND, ScalarFloat

# Static row count of a stateless terminal carry: two grid slots suffice to
# represent a constant function under linear interpolation.
N_STATELESS_CARRY_ROWS = 2


def get_stateless_terminal_carry_producer() -> EgmCarryProducer:
    """Build the carry producer for a terminal regime without states.

    Returns:
        Producer mapping the regime's scalar value-function array to
        constant-value, zero-marginal-utility carry rows.

    """

    def produce_stateless_carry(
        *,
        V_arr: FloatND,
        **kwargs: FloatND | IntND,  # noqa: ARG001
    ) -> EgmCarry:
        """Broadcast the scalar terminal value into constant carry rows."""
        dtype = canonical_float_dtype()
        zeros = jnp.zeros(N_STATELESS_CARRY_ROWS, dtype=dtype)
        return EgmCarry(
            endog_grid=jnp.linspace(0.0, 1.0, N_STATELESS_CARRY_ROWS, dtype=dtype),
            policy=zeros,
            value=jnp.broadcast_to(
                jnp.asarray(V_arr, dtype=dtype), (N_STATELESS_CARRY_ROWS,)
            ),
            marginal_utility=zeros,
            taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        )

    return produce_stateless_carry


def get_terminal_wealth_carry_producer(
    *,
    functions: EconFunctionsMapping,
    state_name: StateName,
) -> EgmCarryProducer:
    """Build the carry producer for a terminal regime with one wealth state.

    The carry's value rows are the regime's value-function array (terminal
    value equals utility on the wealth grid) and its marginal-utility rows
    are `jax.grad` of the utility DAG with respect to the wealth state. The
    rows live in M-space ($R \\equiv M$), so the endogenous grid is the
    wealth grid itself.

    Args:
        functions: The terminal regime's processed functions (params renamed
            to qualified names).
        state_name: Name of the regime's wealth state.

    Returns:
        Producer mapping the wealth grid, the regime's flat params, and its
        value-function array to the terminal carry.

    """
    utility_func = concatenate_functions(
        functions={name: func for name, func in functions.items() if name != "H"},
        targets="utility",
        enforce_signature=False,
        set_annotations=True,
    )
    utility_extra_arg_names = tuple(
        get_union_of_args([utility_func]) - {state_name},
    )

    def produce_terminal_wealth_carry(
        *, V_arr: FloatND, **kwargs: FloatND | IntND
    ) -> EgmCarry:
        """Evaluate the terminal value and its wealth gradient on the grid."""
        dtype = canonical_float_dtype()
        wealth_grid = jnp.asarray(kwargs[state_name], dtype=dtype)
        extra = {name: kwargs[name] for name in utility_extra_arg_names}

        def utility_of_wealth(wealth_value: ScalarFloat) -> ScalarFloat:
            return utility_func(**{state_name: wealth_value}, **extra)

        value = jnp.asarray(V_arr, dtype=dtype)
        marginal_utility = jax.vmap(jax.grad(utility_of_wealth))(wealth_grid)
        return EgmCarry(
            endog_grid=wealth_grid,
            policy=wealth_grid,
            value=value,
            marginal_utility=jnp.where(
                jnp.isneginf(value), 0.0, marginal_utility
            ).astype(dtype),
            taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        )

    return produce_terminal_wealth_carry
