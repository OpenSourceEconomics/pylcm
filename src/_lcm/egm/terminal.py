"""Carry producers for regimes targeted by DC-EGM / BQSEGM regimes.

A regime an endogenous-grid regime transitions into must publish its value and
marginal value of resources on its own state grid so the parent can interpolate
them as a continuation. Terminal targets carry their utility in closed form;
living brute (`GridSearch`) targets carry their solved value array and its
gradient. This module builds both.

A terminal regime's value is its utility, so its carry rows are closed-form
rather than the output of an EGM step. Two cases:

- *Terminal with a wealth state* (e.g. a bequest motive): the carry holds the
  terminal utility and its `jax.grad` with respect to the wealth state on the
  regime's own wealth grid. The grid lives in M-space with $R \\equiv M$; the
  parent's composed-gradient factor $\\partial R'/\\partial A$ handles the
  space uniformly. The terminal may additionally carry discrete states shared
  with the parent (e.g. a fixed `pref_type` whose bequest weight differs by
  type): the carry then has those discrete axes leading (in V state order,
  matching the value-function array's layout), one wealth row per discrete
  combo, and the parent selects its own combo by integer indexing the leading
  axes — the same alignment the non-terminal child read uses.
- *Stateless terminal* (e.g. `dead` with `utility=lambda: 0.0`): there is no
  wealth grid and the marginal value of resources is identically zero. The
  carry holds constant-value, zero-marginal-utility broadcast rows; a parent
  whose entire continuation is such a regime hits the degenerate-inversion
  guard and correctly produces the consume-everything policy.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry
from _lcm.typing import EconFunctionsMapping, EGMCarryProducer, StateName
from _lcm.utils.functools import get_union_of_args
from lcm.typing import FloatND, IntND, ScalarFloat

# Static row count of a stateless terminal carry: two grid slots suffice to
# represent a constant function under linear interpolation.
N_STATELESS_CARRY_ROWS = 2


def get_stateless_terminal_carry_producer() -> EGMCarryProducer:
    """Build the carry producer for a terminal regime without states.

    Returns:
        Producer mapping the regime's scalar value-function array to
        constant-value, zero-marginal-utility carry rows.

    """

    def produce_stateless_carry(
        *,
        V_arr: FloatND,
        **kwargs: FloatND | IntND,  # noqa: ARG001
    ) -> EGMCarry:
        """Broadcast the scalar terminal value into constant carry rows."""
        dtype = canonical_float_dtype()
        zeros = jnp.zeros(N_STATELESS_CARRY_ROWS, dtype=dtype)
        return EGMCarry(
            endog_grid=jnp.linspace(0.0, 1.0, N_STATELESS_CARRY_ROWS, dtype=dtype),
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
    discrete_state_names: tuple[StateName, ...] = (),
    passive_state_names: tuple[StateName, ...] = (),
    continuous_state_order: tuple[StateName, ...] = (),
) -> EGMCarryProducer:
    """Build the carry producer for a terminal regime with one Euler state.

    The carry's value rows are the regime's value-function array (terminal
    value equals utility on the state grid) and its marginal-utility rows
    are `jax.grad` of the utility DAG with respect to the Euler state. The
    rows live in M-space ($R \\equiv M$), so the endogenous grid is the
    Euler-state grid itself.

    The carry's leading axes are the shared discrete states, then the passive
    continuous states (the durable / outer margin of a NEGM parent), both in
    value-function-array order; the Euler state is the trailing (row) axis. Each
    leading combo holds the terminal utility and its Euler-state gradient
    evaluated at that combo's discrete codes and passive node values. The parent
    integer-indexes the discrete axes and interpolates the passive axes — the
    same alignment the non-terminal child read uses.

    Args:
        functions: The terminal regime's processed functions (params renamed
            to qualified names).
        state_name: Name of the regime's Euler state (the parent's continuous
            state, gradient and endogenous-grid axis).
        discrete_state_names: Shared discrete-state names in value-function
            (carry leading-axis) order; empty for a single-Euler-state carry.
        passive_state_names: Passive continuous-state names in value-function
            order — carried as interpolated leading axes; empty for a single
            continuous state.
        continuous_state_order: The value-function array's continuous-axis
            order, used to transpose `V_arr` into `(discrete…, passive…,
            Euler)`; empty defaults to `(state_name,)`.

    Returns:
        Producer mapping the state grids, the regime's flat params, and its
        value-function array to the terminal carry.

    """
    utility_func = concatenate_functions(
        functions={name: func for name, func in functions.items() if name != "H"},
        targets="utility",
        enforce_signature=False,
        set_annotations=True,
    )
    utility_extra_arg_names = tuple(
        get_union_of_args([utility_func])
        - {state_name}
        - set(discrete_state_names)
        - set(passive_state_names),
    )
    cont_order = continuous_state_order or (state_name,)

    def produce_terminal_wealth_carry(
        *, V_arr: FloatND, **kwargs: FloatND | IntND
    ) -> EGMCarry:
        """Evaluate the terminal value and its Euler-state gradient on the grid."""
        dtype = canonical_float_dtype()
        euler_grid = jnp.asarray(kwargs[state_name], dtype=dtype)
        passive_grids = tuple(
            jnp.asarray(kwargs[name], dtype=dtype) for name in passive_state_names
        )
        extra = {name: kwargs[name] for name in utility_extra_arg_names}
        discrete_grids = tuple(kwargs[name] for name in discrete_state_names)

        def euler_gradient_at_combo(
            discrete_codes: tuple[IntND, ...],
        ) -> FloatND:
            """Euler-state gradient block for one shared discrete combo.

            Returns a `(passive…, Euler)` block — the gradient evaluated over
            the mesh of passive node values (leading) and the Euler grid
            (trailing).
            """
            discrete_kwargs = dict(
                zip(discrete_state_names, discrete_codes, strict=True)
            )

            def utility_at_point(
                euler_value: ScalarFloat, passive_values: tuple[ScalarFloat, ...]
            ) -> ScalarFloat:
                passive_kwargs = dict(
                    zip(passive_state_names, passive_values, strict=True)
                )
                return utility_func(
                    **{state_name: euler_value},
                    **discrete_kwargs,
                    **passive_kwargs,
                    **extra,
                )

            grad_euler = jax.grad(utility_at_point, argnums=0)
            mesh = jnp.meshgrid(*passive_grids, euler_grid, indexing="ij")
            flat = jnp.stack([axis.ravel() for axis in mesh], axis=-1)
            grad_flat = jax.vmap(lambda row: grad_euler(row[-1], tuple(row[:-1])))(flat)
            return grad_flat.reshape(mesh[0].shape)

        value = _reorder_terminal_value(
            V_arr=jnp.asarray(V_arr, dtype=dtype),
            n_discrete=len(discrete_state_names),
            continuous_state_order=cont_order,
            passive_state_names=passive_state_names,
            euler_state_name=state_name,
        )
        marginal_utility = _grad_over_discrete_combos(
            wealth_gradient_at_combo=euler_gradient_at_combo,
            discrete_grids=discrete_grids,
        )
        leading_shape = value.shape[:-1]
        endog_grid = jnp.broadcast_to(euler_grid, (*leading_shape, euler_grid.shape[0]))
        return EGMCarry(
            endog_grid=endog_grid,
            value=value,
            marginal_utility=jnp.where(
                jnp.isneginf(value), 0.0, marginal_utility
            ).astype(dtype),
            taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        )

    return produce_terminal_wealth_carry


def get_brute_child_carry_producer(
    *,
    state_name: StateName,
    discrete_state_names: tuple[StateName, ...] = (),
    passive_state_names: tuple[StateName, ...] = (),
    continuous_state_order: tuple[StateName, ...] = (),
) -> EGMCarryProducer:
    """Build the carry producer for a living brute (`GridSearch`) carry target.

    A brute regime an endogenous-grid parent transitions into produces a solved
    value-function array rather than a closed-form value. Its carry holds that
    array as the value rows and the array's gradient with respect to the Euler
    state as the marginal-value-of-resources rows. The carry lives in M-space
    ($R \\equiv M$), so the endogenous grid is the Euler-state grid itself and
    the parent's composed-gradient factor $\\partial R'/\\partial A$ maps it into
    savings space.

    The carry's leading axes are the regime's discrete states (process states
    included, as node-valued discrete dimensions), then its passive continuous
    states, both in value-function-array order; the Euler state trails as the row
    axis. This matches the layout the parent's child read indexes.

    Args:
        state_name: Name of the regime's Euler state (the parent's continuous
            state, gradient and endogenous-grid axis).
        discrete_state_names: Discrete-state names in value-function (carry
            leading-axis) order; empty for a single-Euler-state carry.
        passive_state_names: Passive continuous-state names in value-function
            order — carried as interpolated leading axes; empty for a single
            continuous state.
        continuous_state_order: The value-function array's continuous-axis
            order, used to transpose `V_arr` into `(discrete…, passive…,
            Euler)`; empty defaults to `(state_name,)`.

    Returns:
        Producer mapping the state grids and the regime's value-function array to
        the brute child's carry.

    """
    cont_order = continuous_state_order or (state_name,)

    def produce_brute_child_carry(
        *, V_arr: FloatND, **kwargs: FloatND | IntND
    ) -> EGMCarry:
        """Carry the solved value array and its Euler-state gradient."""
        dtype = canonical_float_dtype()
        euler_grid = jnp.asarray(kwargs[state_name], dtype=dtype)
        value = _reorder_terminal_value(
            V_arr=jnp.asarray(V_arr, dtype=dtype),
            n_discrete=len(discrete_state_names),
            continuous_state_order=cont_order,
            passive_state_names=passive_state_names,
            euler_state_name=state_name,
        )
        # The marginal value of resources is the value array's slope along the
        # Euler grid; infeasible (`-inf`) rows carry zero marginal so the parent's
        # probability-weighted expectation stays finite.
        finite_value = jnp.where(jnp.isneginf(value), jnp.nan, value)
        marginal = jnp.asarray(jnp.gradient(finite_value, euler_grid, axis=-1))
        leading_shape = value.shape[:-1]
        endog_grid = jnp.broadcast_to(euler_grid, (*leading_shape, euler_grid.shape[0]))
        return EGMCarry(
            endog_grid=endog_grid,
            value=value,
            marginal_utility=jnp.where(jnp.isfinite(marginal), marginal, 0.0).astype(
                dtype
            ),
            taste_shock_scale=jnp.asarray(0.0, dtype=dtype),
        )

    return produce_brute_child_carry


def _reorder_terminal_value(
    *,
    V_arr: FloatND,
    n_discrete: int,
    continuous_state_order: tuple[StateName, ...],
    passive_state_names: tuple[StateName, ...],
    euler_state_name: StateName,
) -> FloatND:
    """Transpose `V_arr` into carry layout `(discrete…, passive…, Euler)`.

    `V_arr` has its discrete axes leading, then its continuous axes in
    `continuous_state_order`. The carry wants the passive continuous states
    (in value-function order) before the Euler state, which trails as the row
    axis; the discrete axes keep their leading positions.
    """
    target_continuous = (*passive_state_names, euler_state_name)
    continuous_perm = [
        n_discrete + continuous_state_order.index(name) for name in target_continuous
    ]
    axes = list(range(n_discrete)) + continuous_perm
    return jnp.transpose(V_arr, axes)


def _grad_over_discrete_combos(
    *,
    wealth_gradient_at_combo: object,
    discrete_grids: tuple[FloatND | IntND, ...],
) -> FloatND:
    """Stack the per-combo wealth gradients into the carry's leading-axis shape.

    Iterates the shared discrete grids in Python (their sizes are static), so
    each combo's gradient is a plain `jax.vmap` over the wealth grid. The
    result has the discrete axes leading (in the given order) and the wealth
    axis trailing, matching the value-function array's layout.
    """
    grad_at_combo = cast(
        "Callable[[tuple[IntND, ...]], FloatND]", wealth_gradient_at_combo
    )

    def build_axis(
        prefix: tuple[IntND, ...], remaining: tuple[FloatND | IntND, ...]
    ) -> FloatND:
        if not remaining:
            return grad_at_combo(prefix)
        head, *tail = remaining
        rows = [build_axis((*prefix, code), tuple(tail)) for code in jnp.asarray(head)]
        return jnp.stack(rows, axis=0)

    return build_axis((), discrete_grids)
