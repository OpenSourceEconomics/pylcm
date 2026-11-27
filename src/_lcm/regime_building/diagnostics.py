"""Per-period diagnostic closures and feasibility-conditional reductions.

Cold-path machinery used only when `validate_V` detects NaN in a solved
value-function array. `_build_compute_intermediates_per_period` produces
one JIT-compiled closure per period that productmaps the intermediates of the
period's action kernel (`get_intermediates`) over the full state-action space
and fuses the compute step with on-device reductions (`_wrap_with_reduction`).
The fused output is consumed by `_enrich_with_diagnostics` in
`_lcm.utils.error_handling`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import no_type_check

import jax
import jax.numpy as jnp

from _lcm.engine import StateActionSpace
from _lcm.regime_building.Q_and_F import Intermediates, get_intermediates
from _lcm.typing import ActionName, QAndFFunction, RegimeName, StateName
from _lcm.utils.dispatchers import productmap
from lcm.typing import BoolND, FloatND, IntND

# The NaN and feasibility reductions of one period's intermediates:
# `{Y}_overall` scalars and `{Y}_by_{name}` vectors, plus `regime_probs`, the mean
# transition probability of each target regime.
type DiagnosticReductions = MappingProxyType[
    str, FloatND | MappingProxyType[RegimeName, FloatND]
]


def _build_compute_intermediates_per_period(
    *,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    state_action_space: StateActionSpace,
    enable_jit: bool,
) -> MappingProxyType[int, Callable[..., DiagnosticReductions]]:
    """Build diagnostic intermediate closures for each period of a non-terminal regime.

    Each closure fuses a productmap of the period's kernel intermediates over the
    full state-action space with on-device reductions and is JIT-compiled.
    Periods sharing one kernel share one closure. The caller builds the kernels
    for the full value arrays the diagnostics are handed, so no co-mapped axis has
    been sliced off them.

    Args:
        Q_and_F_functions: Immutable mapping of period to the action kernel the
            diagnostics recompute.
        state_action_space: State-action space used for productmap sizing.
        enable_jit: Whether to JIT-compile the fused closure.

    Returns:
        Immutable mapping of period index to fused closure.

    """
    variable_names = (
        *state_action_space.state_names,
        *state_action_space.action_names,
    )
    built: dict[int, Callable[..., DiagnosticReductions]] = {}
    for kernel in Q_and_F_functions.values():
        if id(kernel) in built:
            continue
        mapped = _productmap_over_state_action_space(
            func=get_intermediates(Q_and_F=kernel),
            action_names=state_action_space.action_names,
            state_names=state_action_space.state_names,
            state_batch_sizes=dict.fromkeys(state_action_space.state_names, 0),
        )
        fused = _wrap_with_reduction(func=mapped, variable_names=variable_names)
        built[id(kernel)] = jax.jit(fused) if enable_jit else fused
    return MappingProxyType(
        {period: built[id(kernel)] for period, kernel in Q_and_F_functions.items()}
    )


def _wrap_with_reduction(
    *,
    func: Callable[..., Intermediates],
    variable_names: tuple[str, ...],
) -> Callable[..., DiagnosticReductions]:
    """Fuse a productmap'd intermediates function with on-device reductions.

    The wrapped function returns a flat pytree of scalars and per-dimension
    vectors instead of full state-action-shaped arrays. When JIT-compiled,
    XLA can often fuse the compute and reduce steps so the full-shape
    intermediates never materialise.

    Args:
        func: Productmap'd closure returning
            `(U_arr, F_arr, CE, Q_arr, regime_probs)`. `regime_probs`
            is a mapping of target regime names to per-point probability
            arrays.
        variable_names: Tuple of state + action names in the order that
            matches the productmap axes of `func`. Used to label the
            `{metric}_by_{name}` reductions.

    Returns:
        Callable taking the same kwargs as `func` and returning a read-only
        mapping with `{Y}_overall` scalars and `{Y}_by_{name}` vectors for `Y`
        in {`U_nan`, `CE_nan`, `Q_nan`, `F_feasible`}, plus `regime_probs` as
        a read-only mapping of per-target scalar means. The `{U,CE,Q}_nan_*` fractions
        are conditional on feasibility (numerator restricted to feasible
        cells, denominator is the feasible-cell count); `F_feasible_*`
        is the plain mean over all cells.

    """

    return _ReducedIntermediates(func=func, variable_names=variable_names)


@dataclass(frozen=True, kw_only=True, eq=False)
class _ReducedIntermediates:
    """Productmap'd intermediates fused with on-device NaN/feasibility reductions."""

    func: Callable[..., Intermediates]
    """Productmap'd function returning `(U_arr, F_arr, CE, Q_arr, regime_probs)`."""
    variable_names: tuple[str, ...]
    """State + action names in the order of `func`'s productmap axes."""

    # `kwargs` carries the wrapped function's full input map: the
    # `next_regime_to_V_arr` mapping alongside the Float/Int/Bool-valued
    # state/action inputs.
    # The kernel is traced with whatever leaves its caller supplies -- tracers,
    # Python scalars, arrays of either integer width -- so its annotations
    # document the contract and are not enforced at call time.
    @no_type_check
    def __call__(
        self,
        **kwargs: MappingProxyType[RegimeName, FloatND] | FloatND | IntND | BoolND,
    ) -> DiagnosticReductions:
        U_arr, F_arr, CE, Q_arr, regime_probs = self.func(**kwargs)
        F_float = jnp.asarray(F_arr, dtype=float)
        # NaN-count arrays are masked by feasibility: only feasible cells
        # contribute to numerators. Infeasible cells are zeroed out because
        # the solver masks them before the max, so a NaN there never
        # propagates to V_arr — reporting it would conflate causes.
        nan_arrays: dict[str, FloatND] = {
            "U_nan": _nan_at_cell(values=U_arr, mask=F_float) * F_float,
            "CE_nan": _nan_at_cell(values=CE, mask=F_float) * F_float,
            "Q_nan": _nan_at_cell(values=Q_arr, mask=F_float) * F_float,
        }

        out: dict[str, FloatND | MappingProxyType[RegimeName, FloatND]] = {}
        F_total = jnp.maximum(jnp.sum(F_float), 1.0)
        for key, arr in nan_arrays.items():
            out[f"{key}_overall"] = jnp.sum(arr) / F_total
            for i, name in enumerate(self.variable_names):
                if i < arr.ndim:
                    axes = tuple(j for j in range(arr.ndim) if j != i)
                    F_slice = jnp.maximum(jnp.sum(F_float, axis=axes), 1.0)
                    out[f"{key}_by_{name}"] = jnp.sum(arr, axis=axes) / F_slice

        # F itself is a plain mean over all cells — it is the denominator's
        # source, not a conditional metric.
        out["F_feasible_overall"] = jnp.mean(F_float)
        for i, name in enumerate(self.variable_names):
            if i < F_float.ndim:
                axes = tuple(j for j in range(F_float.ndim) if j != i)
                out[f"F_feasible_by_{name}"] = jnp.mean(F_float, axis=axes)

        out["regime_probs"] = MappingProxyType(
            {k: jnp.mean(v) for k, v in regime_probs.items()}
        )
        return MappingProxyType(out)


def _nan_at_cell(*, values: FloatND, mask: FloatND) -> FloatND:
    """Return 1.0 where a cell's value is NaN, at the shape of the cell mask.

    In a regime with several members, the values carry a trailing member axis
    the mask does not. Such a cell counts as NaN where any member's value is.
    """
    return jnp.any(jnp.isnan(values).reshape(*mask.shape, -1), axis=-1).astype(float)


def _productmap_over_state_action_space(
    *,
    func: Callable[..., Intermediates],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    state_batch_sizes: dict[StateName, int],
) -> Callable[..., Intermediates]:
    """Wrap a scalar state-action function with productmap over actions then states.

    Matches the pattern used by `get_max_Q_over_a`: actions form the inner
    Cartesian product (unbatched), states form the outer loop (with batching).

    Args:
        func: Scalar function taking state and action values as keyword
            arguments.
        action_names: Tuple of action variable names; becomes the inner
            productmap (unbatched).
        state_names: Tuple of state variable names; becomes the outer
            productmap.
        state_batch_sizes: Mapping of state name to productmap batch size.

    Returns:
        Callable taking the same kwargs as `func` but expecting grid arrays
        instead of scalars for state and action variables. Output axes are
        ordered as `(*state_names, *action_names)`.

    """
    inner = productmap(
        func=func,
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
    )
    return productmap(
        func=inner,
        variables=state_names,
        batch_sizes=state_batch_sizes,
    )
