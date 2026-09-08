"""State tiles reconstruct conditioned fold axes inside unchanged co-map wrappers."""

import functools
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from tests.conftest import assert_agrees_to_ulp


def _Q_and_F(
    *, action, region, risk_kind, shock_outer, shock_inner, next_regime_to_V_arr
):
    """Expose each state coordinate and its co-mapped continuation slice."""
    target = (region + risk_kind + shock_outer + shock_inner) % 4
    best = (
        100.0 * region
        + 10.0 * risk_kind
        + 3.0 * shock_outer
        + 2.0 * shock_inner
        + next_regime_to_V_arr["target"]
    )
    return best - jnp.square(action - target), jnp.ones((), dtype=bool)


@pytest.mark.parametrize("stream_actions", [False, True])
@pytest.mark.parametrize("cell_width", [1, 5, 12])
@pytest.mark.parametrize("execution", ["eager", "jit", "aot"])
@pytest.mark.parametrize("untile_conditioning_axis", [False, True])
def test_tiled_conditioned_folds_preserve_co_mapped_state_axes(
    *,
    stream_actions: bool,
    cell_width: int,
    execution: str,
    untile_conditioning_axis: bool,
) -> None:
    """Each fold reads its own reconstructed axis before device-local wrapping."""
    builder = get_streaming_max_Q_over_a if stream_actions else get_max_Q_over_a
    function = builder(
        Q_and_F=_Q_and_F,
        batch_sizes=dict.fromkeys(
            ("region", "risk_kind", "shock_outer", "shock_inner"), 0
        ),
        action_names=("action",),
        state_names=("region", "risk_kind", "shock_outer", "shock_inner"),
        co_map_state_names=("region",),
        co_map_v_arr_in_axes=(MappingProxyType({"target": 0}),),
        fold_state_names=("shock_outer", "shock_inner"),
        fold_weights=MappingProxyType(
            {
                "shock_outer": jnp.asarray([[0.25, 0.75], [0.75, 0.25]]),
                "shock_inner": jnp.asarray([0.25, 0.25, 0.5]),
            }
        ),
        fold_conditioning=MappingProxyType({"shock_outer": "risk_kind"}),
        cell_width_keyword="cell_width",
        untiled_state_names=("risk_kind",) if untile_conditioning_axis else (),
    )
    widths = {"cell_width": cell_width}
    if stream_actions:
        widths["_lcm_action_block_width"] = 3
    function = functools.partial(function, **widths)
    arguments = {
        "action": jnp.arange(4, dtype=jnp.int32),
        "region": jnp.arange(2, dtype=jnp.int32),
        "risk_kind": jnp.arange(2, dtype=jnp.int32),
        "shock_outer": jnp.arange(2, dtype=jnp.int32),
        "shock_inner": jnp.arange(3, dtype=jnp.int32),
        "next_regime_to_V_arr": MappingProxyType({"target": jnp.asarray([7.0, 11.0])}),
    }
    if execution != "eager":
        function = jax.jit(function)
        if execution == "aot":
            function = function.lower(**arguments).compile()
    actual = cast("jax.Array", function(**arguments))
    assert_agrees_to_ulp(
        got=actual,
        expected=np.asarray([[11.75, 20.25], [115.75, 124.25]], dtype=actual.dtype),
        n_ulp=4,
    )
