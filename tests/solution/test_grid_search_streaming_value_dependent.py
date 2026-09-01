"""Streamed GridSearch preserves every value-dependent input channel."""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import CoreProgram, resolve_core_program
from _lcm.regime_building.max_Q_over_a import (
    get_max_Q_over_a,
    get_streaming_max_Q_over_a,
)
from _lcm.solution.backward_induction import (
    _build_continuation_templates,
    _edge_kwargs,
)
from _lcm.solution.grid_search import _GridSearchPeriodKernel
from lcm import Model
from tests.simulation.test_aot_collective_and_gated import _make_consent_model
from tests.simulation.test_aot_same_period_refs import _make_participation_model


def _materialize_program(
    *,
    model: Model,
    regime_name: str,
    params: Mapping[str, Any],
) -> tuple[_GridSearchPeriodKernel, Mapping[str, object], CoreProgram]:
    """Build exactly the program arguments used by backward induction."""
    flat_params = model._process_params(params)
    next_V, next_continuation, next_edges = _build_continuation_templates(
        regimes=model._regimes,
        flat_params=flat_params,
    )
    regime = model._regimes[regime_name]
    edge_kwargs = cast(
        "dict[str, Any]",
        _edge_kwargs(
            regime=regime,
            regime_name=regime_name,
            next_edge_to_V_arr=next_edges,
        ),
    )
    kernel = regime.solution.period_kernels[0]
    assert isinstance(kernel, _GridSearchPeriodKernel)
    arguments = kernel.build_lower_args(
        core_key="main",
        state_action_space=regime.solution.state_action_space(
            regime_params=flat_params[regime_name]
        ),
        next_regime_to_V_arr=next_V,
        next_regime_to_continuation=next_continuation,
        flat_params=flat_params,
        period=0,
        ages=model.ages,
        **edge_kwargs,
    )
    program = kernel.build_core_program(core_key="main", arguments=arguments)
    assert program is not None
    return kernel, arguments, program


def _assert_tree_equal(*, actual: object, expected: object) -> None:
    """Compare every numerical output leaf exactly."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _participation_case() -> Model:
    """Build the same-period participation fixture."""
    return _make_participation_model(n_subjects=None)


def _consent_case() -> Model:
    """Build the gated-transition consent fixture."""
    return _make_consent_model(n_subjects=None)


@pytest.mark.parametrize(
    ("model_factory", "regime_name", "params", "expected_channels"),
    [
        pytest.param(
            _participation_case,
            "couple",
            {
                "couple": {
                    "koopmans_aggregator": {"discount_factor": 0.95},
                },
                "couple_terminal": {},
                "single_f": {
                    "koopmans_aggregator": {"discount_factor": 0.95},
                },
                "single_f_terminal": {},
            },
            (("single_f",), (), ()),
            id="same-period-reference",
        ),
        pytest.param(
            _consent_case,
            "single",
            {"discount_factor": 0.95},
            ((), ("single_terminal",), ("married_terminal",)),
            id="edge-reference-and-gated-target",
        ),
    ],
)
def test_value_dependent_model_declares_a_dense_equivalent_streamed_program(
    *,
    model_factory: Callable[[], Model],
    regime_name: str,
    params: Mapping[str, Any],
    expected_channels: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> None:
    """Processed same-period and edge routes opt into the planned core."""
    kernel, arguments, program = _materialize_program(
        model=model_factory(),
        regime_name=regime_name,
        params=params,
    )

    assert (
        kernel.same_period_ref_regimes,
        kernel.edge_reference_regimes,
        kernel.edge_target_regimes,
    ) == expected_channels
    assert kernel.unwrapped_core is not None
    dense = kernel.unwrapped_core(**arguments)
    resolved = resolve_core_program(program=program, tile_widths={"action": 1})
    streamed = resolved.function(**resolved.arguments, **resolved.static_kwargs)
    _assert_tree_equal(actual=streamed, expected=dense)


def _observable_Q_and_F(
    *,
    choice: jax.Array,
    next_regime_to_V_arr: Mapping[str, jax.Array],
    same_period_regime_to_V_arr: Mapping[str, jax.Array],
    same_period_regime_to_params: Mapping[str, Mapping[str, jax.Array]],
    edge_reference_regime_to_V_arr: Mapping[str, jax.Array],
    edge_reference_regime_to_params: Mapping[str, Mapping[str, jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """Make every value-dependent input observable in candidate values."""
    value = (
        choice
        + next_regime_to_V_arr["target"][0]
        + same_period_regime_to_V_arr["reference"][0]
        + same_period_regime_to_params["reference"]["offset"]
        + edge_reference_regime_to_V_arr["outside"][0]
        + edge_reference_regime_to_params["outside"]["offset"]
    )
    return value, choice != 1.0


def _observable_route() -> tuple[
    _GridSearchPeriodKernel, Mapping[str, object], CoreProgram
]:
    """Build a three-action route whose final width-two block is partial."""
    dense = get_max_Q_over_a(
        Q_and_F=_observable_Q_and_F,
        batch_sizes={},
        action_names=("choice",),
        state_names=(),
    )
    streamed = get_streaming_max_Q_over_a(
        Q_and_F=_observable_Q_and_F,
        batch_sizes={},
        action_names=("choice",),
        state_names=(),
    )
    kernel = _GridSearchPeriodKernel(
        core=dense,
        unwrapped_core=dense,
        streamed_core=streamed,
        action_names=("choice",),
        action_extents=(3,),
        regime_name="source",
        same_period_ref_regimes=("reference",),
        edge_reference_regimes=("outside",),
        edge_target_regimes=("target",),
    )
    arguments: Mapping[str, object] = MappingProxyType(
        {
            "choice": jnp.asarray([0.0, 1.0, 2.0]),
            "next_regime_to_V_arr": {"target": jnp.asarray([10.0])},
            "same_period_regime_to_V_arr": {"reference": jnp.asarray([20.0])},
            "same_period_regime_to_params": {
                "reference": {"offset": jnp.asarray(30.0)}
            },
            "edge_reference_regime_to_V_arr": {"outside": jnp.asarray([40.0])},
            "edge_reference_regime_to_params": {
                "outside": {"offset": jnp.asarray(50.0)}
            },
        }
    )
    program = kernel.build_core_program(core_key="main", arguments=arguments)
    assert program is not None
    return kernel, arguments, program


@pytest.mark.parametrize("width", [1, 2], ids=["unit", "partial-tail"])
def test_value_dependent_program_matches_dense_eager_jit_and_aot(width: int) -> None:
    """Block width cannot alter support or any value-dependent input."""
    kernel, arguments, program = _observable_route()
    assert kernel.unwrapped_core is not None
    dense = kernel.unwrapped_core(**arguments)
    resolved = resolve_core_program(program=program, tile_widths={"action": width})
    eager = resolved.function(**resolved.arguments, **resolved.static_kwargs)
    jitted_function = jax.jit(
        resolved.function, static_argnames=tuple(resolved.static_kwargs)
    )
    jitted = jitted_function(**resolved.arguments, **resolved.static_kwargs)
    compiled = jitted_function.lower(
        **resolved.arguments, **resolved.static_kwargs
    ).compile()
    aot = compiled(**resolved.arguments)

    for actual in (eager, jitted, aot):
        _assert_tree_equal(actual=actual, expected=dense)
