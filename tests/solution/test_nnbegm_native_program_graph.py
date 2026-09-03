"""An NNBEGM period publishes one native composite core-program graph.

The composite owns no traced body of its own: it republishes the two inner NB-EGM
replay programs under `keeper:` and `adjuster:` prefixes, with the inner roles and
requirements, and hands each its own context. The keeper sees the period's inputs
unchanged; the adjuster sees the outer post-decision bound at the first outer node, the
value every per-node call later rebinds. Both republished programs carry scope `ANY`:
the nested solve reads the inner policy banks under every retention, so a values-only
solve dispatches the same programs as a replay-retaining one.

The kernel class follows the outer search: a finite grid collapses the exact candidate
set, an adaptive mesh refines and collapses continuously. The two share one base that
runs the keeper and settles the replay capability, so a caller that wraps or records the
base call sees both.
"""

import ast
import functools
import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

import jax
import numpy as np
import pytest

from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    ProgramScope,
    core_program_graph,
    materialize_core_program,
)
from _lcm.solution import backward_induction
from _lcm.solution import nnbegm as nnbegm_module
from _lcm.solution.negm import _with_outer_post_decision
from tests.simulation.test_nnbegm_split_workflow_parity import _MESH, _PARAMS
from tests.solution._nbegm_direct_oracle import ride_along_kernel
from tests.test_models import n_nbegm_toy as toy

_ROUTES = {"finite": None, "adaptive": _MESH}


def _kernel(route: str) -> tuple[Any, dict[str, Any]]:
    model = toy.build_model(variant="n_nbegm", n_periods=3, outer_search=_ROUTES[route])
    return ride_along_kernel(model=model, params=_PARAMS)


def _build_context(context: Mapping[str, Any]) -> CoreBuildContext:
    return CoreBuildContext(
        state_action_space=context["state_action_space"],
        next_regime_to_V_arr=context["next_regime_to_V_arr"],
        next_regime_to_continuation=context["next_regime_to_continuation"],
        flat_params=context["flat_params"],
        period=context["period"],
        ages=context["ages"],
    )


def _assert_same_arguments(
    *, actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    assert set(actual) == set(expected)
    assert jax.tree.structure(dict(actual)) == jax.tree.structure(dict(expected))
    for got, want in zip(
        jax.tree.leaves(dict(actual)), jax.tree.leaves(dict(expected)), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize(
    ("route", "kernel_class_name"),
    [
        ("finite", "_FiniteNNBEGMPeriodKernel"),
        ("adaptive", "_AdaptiveNNBEGMPeriodKernel"),
    ],
)
def test_the_kernel_class_follows_the_outer_search(
    *, route: str, kernel_class_name: str
):
    kernel, _ = _kernel(route)

    assert type(kernel) is getattr(nnbegm_module, kernel_class_name)
    assert isinstance(kernel, nnbegm_module._NNBEGMPeriodKernel)


def test_the_graph_republishes_the_inner_replay_programs_under_role_prefixes():
    kernel, _ = _kernel("finite")
    graph = core_program_graph(kernel=kernel)

    assert tuple(graph) == ("keeper:replay", "adjuster:replay")
    for role, inner_kernel in (
        ("keeper", kernel.keeper_kernel),
        ("adjuster", kernel.adjuster_kernel),
    ):
        inner = core_program_graph(kernel=inner_kernel)["replay"]
        program = graph[f"{role}:replay"]
        assert program.scope is ProgramScope.ANY
        assert program.disposition is CoreExecutionDisposition.PLANNED
        assert program.disposition_reason is None
        assert program.function is inner.function
        assert program.requirements == inner.requirements
        assert program.output_roles == inner.output_roles
        assert program.donation_candidates == inner.donation_candidates


def test_the_keeper_builder_hands_the_inner_keeper_the_period_context():
    kernel, context = _kernel("finite")
    build_context = _build_context(context)

    composite = materialize_core_program(
        program=core_program_graph(kernel=kernel)["keeper:replay"],
        context=build_context,
    )
    inner = materialize_core_program(
        program=core_program_graph(kernel=kernel.keeper_kernel)["replay"],
        context=build_context,
    )

    _assert_same_arguments(actual=composite.arguments, expected=inner.arguments)


def test_the_adjuster_builder_binds_the_first_outer_node():
    kernel, context = _kernel("finite")
    first_node = kernel.outer_grid_values[0]

    composite = materialize_core_program(
        program=core_program_graph(kernel=kernel)["adjuster:replay"],
        context=_build_context(context),
    )
    inner = materialize_core_program(
        program=core_program_graph(kernel=kernel.adjuster_kernel)["replay"],
        context=_build_context(
            {
                **context,
                "flat_params": _with_outer_post_decision(
                    flat_params=context["flat_params"],
                    regime_name=kernel.regime_name,
                    outer_post_decision=kernel.outer_post_decision,
                    value=first_node,
                ),
            }
        ),
    )

    _assert_same_arguments(actual=composite.arguments, expected=inner.arguments)
    np.testing.assert_array_equal(
        np.asarray(composite.arguments[kernel.outer_post_decision]),
        np.asarray(first_node),
    )


@pytest.mark.parametrize(
    "legacy_name", ["cores", "core", "split_cores", "build_lower_args"]
)
def test_no_legacy_core_authority_survives_on_the_kernel(legacy_name: str):
    kernel, _ = _kernel("finite")
    assert not hasattr(kernel, legacy_name)


def test_the_composite_never_routes_inner_results_through_the_legacy_adapter():
    """The inner NB-EGM kernels are called directly; no result adapter is named."""
    tree = ast.parse(inspect.getsource(nnbegm_module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "require_legacy_kernel_result" not in names


def test_with_fixed_params_rebinds_both_roles():
    kernel, _ = _kernel("finite")
    graph = core_program_graph(kernel=kernel)
    fixed = MappingProxyType(
        {kernel.regime_name: MappingProxyType({"discount_factor": 0.9})}
    )

    bound_graph = core_program_graph(
        kernel=kernel.with_fixed_params(fixed_flat_params=fixed)
    )

    for name, program in graph.items():
        function = cast("functools.partial", bound_graph[name].function)
        assert isinstance(function, functools.partial)
        assert function.func is program.function
        assert function.keywords["discount_factor"] == 0.9


@pytest.mark.parametrize("route", ["finite", "adaptive"])
def test_a_nested_model_solves_without_any_value_repair(
    *, route: str, monkeypatch: pytest.MonkeyPatch
):
    """Every regime of the toy publishes planned outputs, so nothing is repaired."""

    def refuse(**_kwargs: object) -> object:
        msg = "an unplanned value reached the repair path"
        raise AssertionError(msg)

    monkeypatch.setattr(backward_induction, "_repair_unplanned_kernel_value", refuse)
    model = toy.build_model(variant="n_nbegm", n_periods=3, outer_search=_ROUTES[route])

    model.solve(params=_PARAMS, log_level="off")
