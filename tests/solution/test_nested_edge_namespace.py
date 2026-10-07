"""A nested solver binds its outer node beside its source's edge parameters.

NEGM and NNBEGM fix the chosen outer post-decision node in the source regime's
own parameters before the inner solver binds its kernel. The source's edge
parameters stay in `params["edges"][source]`, so the inner binder joins the two
namespaces exactly once, and a regime law whose parameter is free solves exactly
like the same law with that parameter fixed.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.params.edges import regime_kernel_params
from _lcm.solution.negm import _with_outer_post_decision
from _lcm.typing import FlatParams
from lcm.solvers import AdaptiveOuterMesh
from tests.test_models import n_nbegm_toy as toy

_EDGE_BRANCHES = [
    pytest.param(MappingProxyType({}), id="no-edge-parameter"),
    pytest.param(
        MappingProxyType({"final_age_alive": jnp.asarray(25.0)}),
        id="free-edge-parameter",
    ),
]
_N_PERIODS = 3
_FINAL_AGE_ALIVE = 25.0
_MESH = AdaptiveOuterMesh(
    initial_grid=toy.OUTER_GRID,
    max_nodes=513,
    max_refinement_rounds=10,
    value_atol=1e-4,
    value_rtol=1e-4,
    golden_iterations=40,
)


@pytest.mark.parametrize("edge_branch", _EDGE_BRANCHES)
def test_outer_node_joins_only_the_source_own_parameters(edge_branch):
    """The source's own branch holds its own parameters plus the outer node."""
    bound = _bind_outer_node(flat_params=_flat_params(edge_branch=edge_branch))
    assert set(bound["alive"]) == {"utility__rho", "new_illiquid"}


@pytest.mark.parametrize("edge_branch", _EDGE_BRANCHES)
def test_outer_node_binding_keeps_every_edge_branch(edge_branch):
    """Every source's edge branch, the bound source's included, is unchanged."""
    flat_params = _flat_params(edge_branch=edge_branch)
    bound = _bind_outer_node(flat_params=flat_params)
    assert bound["edges"] is flat_params["edges"]


@pytest.mark.parametrize("edge_branch", _EDGE_BRANCHES)
def test_kernel_parameters_after_outer_node_binding_join_both_namespaces(
    edge_branch,
):
    """The inner kernel reads own parameters, the outer node and edge parameters."""
    bound = _bind_outer_node(flat_params=_flat_params(edge_branch=edge_branch))
    assert set(regime_kernel_params(bound, regime_name="alive")) == {
        "utility__rho",
        "new_illiquid",
        *edge_branch,
    }


@pytest.mark.parametrize(
    ("variant", "outer_search"),
    [
        pytest.param("negm", None, id="negm"),
        pytest.param("n_nbegm", None, id="nnbegm-finite-outer-grid"),
        pytest.param("n_nbegm", _MESH, id="nnbegm-adaptive-outer-mesh"),
    ],
)
def test_free_regime_law_parameter_solves_like_the_fixed_one(*, variant, outer_search):
    """A free `final_age_alive` gives byte-identical values to the fixed one.

    Every value array agrees in its period and regime keys, shape, dtype and
    bytes.
    """
    fixed = toy.build_model(
        variant=variant, n_periods=_N_PERIODS, outer_search=outer_search
    ).solve(params={"discount_factor": 0.95}, log_level="debug")
    free = toy.build_model(
        variant=variant,
        n_periods=_N_PERIODS,
        outer_search=outer_search,
        fixed_params={},
    ).solve(
        params={
            "discount_factor": 0.95,
            "edges": {"alive": {"final_age_alive": _FINAL_AGE_ALIVE}},
        },
        log_level="debug",
    )
    assert _value_fingerprints(free.values) == _value_fingerprints(fixed.values)


def _flat_params(*, edge_branch: Mapping[str, object]) -> FlatParams:
    """Flat params of a nested source `alive` and a second source `retired`."""
    return cast(
        "FlatParams",
        MappingProxyType(
            {
                "alive": MappingProxyType({"utility__rho": jnp.asarray(2.0)}),
                "retired": MappingProxyType({"utility__rho": jnp.asarray(3.0)}),
                "dead": MappingProxyType({}),
                "edges": MappingProxyType(
                    {
                        "alive": edge_branch,
                        "retired": MappingProxyType({"exit_age": jnp.asarray(30.0)}),
                    }
                ),
            }
        ),
    )


def _bind_outer_node(*, flat_params: FlatParams) -> FlatParams:
    return _with_outer_post_decision(
        flat_params=flat_params,
        regime_name="alive",
        outer_post_decision="new_illiquid",
        value=jnp.asarray(5.0),
    )


def _value_fingerprints(
    values: Mapping[int, Mapping[str, object]],
) -> dict[tuple[int, str], tuple[tuple[int, ...], str, bytes]]:
    """Return each value array's shape, dtype and C-order bytes by period and regime."""
    fingerprints: dict[tuple[int, str], tuple[tuple[int, ...], str, bytes]] = {}
    for period, by_regime in values.items():
        for regime_name, value in by_regime.items():
            array = np.asarray(value)
            fingerprints[(period, regime_name)] = (
                array.shape,
                array.dtype.str,
                array.tobytes(order="C"),
            )
    return fingerprints
