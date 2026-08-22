"""The five NBEGM partition sites select explicit ride or branch geometry.

These are routing tests, not throughput or arithmetic tests.  They record the
geometry passed by the production wrappers and return a shape-equivalent
``vmap`` result, so they remain runnable without an exact-affine kernel and
never compile the wide ride tile.
"""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm import fixed_width_map
from _lcm.solution import nbegm


def _record_partitions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int, int, int, int]]:
    """Collect rows, request, admission, width, and window for every call."""
    calls: list[tuple[int, int, int, int, int]] = []

    def spy(*, func, xs, requested_block_size, geometry):
        n_rows = fixed_width_map._leading_size(xs)
        window = fixed_width_map.max_block_size_for_axis(
            n_rows=n_rows,
            geometry=geometry,
        )
        admitted = fixed_width_map.admitted_block_size(
            requested=requested_block_size,
            max_block_size=window,
            microtile_width=geometry.microtile_width,
        )
        calls.append(
            (
                n_rows,
                requested_block_size,
                admitted,
                geometry.microtile_width,
                geometry.profile_window,
            )
        )
        return jax.vmap(func)(xs)

    monkeypatch.setattr(nbegm, "map_partitioned", spy)
    return calls


def _identity(row):
    return row


def test_cell_block_size_reaches_the_fixed_width_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cell request arrives with the ride geometry rather than being dropped."""
    calls = _record_partitions(monkeypatch)
    result = nbegm._map_ride_partitioned(
        func=_identity,
        xs=jnp.arange(3),
        requested_block_size=3,
    )

    assert result.shape == (3,)
    assert calls == [(3, 3, 256, 256, 256)]


def test_interval_batch_size_reaches_the_fixed_width_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interval request uses the same explicitly declared ride geometry."""
    calls = _record_partitions(monkeypatch)
    nbegm._map_ride_partitioned(
        func=_identity,
        xs=jnp.arange(11),
        requested_block_size=2,
    )

    assert calls == [(11, 2, 256, 256, 256)]


@pytest.mark.parametrize("requested", [0, 1, 3])
def test_a_short_ride_axis_admits_one_ride_microtile_whatever_is_requested(
    monkeypatch: pytest.MonkeyPatch, requested: int
) -> None:
    """A short ride axis admits one 256-row microtile for every request."""
    calls = _record_partitions(monkeypatch)
    nbegm._map_ride_partitioned(
        func=_identity,
        xs=jnp.arange(2),
        requested_block_size=requested,
    )

    assert calls[0][2:] == (256, 256, 256)


@pytest.mark.parametrize(("requested", "expected"), [(4, 4), (8, 8)])
def test_wide_branch_axis_admits_distinct_commit_strides(
    monkeypatch: pytest.MonkeyPatch, requested: int, expected: int
) -> None:
    """A 20-row branch axis retains distinct four- and eight-row requests."""
    calls = _record_partitions(monkeypatch)
    nbegm._map_branch_partitioned(
        func=_identity,
        xs=jnp.arange(20),
        requested_block_size=requested,
    )

    assert calls == [(20, requested, expected, 4, 64)]


def test_public_partition_requests_reach_all_five_production_maps() -> None:
    """Each public request reaches every production role through bound statics."""
    tree = ast.parse(Path(nbegm.__file__).read_text(encoding="utf-8"))

    routes: list[tuple[str, str, str, str]] = []

    class RouteCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if self.function_stack and isinstance(node.func, ast.Name):
                callee = node.func.id
                routes.extend(
                    (
                        self.function_stack[-1],
                        callee,
                        keyword.arg,
                        ast.unparse(keyword.value),
                    )
                    for keyword in node.keywords
                    if keyword.arg is not None
                )
            self.generic_visit(node)

    RouteCollector().visit(tree)

    expected = {
        # Public solver fields -> ride-along statics builder.
        (
            "_build_ride_along_kernels",
            "_nbegm_ride_along_statics",
            "cell_block_size",
            "self.cell_block_size",
        ),
        (
            "_build_ride_along_kernels",
            "_nbegm_ride_along_statics",
            "interval_batch_size",
            "self.interval_batch_size",
        ),
        (
            "_build_ride_along_kernels",
            "_nbegm_ride_along_statics",
            "branch_batch_size",
            "self.branch_batch_size",
        ),
        # Builder arguments -> bound statics.
        (
            "_nbegm_ride_along_statics",
            "_NBEGMRideAlongStatics",
            "cell_block_size",
            "cell_block_size",
        ),
        (
            "_nbegm_ride_along_statics",
            "_NBEGMRideAlongStatics",
            "interval_batch_size",
            "interval_batch_size",
        ),
        (
            "_nbegm_ride_along_statics",
            "_NBEGMRideAlongStatics",
            "branch_batch_size",
            "branch_batch_size",
        ),
        # Bound statics -> the five production maps.
        (
            "cell_continuation",
            "_map_branch_partitioned",
            "requested_block_size",
            "statics.branch_batch_size",
        ),
        (
            "_cell_rows_for_pool",
            "_map_ride_partitioned",
            "requested_block_size",
            "statics.interval_batch_size",
        ),
        (
            "_solve_inner_mesh",
            "_map_ride_partitioned",
            "requested_block_size",
            "statics.cell_block_size",
        ),
        (
            "solve_one_cell",
            "_map_branch_partitioned",
            "requested_block_size",
            "statics.branch_batch_size",
        ),
        (
            "envelope_core",
            "_map_ride_partitioned",
            "requested_block_size",
            "statics.cell_block_size",
        ),
    }

    observed = {
        route
        for route in routes
        if route[1]
        in {
            "_nbegm_ride_along_statics",
            "_NBEGMRideAlongStatics",
            "_map_ride_partitioned",
            "_map_branch_partitioned",
        }
        and route[2]
        in {
            "cell_block_size",
            "interval_batch_size",
            "branch_batch_size",
            "requested_block_size",
        }
    }

    assert observed == expected


def test_production_ride_and_branch_geometries_are_independent() -> None:
    """The measured 256x4 geometry is expressible without widening branches."""
    assert (
        fixed_width_map.FixedWidthMapGeometry(
            microtile_width=256,
            profile_window=256,
        )
        == nbegm._RIDE_MAP_GEOMETRY
    )
    assert (
        fixed_width_map.FixedWidthMapGeometry(
            microtile_width=4,
            profile_window=64,
        )
        == nbegm._BRANCH_MAP_GEOMETRY
    )
