"""The ride knobs reach the fixed-width map, and the mesh decides what they buy.

`cell_block_size` and `interval_batch_size` are requests, not partitions: each
is coarsened to a multiple of the vector width and capped by the window that
covers its axis. On a short axis every request therefore admits to the same
partition, and a bit-identity test over such an axis compares a configuration
with itself. These tests assert the routing — that the requested value reaches
`map_partitioned` at the ride call sites — and state the admitted partition the
toy's mesh actually produces, so a mesh change that makes a knob inert is
visible rather than silent.

Bit identity across *distinct* admitted partitions is covered where the axis is
long enough to have them: `test_fixed_width_map.py` at the unit level and
`test_nbegm_partition_bit_identity.py` on the branch axis.
"""

from collections.abc import Mapping

import pytest

from _lcm.egm import fixed_width_map
from tests.test_models import nbegm_next_asset_cliff_toy as interval_toy
from tests.test_models import nbegm_ride_along_toy as cell_toy


def _record_partitions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    """Collect `(n_rows, requested, admitted)` for every fixed-width map call."""
    calls: list[tuple[int, int, int]] = []
    original = fixed_width_map.map_partitioned

    def spy(*, func, xs, requested_block_size):
        n_rows = fixed_width_map._leading_size(xs)
        window = min(
            fixed_width_map.PROFILE_WINDOW,
            fixed_width_map.enclosing_max_block_size(
                n_rows=n_rows, microtile_width=fixed_width_map.MICROTILE_WIDTH
            ),
        )
        calls.append(
            (
                n_rows,
                requested_block_size,
                fixed_width_map.admitted_block_size(
                    requested=requested_block_size,
                    max_block_size=window,
                    microtile_width=fixed_width_map.MICROTILE_WIDTH,
                ),
            )
        )
        return original(func=func, xs=xs, requested_block_size=requested_block_size)

    monkeypatch.setattr(fixed_width_map, "map_partitioned", spy)
    monkeypatch.setattr("_lcm.solution.nbegm.map_partitioned", spy)
    return calls


def _solve_cells(*, cell_block_size: int) -> Mapping[int, Mapping]:
    return cell_toy.build_model(
        variant="nbegm",
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        nbegm_overrides={"cell_block_size": cell_block_size},
    ).solve(params=cell_toy.build_params(), log_level="off")


def _solve_intervals(*, interval_batch_size: int) -> Mapping[int, Mapping]:
    return interval_toy.build_model(
        variant="nbegm", interval_batch_size=interval_batch_size
    ).solve(params=interval_toy.build_params(), log_level="off")


def test_cell_block_size_reaches_the_fixed_width_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requested cell block arrives at the map rather than being dropped."""
    calls = _record_partitions(monkeypatch)
    _solve_cells(cell_block_size=3)
    assert 3 in {requested for _n_rows, requested, _admitted in calls}


def test_interval_batch_size_reaches_the_fixed_width_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requested interval batch arrives at the map rather than being dropped."""
    calls = _record_partitions(monkeypatch)
    _solve_intervals(interval_batch_size=2)
    assert 2 in {requested for _n_rows, requested, _admitted in calls}


@pytest.mark.parametrize("requested", [0, 1, 3])
def test_a_short_ride_axis_admits_one_partition_whatever_is_requested(
    monkeypatch: pytest.MonkeyPatch, requested: int
) -> None:
    """Every request collapses to one partition on an axis shorter than the width.

    This is what makes a bit-identity test over this toy's ride mesh vacuous:
    the compared solves run the same partition. A toy whose mesh grew past the
    vector width would fail here, which is the signal to add the comparison.
    """
    calls = _record_partitions(monkeypatch)
    _solve_cells(cell_block_size=requested)
    admitted = {
        admitted
        for n_rows, _requested, admitted in calls
        if n_rows <= fixed_width_map.MICROTILE_WIDTH
    }
    assert admitted == {fixed_width_map.MICROTILE_WIDTH}
