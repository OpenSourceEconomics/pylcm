"""The ride axes publish the same bits however finely they are partitioned.

`cell_block_size` streams the flattened ride mesh and `interval_batch_size`
streams the per-interval continuation read. Neither changes an operation or an
operand order, so each publishes the same real number at every setting. Bit
identity is the stronger claim that it is also the same *floating* number,
which holds only once the partition stops being part of the compilation key —
so that every setting runs one executable rather than a differently vectorized
one.

`test_nbegm_ride_along_agreement.py` and `test_nbegm_interval_batch.py` state
the same invariance bounded in ULP, which a body outside the fixed-width
construction still needs.
"""

from collections.abc import Mapping

import numpy as np
import pytest

from tests.test_models import nbegm_next_asset_cliff_toy as interval_toy
from tests.test_models import nbegm_ride_along_toy as cell_toy

_ALIVE = "alive"


def _solve_cells(*, cell_block_size: int) -> Mapping[int, Mapping]:
    model = cell_toy.build_model(
        variant="nbegm",
        n_liquid=120,
        liquid_max=30.0,
        n_savings=180,
        savings_max=28.0,
        nbegm_overrides={"cell_block_size": cell_block_size},
    )
    return model.solve(params=cell_toy.build_params(), log_level="debug")


def _solve_intervals(*, interval_batch_size: int) -> Mapping[int, Mapping]:
    model = interval_toy.build_model(
        variant="nbegm", interval_batch_size=interval_batch_size
    )
    return model.solve(params=interval_toy.build_params(), log_level="debug")


def _assert_identical(
    partitioned: Mapping[int, Mapping], whole: Mapping[int, Mapping]
) -> None:
    assert partitioned.keys() == whole.keys()
    compared = 0
    for period in whole:
        if _ALIVE not in whole[period]:
            continue
        np.testing.assert_array_equal(
            np.asarray(partitioned[period][_ALIVE]),
            np.asarray(whole[period][_ALIVE]),
            err_msg=f"period={period}",
        )
        compared += 1
    assert compared > 0, "no period was compared"


@pytest.mark.parametrize("cell_block_size", [1, 3])
def test_cell_partition_publishes_bit_identical_values(cell_block_size: int) -> None:
    """Blocking the ride mesh leaves every published value bit for bit equal."""
    _assert_identical(
        _solve_cells(cell_block_size=cell_block_size),
        _solve_cells(cell_block_size=0),
    )


@pytest.mark.parametrize("interval_batch_size", [1, 2])
def test_interval_partition_publishes_bit_identical_values(
    interval_batch_size: int,
) -> None:
    """Chunking the interval read leaves every published value bit for bit equal."""
    _assert_identical(
        _solve_intervals(interval_batch_size=interval_batch_size),
        _solve_intervals(interval_batch_size=0),
    )
