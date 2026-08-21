"""The ASV benchmark models build against the public solver API.

A benchmark that cannot construct its model wastes a scheduled GPU run and
reports nothing, so construction is checked here — cheaply, at grid sizes far
below the benchmark's own — rather than on the benchmark runner.
"""

import pytest

from lcm.model import Model
from tests.conftest import EXACT_KERNEL_SKIP_REASON

_NEEDS_KERNEL = pytest.mark.requires_exact_affine_kernel(
    reason=EXACT_KERNEL_SKIP_REASON
)

_BENCHMARK_MODULES = pytest.importorskip("benchmarks.asv.bench_iskhakov_et_al_2017")


@pytest.mark.parametrize(
    "solver",
    [
        "brute_force",
        pytest.param(
            "dcegm",
            marks=pytest.mark.requires_exact_affine_kernel(
                reason=EXACT_KERNEL_SKIP_REASON
            ),
        ),
    ],
)
def test_make_model_and_params_constructs_a_model(solver: str) -> None:
    """Both benchmark solver variants build a `Model` from tiny grids."""
    model, _params = _BENCHMARK_MODULES._make_model_and_params(
        wealth_n_points=5,
        consumption_n_points=5,
        solver=solver,
    )
    assert isinstance(model, Model)
