"""NBEGM's branch commit stride does not change the solution.

Every iteration evaluates the same static branch window. `branch_batch_size`
changes only how many completed rows are committed before the loop advances, so a
smaller stride adds iterations without reducing the fixed workspace.
"""

from collections.abc import Mapping

from tests.conftest import assert_agrees_to_ulp
from tests.test_models import nbegm_ride_discrete_toy as toy

# This solve-level assertion retains the shared ULP diagnostic and its helpful
# period/regime context. `test_nbegm_partition_bit_identity.py` pins the stronger
# fixed-window guarantee that every published floating value is bit-identical.
_PARTITION_ULP = 32


def _solve(*, branch_batch_size: int) -> Mapping[int, Mapping]:
    model = toy.build_model(
        variant="nbegm",
        n_liquid=40,
        liquid_max=30.0,
        n_savings=60,
        savings_max=28.0,
        n_consumption=40,
        action_in_costate=True,
        action_in_utility=True,
        action_in_regime_transition=True,
        branch_batch_size=branch_batch_size,
    )
    return model.solve(params=toy.build_params(), log_level="debug")


def test_branch_commit_stride_one_matches_default() -> None:
    """The branch commit stride preserves `V` across all branch dependencies."""
    default = _solve(branch_batch_size=0)
    stride_one = _solve(branch_batch_size=1)
    assert default.keys() == stride_one.keys()
    for period in default:
        for regime in default[period]:
            assert_agrees_to_ulp(
                stride_one[period][regime],
                default[period][regime],
                n_ulp=_PARTITION_ULP,
                err_msg=f"period={period} regime={regime}",
            )
