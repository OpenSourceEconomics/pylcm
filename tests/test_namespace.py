"""The params-qname depth grammar matches real flat param names.

A flat param qname is a `__`-joined tree path. `ParamsQnameDepth` names the
depth (number of tree levels) of each pattern the params machinery classifies
by length. These tests pin each named depth to a concrete qname of that
pattern, so a future change to one without the other fails loudly.
"""

import pytest
from dags.tree import qname_from_tree_path, tree_path_from_qname

from _lcm.utils.namespace import ParamsQnameDepth


@pytest.mark.parametrize(
    ("tree_path", "expected_depth"),
    [
        (
            ("retirement", "labor_income", "slope"),
            ParamsQnameDepth.REGIME__FUNC__PARAM,
        ),
        (
            ("retirement", "retired", "next_health", "rate"),
            ParamsQnameDepth.REGIME__TARGETREGIME__FUNC__PARAM,
        ),
        (
            ("retired", "next_health", "rate"),
            ParamsQnameDepth.TARGETREGIME__FUNC__PARAM,
        ),
    ],
)
def test_params_qname_depth_matches_real_qname(
    tree_path: tuple[str, ...], expected_depth: int
) -> None:
    qname = qname_from_tree_path(tree_path)
    assert len(tree_path_from_qname(qname)) == expected_depth


def test_full_qname_function_and_per_target_depths_differ_by_one() -> None:
    """Inserting the target regime takes a plain function param one level
    deeper than its coarse, target-blind form."""
    assert (
        ParamsQnameDepth.REGIME__TARGETREGIME__FUNC__PARAM
        == ParamsQnameDepth.REGIME__FUNC__PARAM + 1
    )


def test_stripping_the_regime_prefix_preserves_per_target_depth() -> None:
    """The within-regime per-target qname is the full per-target qname minus
    its regime prefix — one level shallower."""
    assert (
        ParamsQnameDepth.TARGETREGIME__FUNC__PARAM
        == ParamsQnameDepth.REGIME__TARGETREGIME__FUNC__PARAM - 1
    )
