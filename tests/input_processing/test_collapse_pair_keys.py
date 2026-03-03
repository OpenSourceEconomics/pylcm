"""Tests for collapse_pair_keys."""

from types import MappingProxyType

import pytest

from lcm.input_processing.params_processing import collapse_pair_keys


@pytest.fixture
def identical_boundaries():
    """Template where all boundaries from 'working' have identical params."""
    return MappingProxyType(
        {
            "working": MappingProxyType(
                {
                    "utility": MappingProxyType(
                        {"risk_aversion": float, "wage": float}
                    ),
                    "H": MappingProxyType({"discount_factor": float}),
                }
            ),
            "working_to_working": MappingProxyType(
                {
                    "next_wealth": MappingProxyType({"interest_rate": float}),
                }
            ),
            "working_to_retired": MappingProxyType(
                {
                    "next_wealth": MappingProxyType({"interest_rate": float}),
                }
            ),
            "retired": MappingProxyType(
                {
                    "utility": MappingProxyType({"risk_aversion": float}),
                }
            ),
        }
    )


@pytest.fixture
def differing_boundaries():
    """Template where boundaries from 'growing' have different params."""
    return MappingProxyType(
        {
            "growing": MappingProxyType(
                {
                    "utility": MappingProxyType({"risk_aversion": float}),
                    "H": MappingProxyType({"discount_factor": float}),
                }
            ),
            "growing_to_growing": MappingProxyType(
                {
                    "next_wealth": MappingProxyType({"growth_rate": float}),
                }
            ),
            "growing_to_stable": MappingProxyType(
                {
                    "next_wealth": MappingProxyType({"tax_rate": float}),
                }
            ),
            "stable": MappingProxyType(
                {
                    "utility": MappingProxyType({"risk_aversion": float}),
                }
            ),
        }
    )


def testidentical_boundaries_collapsed(identical_boundaries):
    result = collapse_pair_keys(identical_boundaries)

    # Pair keys should be gone
    assert "working_to_working" not in result
    assert "working_to_retired" not in result

    # Transition params should be merged into source regime
    assert "next_wealth" in result["working"]
    assert result["working"]["next_wealth"]["interest_rate"] is float

    # Non-transition params preserved
    assert "utility" in result["working"]
    assert "H" in result["working"]

    # Other regimes unchanged
    assert "retired" in result
    assert "utility" in result["retired"]


def testdiffering_boundaries_kept(differing_boundaries):
    result = collapse_pair_keys(differing_boundaries)

    # Pair keys should remain because params differ
    assert "growing_to_growing" in result
    assert "growing_to_stable" in result

    # Source regime should not have transition params merged in
    assert "next_wealth" not in result["growing"]

    # Pair key contents preserved
    assert result["growing_to_growing"]["next_wealth"]["growth_rate"] is float
    assert result["growing_to_stable"]["next_wealth"]["tax_rate"] is float


def test_no_pair_keys():
    """Template with no pair keys should be returned unchanged."""
    template = MappingProxyType(
        {
            "regime_0": MappingProxyType({"fun_0": MappingProxyType({"arg_0": float})}),
            "regime_1": MappingProxyType({"fun_0": MappingProxyType({"arg_0": float})}),
        }
    )
    result = collapse_pair_keys(template)
    assert set(result.keys()) == {"regime_0", "regime_1"}


def test_mixed_some_collapsible_some_not():
    """Template with two source regimes: one collapsible, one not."""
    template = MappingProxyType(
        {
            "alpha": MappingProxyType({"utility": MappingProxyType({"x": float})}),
            # alpha's boundaries are identical -> collapsible
            "alpha_to_alpha": MappingProxyType(
                {"next_s": MappingProxyType({"rate": float})}
            ),
            "alpha_to_beta": MappingProxyType(
                {"next_s": MappingProxyType({"rate": float})}
            ),
            "beta": MappingProxyType({"utility": MappingProxyType({"x": float})}),
            # beta's boundaries differ -> not collapsible
            "beta_to_alpha": MappingProxyType(
                {"next_s": MappingProxyType({"rate": float})}
            ),
            "beta_to_beta": MappingProxyType(
                {"next_s": MappingProxyType({"factor": float})}
            ),
        }
    )
    result = collapse_pair_keys(template)

    # alpha's pair keys collapsed
    assert "alpha_to_alpha" not in result
    assert "alpha_to_beta" not in result
    assert "next_s" in result["alpha"]

    # beta's pair keys kept
    assert "beta_to_alpha" in result
    assert "beta_to_beta" in result
    assert "next_s" not in result["beta"]


def test_single_pair_key_collapsed():
    """A source regime with only one boundary should also be collapsed."""
    template = MappingProxyType(
        {
            "source": MappingProxyType({"utility": MappingProxyType({"x": float})}),
            "source_to_target": MappingProxyType(
                {"next_s": MappingProxyType({"rate": float})}
            ),
            "target": MappingProxyType({"utility": MappingProxyType({"x": float})}),
        }
    )
    result = collapse_pair_keys(template)

    assert "source_to_target" not in result
    assert "next_s" in result["source"]
    assert result["source"]["next_s"]["rate"] is float


@pytest.mark.parametrize(
    ("a_params", "b_params", "should_collapse"),
    [
        # Same function, same params -> collapse
        (
            {"next_s": {"rate": float}},
            {"next_s": {"rate": float}},
            True,
        ),
        # Same function, different param names -> no collapse
        (
            {"next_s": {"rate": float}},
            {"next_s": {"factor": float}},
            False,
        ),
        # Same function, different param types -> no collapse
        (
            {"next_s": {"rate": float}},
            {"next_s": {"rate": int}},
            False,
        ),
        # Different function names -> no collapse
        (
            {"next_s": {"rate": float}},
            {"next_t": {"rate": float}},
            False,
        ),
        # Extra function in one -> no collapse
        (
            {"next_s": {"rate": float}},
            {"next_s": {"rate": float}, "next_t": {"x": float}},
            False,
        ),
    ],
)
def test_collapse_structural_comparison(a_params, b_params, should_collapse):
    """Verify structural comparison logic for various param combinations."""
    template = MappingProxyType(
        {
            "src": MappingProxyType({"utility": MappingProxyType({"x": float})}),
            "src_to_a": MappingProxyType(
                {k: MappingProxyType(v) for k, v in a_params.items()}
            ),
            "src_to_b": MappingProxyType(
                {k: MappingProxyType(v) for k, v in b_params.items()}
            ),
            "a": MappingProxyType({}),
            "b": MappingProxyType({}),
        }
    )
    result = collapse_pair_keys(template)

    if should_collapse:
        assert "src_to_a" not in result
        assert "src_to_b" not in result
    else:
        assert "src_to_a" in result
        assert "src_to_b" in result
