"""Public mixed-corner family and scalar wrong-rule controls.

The public read must blend each kind over the passive stencil before choosing
at the actual query. The scalar rule alternatives discriminate the witness
arithmetic; they are not installed production mutants. The nodewise order is
checked against the exact child optimum: it credits a different choice on
each source row and overstates the optimum, while the blended read does not.
"""

from fractions import Fraction

import pytest

from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_negm_mixed_corner_witness import (
    Pair,
    actual_corner,
    blend_then_choose,
    exact_optimum,
    nodewise_alternative,
)
from tests.egm.test_finite_negm_public_mixed_corner import (
    assert_blended_mixed,
    observe_mixed,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.parametrize("stock", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_public_mixed_family(*, stock, runtime_checks, enable_jit, monkeypatch):
    observed = observe_mixed(
        stock=stock,
        runtime_checks=runtime_checks,
        enable_jit=enable_jit,
        monkeypatch=monkeypatch,
    )
    assert_blended_mixed(observed=observed, stock=stock, runtime_checks=runtime_checks)


@pytest.mark.parametrize("stock", [0.0, 1.0])
def test_public_source_node_control(*, stock, monkeypatch):
    observed = observe_mixed(
        stock=stock, runtime_checks=True, enable_jit=True, monkeypatch=monkeypatch
    )
    assert_blended_mixed(observed=observed, stock=stock, runtime_checks=True)


@pytest.mark.parametrize(
    "mutant",
    [
        "nodewise",
        "ordinary_channel",
        "correct_value_wrong_marginal",
    ],
)
def test_literal_reference_kills_separated_wrong_rules(*, mutant):
    stock = Fraction(1, 2)
    reference = blend_then_choose(stock=stock)
    wrong = {
        "nodewise": nodewise_alternative(stock=stock),
        "ordinary_channel": Pair(Fraction(35, 8), Fraction(5, 8)),
        "correct_value_wrong_marginal": Pair(reference.value, Fraction(1, 2)),
    }[mutant]
    assert wrong != reference


def test_witness_values_at_half_stock():
    assert blend_then_choose(stock=Fraction(1, 2)) == Pair(
        Fraction(39, 8), Fraction(5, 8)
    )
    assert nodewise_alternative(stock=Fraction(1, 2)) == Pair(
        Fraction(47, 8), Fraction(1, 2)
    )
    assert actual_corner(stock=Fraction(1, 2), outer=None) == Pair(
        Fraction(39, 8), Fraction(5, 8)
    )


@pytest.mark.parametrize("stock", [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
def test_nodewise_order_overstates_the_exact_optimum(*, stock):
    """Choosing on each row then blending exceeds what any single action attains."""
    assert nodewise_alternative(stock=stock).value > exact_optimum(stock=stock).value


@pytest.mark.parametrize("stock", [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)])
def test_blended_read_does_not_exceed_the_exact_optimum(*, stock):
    assert blend_then_choose(stock=stock).value <= exact_optimum(stock=stock).value


@pytest.mark.parametrize(
    "stock", [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
)
def test_nodewise_order_dominates_blended_read_by_jensen(*, stock):
    """Row maxima blended are never below the blend of one chosen kind."""
    assert (
        nodewise_alternative(stock=stock).value >= blend_then_choose(stock=stock).value
    )


def test_nodewise_order_strictly_dominates_at_the_witness():
    stock = Fraction(1, 2)
    assert (
        nodewise_alternative(stock=stock).value > blend_then_choose(stock=stock).value
    )
