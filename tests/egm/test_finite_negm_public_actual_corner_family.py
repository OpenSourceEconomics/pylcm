"""Public actual-corner family and scalar wrong-rule controls.

Scalar rule mutants discriminate the witness arithmetic; they are not
installed production mutants.
"""

import math

import pytest

from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_negm_actual_corner_witness import (
    actual_corner,
    node_budget_substitution,
)
from tests.egm.test_finite_negm_public_actual_corner import (
    assert_actual_corner,
    observe_public_corner,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@pytest.mark.parametrize("stock", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("enable_jit", [False, True])
def test_public_interior_corner_family(
    *, stock, runtime_checks, enable_jit, monkeypatch
):
    observed = observe_public_corner(
        stock=stock,
        runtime_checks=runtime_checks,
        enable_jit=enable_jit,
        monkeypatch=monkeypatch,
    )
    assert_actual_corner(observed=observed, stock=stock)


@pytest.mark.parametrize("stock", [0.0, 1.0])
def test_exact_passive_node_is_a_nondiscriminating_control(*, stock):
    """At one nonzero source weight the two arithmetic frames coincide."""
    expected = actual_corner(stock=stock)
    assert node_budget_substitution(stock=stock) == (expected.value, expected.marginal)


@pytest.mark.parametrize(
    "mutant",
    [
        "nodal_pair",
        "nodal_marginal",
        "source_floor",
        "true_W_at_actual_floor",
        "resource_chain_twice",
    ],
)
def test_scalar_oracle_distinguishes_specific_wrong_corner_rules(*, mutant):
    expected = actual_corner(resource_slope=2.0)
    nodal_value, nodal_marginal = node_budget_substitution(resource_slope=2.0)
    wrong = {
        "nodal_pair": (nodal_value, nodal_marginal),
        "nodal_marginal": (expected.value, nodal_marginal),
        "source_floor": (math.log(6.0), 2.0 / 6.0),
        # The contract intentionally holds W_B from source floors; it is not W(1).
        "true_W_at_actual_floor": (expected.value + 0.5 / 8.0, expected.marginal),
        "resource_chain_twice": (expected.value, 2.0 * expected.marginal),
    }[mutant]
    assert not math.isclose(
        wrong[0], expected.value, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(wrong[1], expected.marginal, rel_tol=0.0, abs_tol=1e-12)
