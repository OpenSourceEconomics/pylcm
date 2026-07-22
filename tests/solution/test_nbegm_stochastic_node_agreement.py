"""NBEGM with a stochastic ride-along income node: brute agreement + splay invariance.

The income process node rides along the consumption--saving Euler axis; the continuation
weights the per-node child reads by the process's intrinsic transition probabilities.
Two properties are pinned:

- the weighted node expectation matches a dense brute-force solve that averages the
  action-aggregated next-period V over the same income nodes, and
- splaying that expectation into `lax.scan` blocks (`stochastic_node_batch_size > 0`)
  leaves the solved value function unchanged — it is a memory knob, not a result knob.
"""

import numpy as np
import pytest

from tests.conftest import X64_ENABLED
from tests.test_models import nbegm_stochastic_node_toy as toy

# Liquid nodes excluded from the brute comparison, restricting it to the interior band
# where both solvers are reliable:
# - the lowest nodes, where the coarse consumption grid and CRRA curvature make brute
#   force itself unreliable, and
# - the highest nodes, where the income-perturbed continuation wealth leaves the bounded
#   liquid grid: brute edge-clamps its next-period V lookup while NBEGM extrapolates
#   past its last endogenous savings point, so the two diverge at the top edge (a grid
#   artifact, not a solver disagreement — the splay-invariance test below confirms the
#   NBEGM machinery is internally consistent there).
_N_LOW_EDGE_NODES = 12
_N_HIGH_EDGE_NODES = 16

# The block reduction reorders the weighted-sum adds, so the splay match is to
# tolerance, tight in float64 and at single-precision scale in float32.
_INVARIANCE_RTOL = 1e-9 if X64_ENABLED else 1e-4
_INVARIANCE_ATOL = 1e-9 if X64_ENABLED else 1e-4


def _solve(variant: str, stochastic_node_batch_size: int = 0):
    model = toy.build_model(
        variant=variant, stochastic_node_batch_size=stochastic_node_batch_size
    )
    return model.solve(params=toy.build_params(), log_level="debug")


def test_nbegm_stochastic_node_matches_dense_brute_force():
    """NBEGM integrating a stochastic income node matches dense brute force.

    The child income node is distributed; the NBEGM continuation weights the per-node
    reads by the IID Gauss-Hermite probabilities, the same expectation brute force takes
    by averaging the action-aggregated next-period V over the income nodes. Agreement
    holds up to the brute solver's consumption-grid resolution across the interior
    liquid band, excluding the edge nodes where brute force is unreliable (low) or the
    continuation leaves the bounded grid (high).
    """
    nbegm_solution = _solve("nbegm")
    brute_solution = _solve("brute")
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["alive"])
        nbegm_V = np.asarray(nbegm_solution[period]["alive"])
        assert brute_V.shape == nbegm_V.shape
        interior = slice(_N_LOW_EDGE_NODES, -_N_HIGH_EDGE_NODES)
        np.testing.assert_allclose(
            nbegm_V[..., interior],
            brute_V[..., interior],
            atol=2e-2,
            rtol=2e-3,
            err_msg=f"period={period}",
        )


@pytest.mark.parametrize("stochastic_node_batch_size", [1, 2, 3, toy.N_INCOME_NODES])
def test_nbegm_stochastic_node_batch_size_leaves_value_function_unchanged(
    stochastic_node_batch_size,
):
    """Splaying the income-node expectation into blocks does not change the solved V.

    `stochastic_node_batch_size` only changes how the per-node continuation reads are
    scheduled and reduced, so the value function at every period matches the unsplayed
    `stochastic_node_batch_size=0` solve to tight numerical tolerance — including a
    block size (3) that does not divide the 5-node income mesh, and the boundary size
    equal to the mesh length.
    """
    reference = _solve("nbegm", stochastic_node_batch_size=0)
    splayed = _solve("nbegm", stochastic_node_batch_size=stochastic_node_batch_size)
    assert set(reference) == set(splayed)
    for period in sorted(reference):
        assert set(reference[period]) == set(splayed[period])
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(splayed[period][regime_name])
            assert ref_V.shape == got_V.shape
            np.testing.assert_allclose(
                got_V,
                ref_V,
                rtol=_INVARIANCE_RTOL,
                atol=_INVARIANCE_ATOL,
                err_msg=f"period={period}, regime={regime_name}",
            )
