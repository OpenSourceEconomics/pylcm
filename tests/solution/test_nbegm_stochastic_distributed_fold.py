"""NBEGM stochastic-node fold stays correct when a fixed ride state is distributed.

A model-level distributed ride state (`kind`) is sharded off the carry's leading
axis, so the continuation reads a carry with that axis peeled. The stochastic
income node's intrinsic-weight fold must target the income axis *in that peeled
carry*, not its pre-peel position. Because the income node (5 nodes) and `kind`
(2) differ in size, a mis-targeted fold axis is a hard broadcasting error, so the
distributed solve must both build and reproduce the non-distributed value exactly.
"""

import numpy as np

from tests.test_models import nbegm_stochastic_node_toy as toy


def test_stochastic_fold_is_invariant_to_distributing_a_ride_state():
    """Distributing the fixed `kind` ride state leaves the solved value unchanged."""
    params = toy.build_params(with_kind=True)
    plain = toy.build_model(
        variant="nbegm", with_kind=True, n_periods=4, n_liquid=24, n_savings=32
    ).solve(params=params, log_level="off")
    distributed = toy.build_model(
        variant="nbegm",
        with_kind=True,
        distributed_kind=True,
        n_periods=4,
        n_liquid=24,
        n_savings=32,
    ).solve(params=params, log_level="off")
    for period in plain:
        if "alive" not in plain[period]:
            continue
        plain_v = np.asarray(plain[period]["alive"])
        # Distributing `kind` sorts it to the leading discrete axis, so the value
        # is `(kind, income, liquid)` where the plain solve is `(income, kind,
        # liquid)`; align the two before the exact-value comparison.
        distributed_v = np.moveaxis(np.asarray(distributed[period]["alive"]), 0, 1)
        np.testing.assert_array_equal(distributed_v, plain_v)
