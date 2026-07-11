"""Integration test for the terminal collective-regime (E1) solve (slice 1b).

Wires the already-committed `collective_readout` core into the GridSearch
terminal kernel: a terminal `Regime(stakeholders=("f", "m"))` solved end-to-end
through the real processing + backward-induction path must produce a
value-function array carrying a trailing stakeholder axis, with each
stakeholder's value read off its OWN utility at the shared household argmax
(design doc `pylcm-extension-collective-regimes.md` §2, E1).

The economics mirror `test_collective_readout.test_terminal_e1_end_to_end_with_
real_utilities`: a binary work choice and a two-point wage state where the
household's joint argmax differs from either stakeholder's own preferred action
(at the low wage the wife's leisure taste dominates the joint objective; at the
high wage her working wins jointly).
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, categorical
from lcm.ages import AgeGrid
from lcm.regime import Regime
from lcm.typing import DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _utility_f(wage: FloatND, work: DiscreteAction) -> FloatND:
    """Wife: values her own leisure highly, also sees household consumption."""
    consumption = wage * work
    return consumption + 30.0 * (1.0 - work)


def _utility_m(wage: FloatND, work: DiscreteAction) -> FloatND:
    """Husband: values household consumption, indifferent to leisure."""
    consumption = wage * work
    return 2.0 * consumption


def test_terminal_collective_regime_solves_with_stakeholder_axis():
    regime = Regime(
        transition=None,  # terminal
        stakeholders=("f", "m"),
        states={"wage": LinSpacedGrid(start=10.0, stop=40.0, n_points=2)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility_f": _utility_f, "utility_m": _utility_m},
    )
    ages = AgeGrid(start=0, stop=2, step="Y")
    regimes = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes={"couple": regime}, derived_categoricals={}
        ),
        ages=ages,
        regime_names_to_ids=MappingProxyType({"couple": jnp.int32(0)}),
        enable_jit=False,
    )

    solution, _sim_policies, _divorce_flags = solve(
        flat_params=MappingProxyType({"couple": MappingProxyType({})}),
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )

    V = solution[0]["couple"]
    # V carries a trailing stakeholder axis: (wage grid = 2) x (stakeholders = 2).
    assert V.shape == (2, 2)

    # Each stakeholder's value is its OWN utility at the shared household argmax
    # of O = 1/2 (u_f + u_m):
    #   wage=10: a=0 -> u_f=30, u_m=0 (O=15);  a=1 -> u_f=10, u_m=20 (O=15) tie -> a=0.
    #   wage=40: a=0 -> u_f=30, u_m=0 (O=15);  a=1 -> u_f=40, u_m=80 (O=60)      -> a=1.
    np.testing.assert_allclose(np.asarray(V[:, 0]), [30.0, 40.0])  # wife (f)
    np.testing.assert_allclose(np.asarray(V[:, 1]), [0.0, 80.0])  # husband (m)
    # The two stakeholder value arrays genuinely differ (not the scalarization).
    assert not np.allclose(np.asarray(V[:, 0]), np.asarray(V[:, 1]))
