"""Periods whose exit-query graphs trace the same program share one lowering.

The formation toy lives through three periods before its terminal one. The first
two read a living next period and so trace the same chunk program; the third
reads the terminal regimes and traces another. A single who marries is paid a
gift, an exit adjustment the engine binds anew in every period.
"""

from collections import Counter
from itertools import count

import numpy as np
import pytest

from _lcm import exit_queries
from _lcm.solution import backward_induction as backward
from lcm.shocks import QuasiMonteCarlo
from tests.conftest import assert_agrees_to_ulp
from tests.solution.test_formation_read_classes import _model, _solve

_ROOTS = ("couple", "single_f", "single_m")


def _four_period_values() -> dict[str, np.ndarray]:
    return _solve(_model(nodes=QuasiMonteCarlo(n=8, seed=0), n_periods=4, gift=True))


def test_periods_with_one_query_structure_lower_their_chunk_program_once(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each root lowers one chunk program per query structure, not per period."""
    lowered: Counter[str] = Counter()
    lower = backward._lower_resolved_candidate

    def record(*, resolved, label, **kwargs):
        regime = label.split(" ", 1)[0]
        if resolved.name == "_chunks" and regime in _ROOTS:
            lowered[regime] += 1
        return lower(resolved=resolved, label=label, **kwargs)

    monkeypatch.setattr(backward, "_lower_resolved_candidate", record)
    _four_period_values()
    assert lowered == dict.fromkeys(_ROOTS, 2)


def test_a_shared_chunk_program_reproduces_each_period_s_own(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sharing a lowering leaves every period's values where its own trace puts them."""
    shared = _four_period_values()
    tokens = count()
    monkeypatch.setattr(
        exit_queries, "_traced_structure", lambda **_: next(tokens), raising=False
    )
    own = _four_period_values()
    assert_agrees_to_ulp(
        got=np.concatenate([shared[key].ravel() for key in sorted(shared)]),
        expected=np.concatenate([own[key].ravel() for key in sorted(own)]),
        n_ulp=4,
    )
