"""Folding query-irrelevant stochastic dims out of the child carry read.

A child stochastic dimension folds out of the per-savings node loop when its
nodes cannot move the read: the child's resources map does not read its node
value, the child's carry rows share the state grid as abscissae, and no
per-node discrete-choice aggregation sits between interpolation and the
expectation. All three conditions are static model topology. Value-only
children (brute `GridSearch` terminal, the case-piece `BQSEGM`) publish
state-space carries read through an identity resources map, so every one of
their stochastic dims folds; endogenous-grid children (DC-EGM, NEGM) publish
per-row abscissae and keep every dim in the node loop. The fold is a
scheduling change only — the solved value function is unchanged to numerical
tolerance.
"""

import dataclasses

import numpy as np

import _lcm.egm.continuation as cont_mod
from tests.conftest import X64_ENABLED
from tests.solution import test_egm_process_states as dcegm_fixture
from tests.test_models import bqsegm_stochastic_node_toy as toy

_TOLERANCE = (
    {"rtol": 1e-9, "atol": 1e-9} if X64_ENABLED else {"rtol": 1e-4, "atol": 1e-4}
)


def _capture_child_reads(monkeypatch, solve) -> list:
    captured = []
    original = cont_mod._get_child_carry_reader

    def spy(**kwargs):
        captured.append(kwargs["read"])
        return original(**kwargs)

    monkeypatch.setattr(cont_mod, "_get_child_carry_reader", spy)
    solve()
    return captured


def _foldable_flag(read, state_name: str) -> bool:
    by_name = dict(
        zip(read.stochastic_state_names, read.foldable_stochastic_flags, strict=True)
    )
    return by_name[state_name]


def test_stochastic_dim_of_a_state_space_carry_child_is_foldable(monkeypatch):
    """A value-only child's income dim folds: its read is a state-space gather."""
    reads = _capture_child_reads(
        monkeypatch,
        lambda: toy.build_model(variant="bqsegm").solve(
            params=toy.build_params(), log_level="off"
        ),
    )
    with_income = [r for r in reads if "income" in r.stochastic_state_names]
    assert with_income
    assert all(_foldable_flag(read, "income") for read in with_income)


def test_stochastic_dim_of_an_endogenous_grid_child_is_not_foldable(monkeypatch):
    """A DC-EGM child's per-row abscissae keep every dim in the node loop."""
    reads = _capture_child_reads(
        monkeypatch,
        lambda: dcegm_fixture._get_model("dcegm", "iid").solve(
            params=dcegm_fixture._get_params("iid"), log_level="off"
        ),
    )
    with_income = [r for r in reads if "income" in r.stochastic_state_names]
    assert with_income
    assert not any(_foldable_flag(read, "income") for read in with_income)


def test_fold_leaves_value_function_unchanged(monkeypatch):
    """The fold only reschedules the expectation: V matches the unfolded read."""
    params = toy.build_params()
    folded = toy.build_model(variant="bqsegm").solve(params=params, log_level="off")

    def keep_unfolded(read):
        return dataclasses.replace(
            read,
            foldable_stochastic_flags=tuple(
                False for _ in read.foldable_stochastic_flags
            ),
        )

    original = cont_mod._get_child_carry_reader

    def unfolded_reader(**kwargs):
        return original(**{**kwargs, "read": keep_unfolded(kwargs["read"])})

    monkeypatch.setattr(cont_mod, "_get_child_carry_reader", unfolded_reader)
    unfolded = toy.build_model(variant="bqsegm").solve(params=params, log_level="off")

    for period in folded:
        for regime_name in folded[period]:
            np.testing.assert_allclose(
                np.asarray(folded[period][regime_name]),
                np.asarray(unfolded[period][regime_name]),
                err_msg=f"period={period} regime={regime_name}",
                **_TOLERANCE,
            )
