"""Folding query-irrelevant stochastic dims out of the child carry read.

A child stochastic dimension folds out of the per-savings node loop when its
nodes cannot move the read: the child's resources map does not read its node
value, the child's carry rows share the state grid as abscissae, and no
per-node discrete-choice aggregation sits between interpolation and the
expectation. All three conditions are static model topology. Value-only
children (brute `GridSearch` terminal, the case-piece `NBEGM`) publish
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
from tests.test_models import nbegm_stochastic_node_toy as toy

_TOL = 1e-9 if X64_ENABLED else 1e-4


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
        lambda: toy.build_model(variant="nbegm").solve(
            params=toy.build_params(), log_level="off"
        ),
    )
    with_income = [r for r in reads if "income" in r.stochastic_state_names]
    assert with_income
    assert all(_foldable_flag(read, "income") for read in with_income)


def test_stochastic_dim_of_an_endogenous_grid_child_is_not_foldable(monkeypatch):
    """A DC-EGM child's per-row abscissae keep every dim in the node loop.

    The spy observes reads at trace time, so the model is built fresh
    (bypassing the fixture cache) — a cached model whose solve was already
    traced earlier in the process would re-run without tracing.
    """
    reads = _capture_child_reads(
        monkeypatch,
        lambda: dcegm_fixture._get_model.__wrapped__("dcegm", "iid").solve(
            params=dcegm_fixture._get_params("iid"), log_level="off"
        ),
    )
    with_income = [r for r in reads if "income" in r.stochastic_state_names]
    assert with_income
    assert not any(_foldable_flag(read, "income") for read in with_income)


def test_fold_leaves_value_function_unchanged(monkeypatch):
    """The fold only reschedules the expectation: V matches the unfolded read."""
    params = toy.build_params()
    folded = toy.build_model(variant="nbegm").solve(params=params, log_level="off")

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
    unfolded = toy.build_model(variant="nbegm").solve(params=params, log_level="off")

    for period in folded:
        for regime_name in folded[period]:
            np.testing.assert_allclose(
                np.asarray(folded[period][regime_name]),
                np.asarray(unfolded[period][regime_name]),
                rtol=_TOL,
                atol=_TOL,
                err_msg=f"period={period} regime={regime_name}",
            )


def test_foldable_smooth_read_bypasses_the_node_loop(monkeypatch):
    """A foldable dim of a smooth-valued child skips the per-node expectation.

    The folded dims' expectation is pre-applied to the carry rows once per
    cell, so the per-savings read is a single interpolation. A child that
    publishes value-jump locations keeps its node loop (the fold would
    average across each node's cliff), so only reads of breakpoint-free
    carries must have `income` folded away.
    """
    calls = []
    original = cont_mod._expect_over_stochastic_nodes

    def spy(**kwargs):
        calls.append(
            (
                kwargs["read"].stochastic_state_names,
                kwargs["carry"].breakpoints is not None,
            )
        )
        return original(**kwargs)

    monkeypatch.setattr(cont_mod, "_expect_over_stochastic_nodes", spy)
    toy.build_model(variant="nbegm").solve(params=toy.build_params(), log_level="off")
    smooth_calls = [names for names, has_jumps in calls if not has_jumps]
    jumped_calls = [names for names, has_jumps in calls if has_jumps]
    assert all("income" not in names for names in smooth_calls)
    assert all("income" in names for names in jumped_calls)


def _solve_jump_variant(**kwargs):
    return toy.build_model(variant="nbegm", tax_kind="jump", **kwargs).solve(
        params=toy.build_params(tax_lump=1.5), log_level="off"
    )


def test_breakpoint_independent_dim_folds_under_published_topology(monkeypatch):
    """A dim no jump source reads folds out of a jump-bearing carry's read.

    The cliff sits on the child's own liquid axis, so the income node cannot
    move the published jump preimages: every income row shares the duplicated
    abscissae and the expectation pre-folds into the carry exactly as for a
    smooth carry. The jump-bearing read therefore never loops the income
    nodes.
    """
    calls = []
    original = cont_mod._expect_over_stochastic_nodes

    def spy(**kwargs):
        calls.append(
            (
                kwargs["read"].stochastic_state_names,
                kwargs["carry"].breakpoints is not None,
            )
        )
        return original(**kwargs)

    jump_carry_reads = []
    original_reader = cont_mod._get_child_carry_reader

    def reader_spy(**kwargs):
        if kwargs["carry"].breakpoints is not None:
            jump_carry_reads.append(True)
        return original_reader(**kwargs)

    monkeypatch.setattr(cont_mod, "_expect_over_stochastic_nodes", spy)
    monkeypatch.setattr(cont_mod, "_get_child_carry_reader", reader_spy)
    _solve_jump_variant()

    assert jump_carry_reads, "no jump-bearing carry was read"
    jumped_calls = [names for names, has_jumps in calls if has_jumps]
    assert all("income" not in names for names in jumped_calls)


def test_fold_leaves_value_function_unchanged_under_published_topology(monkeypatch):
    """Folding a breakpoint-independent dim reproduces the unfolded V.

    The rows share abscissae and jump locations along the folded dim, so the
    fold commutes with the read wherever the value read's monotone slope
    limiter is inactive. Where the limiter binds (near the published jumps)
    the folded and per-node reads are different valid interpolants of the
    same one-sided data and may differ at interpolation-error order — far
    below the solver's accuracy frontier, hence the loose tolerance.
    """
    folded = _solve_jump_variant()

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
    unfolded = _solve_jump_variant()

    for period in folded:
        for regime_name in folded[period]:
            np.testing.assert_allclose(
                np.asarray(folded[period][regime_name]),
                np.asarray(unfolded[period][regime_name]),
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"period={period} regime={regime_name}",
            )
