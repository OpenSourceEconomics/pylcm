"""DS-2026 Application 3 VFI Euler-error accuracy harness.

Application 3 of Dobrescu & Shanker (2026) is the discrete-housing model
(extended Fella 2014): log Cobb-Douglas utility over consumption and housing
services, a discrete housing stock with an own-vs-rent choice, a proportional
housing-adjustment cost, and a **Markov wage**. The paper's no-tax tables
(Table 4 / Table 7) report the VFI accuracy column as the mean `log10`
consumption Euler error along a simulated sample path. The brute/VFI (GridSearch)
solver solves this model locally, so the harness solves, simulates, and scores
the stochastic Euler equation along the working-regime path — the wage being
Markov, the Euler expectation is a transition-probability-weighted sum over
next-period wage nodes.

These tests run a single small solve at a time (asset grid <= 80, wage nodes
<= 5, shortened horizon) so they stay local-safe; the full paper grids at T=20
are a GPU/CI sweep.
"""

import numpy as np
import pandas as pd
import pytest

from benchmarks.ds_replication.app3_discrete_housing_accuracy import (
    _wage_nodes_and_transition,
    app3_vfi_accuracy_table,
    app3_vfi_euler_error,
    sample_path_euler_error,
)

# Local-safe horizon: shorter than the paper's T=20 so a single solve+simulate is
# fast, but long enough that the working-regime Euler equation has interior,
# non-switch points to score.
_LOCAL_N_PERIODS = 10
_LOCAL_N_SUBJECTS = 200


def test_app3_vfi_euler_error_is_finite_negative_and_in_vfi_band():
    """The VFI Euler error at a small grid is finite, negative, and VFI-plausible.

    The mean log10 consumption Euler error is a negative number (more negative =
    more accurate). The brute/VFI solver searches consumption on a discrete grid,
    so at these small grids the metric sits in a coarse-VFI band, roughly -3.0 to
    -0.5.
    """
    error = app3_vfi_euler_error(
        n_assets=40,
        n_wage_nodes=3,
        n_periods=_LOCAL_N_PERIODS,
        n_consumption=60,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert np.isfinite(error)
    assert -3.0 < error < -0.5


def test_app3_vfi_euler_error_stays_in_band_across_consumption_grids():
    """The VFI error stays in the coarse-VFI band across consumption-grid sizes.

    The grid-search solver chooses consumption on a discrete grid, so the
    consumption-grid resolution drives its accuracy. Both a coarse and a finer
    consumption grid land in the same plausible VFI band — the metric is a finite,
    negative number throughout (the monotone full-horizon refinement is a T=20
    GPU/CI-scale property, noisy at the short local horizon).
    """
    coarse = app3_vfi_euler_error(
        n_assets=40,
        n_wage_nodes=3,
        n_periods=_LOCAL_N_PERIODS,
        n_consumption=40,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    fine = app3_vfi_euler_error(
        n_assets=40,
        n_wage_nodes=3,
        n_periods=_LOCAL_N_PERIODS,
        n_consumption=120,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert -3.0 < coarse < -0.5
    assert -3.0 < fine < -0.5


def test_app3_vfi_accuracy_table_has_one_row_per_asset_grid():
    """The sweep returns one VFI Euler-error row per asset-grid size."""
    table = app3_vfi_accuracy_table(
        n_assets_grid=(30, 40),
        n_wage_nodes=3,
        n_periods=_LOCAL_N_PERIODS,
        n_consumption=60,
        n_subjects=_LOCAL_N_SUBJECTS,
        seed=0,
    )
    assert list(table.columns) == ["n_assets", "n_wage_nodes", "vfi_euler_error"]
    assert len(table) == 2
    assert table["vfi_euler_error"].between(-3.0, -0.5).all()


def test_sample_path_euler_error_weights_the_markov_wage_expectation():
    """A hand-built path reproduces its analytic transition-weighted Euler error.

    With two wage nodes whose transition row `P[wage_t] = (0.25, 0.75)` mixes a
    next-period consumption of 4 (at wage node 0) and 8 (at wage node 1) for the
    held housing level, the expected marginal utility is
    `0.25*alpha/4 + 0.75*alpha/8`, so
    `c_euler = alpha / (beta*(1+r) * E[u'])`. The single interior,
    non-housing-switch working-to-working transition reproduces
    `log10(|c_euler/c_t - 1|)` exactly.
    """
    alpha, beta, r = 0.77, 0.93, 0.06
    wage_nodes = np.array([-0.4, 0.4])
    # Source wage node 0; its transition row mixes the two next-period nodes.
    wage_transition = np.array([[0.25, 0.75], [0.5, 0.5]])
    c_next_node0, c_next_node1 = 4.0, 8.0
    expected_marginal = 0.25 * alpha / c_next_node0 + 0.75 * alpha / c_next_node1
    c_euler = alpha / (beta * (1.0 + r) * expected_marginal)
    c_t = c_euler / 1.1  # so |c_euler / c_t - 1| = 0.1

    sample_panel = pd.DataFrame(
        {
            "subject_id": [0, 0],
            "period": [0, 1],
            "regime_name": ["working", "working"],
            "assets": [10.0, 5.0],  # next assets > 0: interior, unconstrained
            "wage": [wage_nodes[0], wage_nodes[0]],
            "consumption": [c_t, 6.0],
            "housing": ["own_h2", "own_h2"],
            "housing_choice": ["own_h2", "own_h2"],  # no switch this period or next
        }
    )
    # The policy panel supplies `c_{t+1}(a'=5, own_h2, wage_j)` for each next node.
    policy_panel = pd.DataFrame(
        {
            "subject_id": [0, 1],
            "period": [1, 1],
            "regime_name": ["working", "working"],
            "assets": [5.0, 5.0],
            "wage": [wage_nodes[0], wage_nodes[1]],
            "consumption": [c_next_node0, c_next_node1],
            "housing": ["own_h2", "own_h2"],
            "housing_choice": ["own_h2", "own_h2"],
        }
    )
    error = sample_path_euler_error(
        sample_panel=sample_panel,
        policy_panel=policy_panel,
        wage_nodes=wage_nodes,
        wage_transition=wage_transition,
        interest_rate=r,
        discount_factor=beta,
        alpha=alpha,
    )
    assert error == pytest.approx(np.log10(0.1), abs=1e-9)


def test_sample_path_euler_error_drops_housing_switch_points():
    """A working-to-working transition that switches housing is not scored.

    The discrete housing-adjustment margin is a value-function kink, so a point
    where the held housing differs from the chosen housing is excluded — exactly
    as Application 1 excludes the work/retire switch. With every transition a
    switch, no point survives and the scorer raises.
    """
    wage_nodes = np.array([-0.4, 0.4])
    wage_transition = np.array([[0.5, 0.5], [0.5, 0.5]])
    sample_panel = pd.DataFrame(
        {
            "subject_id": [0, 0],
            "period": [0, 1],
            "regime_name": ["working", "working"],
            "assets": [10.0, 5.0],
            "wage": [wage_nodes[0], wage_nodes[0]],
            "consumption": [3.0, 6.0],
            "housing": ["rent", "rent"],
            "housing_choice": ["own_h1", "own_h1"],  # switches both periods
        }
    )
    policy_panel = pd.DataFrame(
        {
            "subject_id": [0],
            "period": [1],
            "regime_name": ["working"],
            "assets": [5.0],
            "wage": [wage_nodes[0]],
            "consumption": [6.0],
            "housing": ["own_h1"],
            "housing_choice": ["own_h1"],
        }
    )
    with pytest.raises(ValueError, match="No valid interior"):
        sample_path_euler_error(
            sample_panel=sample_panel,
            policy_panel=policy_panel,
            wage_nodes=wage_nodes,
            wage_transition=wage_transition,
        )


def test_sample_path_euler_error_with_taxes_uses_after_tax_marginal_return():
    """With taxes the scorer discounts by `1 + r - T'(a')`, not the gross `1 + r`.

    The with-tax budget is `R = (1 + r) a - T(a) + y + h`, so a unit of saving earns
    `1 + r - T'(a')` and the interior consumption Euler equation reads
    `u'(c_t) = beta (1 + r - tau_k) E[u'(c_{t+1})]`, with `tau_k` the marginal rate
    of the bracket holding `a'`. At `a' = 5.0` the bracket `[3.87, 6.97)` has
    `tau_k = 0.024`, so the implied consumption uses the return `1.06 - 0.024`.
    """
    alpha, beta, r = 0.77, 0.93, 0.06
    after_tax_return = 1.0 + r - 0.024  # bracket [3.87, 6.97) marginal rate
    wage_nodes = np.array([-0.4, 0.4])
    wage_transition = np.array([[0.25, 0.75], [0.5, 0.5]])
    c_next_node0, c_next_node1 = 4.0, 8.0
    expected_marginal = 0.25 * alpha / c_next_node0 + 0.75 * alpha / c_next_node1
    c_euler = alpha / (beta * after_tax_return * expected_marginal)
    c_t = c_euler / 1.1  # so |c_euler / c_t - 1| = 0.1

    sample_panel = pd.DataFrame(
        {
            "subject_id": [0, 0],
            "period": [0, 1],
            "regime_name": ["working", "working"],
            "assets": [10.0, 5.0],  # a' = 5.0 sits in bracket [3.87, 6.97)
            "wage": [wage_nodes[0], wage_nodes[0]],
            "consumption": [c_t, 6.0],
            "housing": ["own_h2", "own_h2"],
            "housing_choice": ["own_h2", "own_h2"],
        }
    )
    policy_panel = pd.DataFrame(
        {
            "subject_id": [0, 1],
            "period": [1, 1],
            "regime_name": ["working", "working"],
            "assets": [5.0, 5.0],
            "wage": [wage_nodes[0], wage_nodes[1]],
            "consumption": [c_next_node0, c_next_node1],
            "housing": ["own_h2", "own_h2"],
            "housing_choice": ["own_h2", "own_h2"],
        }
    )
    error = sample_path_euler_error(
        sample_panel=sample_panel,
        policy_panel=policy_panel,
        wage_nodes=wage_nodes,
        wage_transition=wage_transition,
        interest_rate=r,
        discount_factor=beta,
        alpha=alpha,
        use_taxes=True,
    )
    assert error == pytest.approx(np.log10(0.1), abs=1e-9)


def test_sample_path_euler_error_with_taxes_excludes_tax_notch_points():
    """A point whose next assets sit on a tax-bracket boundary is not scored.

    At a bracket boundary the capital-income tax's level jump makes `T'` one-sided
    or undefined, so the smooth Euler equality does not hold there. With the single
    transition landing exactly on the boundary `a' = 6.97`, no point survives the
    with-tax filter and the scorer raises.
    """
    wage_nodes = np.array([-0.4, 0.4])
    wage_transition = np.array([[0.5, 0.5], [0.5, 0.5]])
    sample_panel = pd.DataFrame(
        {
            "subject_id": [0, 0],
            "period": [0, 1],
            "regime_name": ["working", "working"],
            "assets": [10.0, 6.97],  # a' = 6.97 is the bracket boundary
            "wage": [wage_nodes[0], wage_nodes[0]],
            "consumption": [3.0, 6.0],
            "housing": ["own_h2", "own_h2"],
            "housing_choice": ["own_h2", "own_h2"],
        }
    )
    policy_panel = pd.DataFrame(
        {
            "subject_id": [0],
            "period": [1],
            "regime_name": ["working"],
            "assets": [6.97],
            "wage": [wage_nodes[0]],
            "consumption": [6.0],
            "housing": ["own_h2"],
            "housing_choice": ["own_h2"],
        }
    )
    with pytest.raises(ValueError, match="No valid interior"):
        sample_path_euler_error(
            sample_panel=sample_panel,
            policy_panel=policy_panel,
            wage_nodes=wage_nodes,
            wage_transition=wage_transition,
            use_taxes=True,
        )


def test_wage_transition_rows_are_probabilities():
    """The Rouwenhorst wage transition matrix has rows summing to one."""
    _nodes, transition = _wage_nodes_and_transition(n_wage_nodes=5)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-10)
