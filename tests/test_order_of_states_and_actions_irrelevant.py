"""State declaration order relabels array axes without changing solved values.

Canonical order keeps discrete states, continuous states, and actions in their
existing groups. Within a state group, declaration order is stable except for
explicitly sharded states, which lead. Execution widths never reorder axes.
"""

import numpy as np
import pytest

from lcm import ExecutionConfig, LinSpacedGrid, Model
from tests.conftest import assert_agrees_to_ulp
from tests.test_models.deterministic.ds_pension import RegimeId, get_model, get_params

_N_PERIODS = 5
_N_BOTH = 8
_PLAIN = LinSpacedGrid(start=0.0, stop=15.0, n_points=_N_BOTH)


def _solve(*, reverse_working_states=False, cell_width=1, **overrides):
    model = get_model(
        n_periods=_N_PERIODS, n_liquid=_N_BOTH, n_pension=_N_BOTH, **overrides
    )
    regimes = dict(model.user_regimes)
    if reverse_working_states:
        regimes["working"] = regimes["working"].replace(
            states=dict(reversed(tuple(regimes["working"].states.items())))
        )
    reordered = Model(
        regimes=regimes,
        ages=model.ages,
        regime_id_class=RegimeId,
        execution_config=ExecutionConfig(axis_widths={"cell": cell_width}),
    )
    return reordered.solve(params=get_params(), log_level="off").values


def _periods_with(*, solution, regime):
    return [period for period, regimes in solution.items() if regime in regimes]


def test_declaring_the_pension_grid_explicitly_changes_nothing():
    """An override with the same outcome-space definition preserves the result."""
    default = _solve()
    explicit = _solve(working_pension_grid=_PLAIN)
    periods = _periods_with(solution=default, regime="working")
    assert periods
    for period in periods:
        np.testing.assert_array_equal(
            np.asarray(explicit[period]["working"]),
            np.asarray(default[period]["working"]),
        )


@pytest.mark.parametrize("cell_width", [1, 3])
def test_reversed_working_state_declarations_transpose_values(cell_width):
    """The continuous-state declaration order fixes output axes at every width."""
    plain = _solve(working_pension_grid=_PLAIN)
    reordered = _solve(reverse_working_states=True, cell_width=cell_width)
    periods = _periods_with(solution=plain, regime="working")
    assert periods
    for period in periods:
        assert_agrees_to_ulp(
            got=reordered[period]["working"],
            expected=np.asarray(plain[period]["working"]).T,
            n_ulp=8,
        )


@pytest.mark.parametrize("regime", ["retired", "dead"])
def test_reordering_working_states_preserves_other_regimes(regime):
    """A state-axis permutation stays local to the regime whose order changed."""
    plain = _solve()
    reordered = _solve(reverse_working_states=True)
    periods = _periods_with(solution=plain, regime=regime)
    assert periods
    for period in periods:
        np.testing.assert_array_equal(reordered[period][regime], plain[period][regime])
