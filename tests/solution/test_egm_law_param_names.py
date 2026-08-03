"""The EGM kernels read law parameters the modeller names, not names they pick.

A kernel that inverts an Euler equation needs specific economic quantities out
of a law of motion — a gross return, a labour income, a pension match. Which
*parameter* carries each is the modeller's choice, exactly as which state fills
the liquid role is. Both are declared on the solver.

Renaming a parameter is a relabelling: it cannot move the solution. Each case
below solves one model twice, once under the default spellings and once under
deliberately different ones, and requires the two to be identical. Equality, not
a tolerance — the two runs execute the same arithmetic on the same nodes, so any
difference at all would mean a parameter reached the wrong slot.
"""

import numpy as np
import pytest
from dags import rename_arguments

from lcm.solvers import EGM, TwoAssetEGM
from tests.solution.test_egm_continuation_grid_provenance import (
    _A_GRID,
    _B_GRID,
    _CONSUMPTION_GRID,
    _N_PERIODS,
    _SAVINGS_GRID,
)
from tests.test_models.deterministic import ds_pension as ds
from tests.test_models.deterministic.ds_pension import get_model, get_params

# The modeller's spellings, deliberately unlike the kernel's role names.
_RENAMED = {
    "return_liquid": "gross_return",
    "return_pension": "fund_return",
    "match_rate": "employer_match",
    "wage": "labour_income",
    "retirement_income": "pension_benefit",
    "pension_payout_return": "payout_return",
}


def _renamed_laws():
    """The DS pension laws with every parameter renamed, behaviour untouched."""
    return {
        "next_liquid_working": rename_arguments(
            ds.next_liquid_working, mapper=_RENAMED
        ),
        "next_liquid_retiring": rename_arguments(
            ds.next_liquid_retiring, mapper=_RENAMED
        ),
        "next_pension_working": rename_arguments(
            ds.next_pension_working, mapper=_RENAMED
        ),
        "next_liquid_retired": rename_arguments(
            ds.next_liquid_retired, mapper=_RENAMED
        ),
    }


def _rename_params(params):
    """Rewrite every law's leaf names in a params tree, leaving the shape alone."""
    if not isinstance(params, dict):
        return params
    return {
        _RENAMED.get(key, key): _rename_params(value) for key, value in params.items()
    }


def _default_solvers():
    return {
        "working": TwoAssetEGM(
            a_grid=_A_GRID, b_grid=_B_GRID, consumption_grid=_CONSUMPTION_GRID
        ),
        "retired": EGM(savings_grid=_SAVINGS_GRID),
    }


def _renamed_solvers():
    """The same solvers, told which parameter fills each of their roles."""
    return {
        "working": TwoAssetEGM(
            a_grid=_A_GRID,
            b_grid=_B_GRID,
            consumption_grid=_CONSUMPTION_GRID,
            return_liquid_param=_RENAMED["return_liquid"],
            return_pension_param=_RENAMED["return_pension"],
            match_rate_param=_RENAMED["match_rate"],
            wage_param=_RENAMED["wage"],
            retirement_income_param=_RENAMED["retirement_income"],
            pension_payout_return_param=_RENAMED["pension_payout_return"],
        ),
        "retired": EGM(
            savings_grid=_SAVINGS_GRID,
            return_param=_RENAMED["return_liquid"],
            income_param=_RENAMED["retirement_income"],
        ),
    }


def _solve_default():
    return get_model(n_periods=_N_PERIODS, solvers=_default_solvers()).solve(
        params=get_params(), log_level="off"
    )


def _solve_renamed():
    return get_model(
        n_periods=_N_PERIODS, solvers=_renamed_solvers(), laws=_renamed_laws()
    ).solve(params=_rename_params(get_params()), log_level="off")


@pytest.mark.parametrize("regime", ["working", "retired"])
def test_renaming_every_law_parameter_leaves_the_solution_unchanged(regime):
    """Both EGM solvers read their roles out of the declared parameter names."""
    default = _solve_default()
    renamed = _solve_renamed()
    compared = 0
    for period, regimes in default.items():
        if regime not in regimes:
            continue
        np.testing.assert_array_equal(
            np.asarray(renamed[period][regime]), np.asarray(regimes[regime])
        )
        compared += 1
    assert compared > 0


def test_the_params_template_asks_for_the_modellers_names():
    """The template names the modeller's parameters, never the kernel's roles."""
    model = get_model(
        n_periods=_N_PERIODS, solvers=_renamed_solvers(), laws=_renamed_laws()
    )
    leaves = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                leaves.add(key)
                walk(value)

    walk(model.get_params_template())
    assert _RENAMED["return_liquid"] in leaves
    assert "return_liquid" not in leaves
