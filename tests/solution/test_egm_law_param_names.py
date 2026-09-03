"""The EGM kernel never has to be told what a law's parameters are called.

A kernel that inverts an Euler equation needs specific economic quantities out
of a law of motion — a gross return, a labour income, a pension match. It gets
them by composing and differentiating the law the regime declared, so the names
the modeller chose are never consulted and never have to be declared.

Renaming a parameter is a relabelling: it cannot move the solution. Each case
below solves one model twice, once under the default spellings and once under
deliberately different ones, with the *same* solver declaration in both. The two
must be identical — equality, not a tolerance, since the two runs execute the
same arithmetic on the same nodes, so any difference at all would mean a name
reached a slot it should never have reached.
"""

import numpy as np
from dags import rename_arguments

from lcm.solvers import EGM
from tests.solution.test_egm_continuation_grid_provenance import (
    _N_PERIODS,
    _SAVINGS_GRID,
)
from tests.test_models.deterministic import ds_pension as ds
from tests.test_models.deterministic.ds_pension import get_model, get_params

# The modeller's spellings, deliberately unlike anything the kernel could guess.
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


def _solvers():
    """One declaration, used for both spellings — it names no parameter."""
    return {"retired": EGM(savings_grid=_SAVINGS_GRID)}


def _solve_default():
    return (
        get_model(n_periods=_N_PERIODS, solvers=_solvers())
        .solve(params=get_params(), log_level="debug")
        .values
    )


def _solve_renamed():
    return (
        get_model(n_periods=_N_PERIODS, solvers=_solvers(), laws=_renamed_laws())
        .solve(params=_rename_params(get_params()), log_level="debug")
        .values
    )


def test_renaming_every_law_parameter_leaves_the_solution_unchanged():
    """The solution is invariant to what the law's parameters are called."""
    regime = "retired"
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
    model = get_model(n_periods=_N_PERIODS, solvers=_solvers(), laws=_renamed_laws())
    leaves = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                leaves.add(key)
                walk(value)

    walk(model.get_params_template())
    assert _RENAMED["return_liquid"] in leaves
    assert "return_liquid" not in leaves
