"""The case-piece kernels name a grid they cannot read at model construction.

NBEGM and NNBEGM lay out savings nodes, carry shapes, and interval geometry
while the model is built, so the grids those steps read must carry their points
by then. A grid that withholds them until params arrive is named here, with its
regime and its role, instead of raising from whichever `to_jax()` reaches it
first — a message that names neither.

The continuous action grids are the exception, and deliberately so: no
build-time step reads them, and a runtime-supplied one solves to the same value
function. Refusing them would forbid a configuration that works.
"""

import dataclasses

import pytest

from _lcm.grids import IrregSpacedGrid
from lcm import Model
from lcm.exceptions import ModelInitializationError
from tests.test_models import n_nbegm_toy, nbegm_common, nbegm_tax_toy


def _rebuilt(*, model, regime_id_class, regime_name, kind, name):
    """The model with one grid slot replaced by a runtime-supplied grid."""
    regime = model.user_regimes[regime_name]
    if kind in {"states", "actions"}:
        grids = dict(getattr(regime, kind))
        grids[name] = IrregSpacedGrid(n_points=int(grids[name].n_points))
        regime = dataclasses.replace(regime, **{kind: grids})
    elif kind == "outer_search":
        solver = regime.solver
        search = solver.outer_search
        runtime = IrregSpacedGrid(n_points=int(getattr(search, name).n_points))
        solver = dataclasses.replace(
            solver,
            outer_search=dataclasses.replace(search, **{name: runtime}),
        )
        regime = dataclasses.replace(regime, solver=solver)
    else:
        solver = regime.solver
        holder = solver if kind == "solver" else solver.inner
        runtime = IrregSpacedGrid(n_points=int(getattr(holder, name).n_points))
        holder = dataclasses.replace(holder, **{name: runtime})
        solver = (
            holder if kind == "solver" else dataclasses.replace(solver, inner=holder)
        )
        regime = dataclasses.replace(regime, solver=solver)
    return Model(
        regimes={**model.user_regimes, regime_name: regime},
        ages=model.ages,
        regime_id_class=regime_id_class,
    )


def _nbegm(*, kind, name):
    return _rebuilt(
        model=nbegm_tax_toy.build_model(variant="nbegm", n_periods=3),
        regime_id_class=nbegm_common.RegimeId,
        regime_name="alive",
        kind=kind,
        name=name,
    )


def _nnbegm(*, kind, name):
    return _rebuilt(
        model=n_nbegm_toy.build_model(variant="n_nbegm"),
        regime_id_class=n_nbegm_toy.RegimeId,
        regime_name="alive",
        kind=kind,
        name=name,
    )


@pytest.mark.parametrize(
    ("kind", "name", "role"),
    [
        ("solver", "savings_grid", "savings grid"),
        ("states", "liquid", "grid of the liquid state 'liquid'"),
    ],
)
def test_nbegm_names_a_runtime_supplied_grid_its_kernel_reads(*, kind, name, role):
    """The refusal identifies the regime and what the grid is to the solver."""
    with pytest.raises(ModelInitializationError, match=f"{role} in regime 'alive'"):
        _nbegm(kind=kind, name=name)


def test_nbegm_accepts_a_runtime_supplied_continuous_action_grid():
    """The consumption grid may withhold its points: no build step reads it."""
    model = _nbegm(kind="actions", name="consumption")

    assert model.user_regimes["alive"].actions["consumption"].pass_points_at_runtime


@pytest.mark.parametrize(
    ("kind", "name", "role"),
    [
        ("outer_search", "grid", "outer grid"),
        ("inner", "savings_grid", "inner savings grid"),
        ("states", "illiquid", "grid of the outer state 'illiquid'"),
        ("states", "wealth", "grid of the liquid state 'wealth'"),
    ],
)
def test_nnbegm_names_a_runtime_supplied_grid_its_kernel_reads(*, kind, name, role):
    """Both margins' kernel grids are named the same way the one-margin ones are."""
    with pytest.raises(ModelInitializationError, match=f"{role} in regime 'alive'"):
        _nnbegm(kind=kind, name=name)


@pytest.mark.parametrize("name", ["consumption", "illiquid_investment"])
def test_nnbegm_accepts_a_runtime_supplied_continuous_action_grid(name):
    """Neither margin's action grid is read while the model is built."""
    model = _nnbegm(kind="actions", name=name)

    assert model.user_regimes["alive"].actions[name].pass_points_at_runtime
