"""NB-EGM save-to-cliff targets follow every child row when two states are drawn.

The `alive` regime carries two binary states, `kind` and `shock`, each drawn as an
independent fair coin. The liquid law is `s + 0.1 * next_<law state>`. The income
cliff varies with the geometry state, either through a threshold indexed by it
(`fpl_cliff = (11, 8)`) or through a derived income `liquid + 2 + 3 * <geometry
state>` against a scalar threshold 11; both put the child liquid cliffs at 9 and 6.
Every case constructs, solves, and its targets straddle exactly the preimages of
the child rows' own cliffs:

- sibling: the geometry state differs from the drawn law state, so the joint
  children give centres 9, 8.9, 6 and 5.9, whichever state plays which role and
  in either declaration order;
- same axis: the geometry state is the drawn state, so the centres are 9 and 5.9;
- invariant: the cliff is 9 for every kind, so the centres are 9 and 8.9;
- fixed: the geometry state is carried by a fixed law, so the child keeps the
  source cell's cliff `c` and the centres are `c` and `c - 0.1`.

The exact centres come from `_cliff_pullback_reference`.
"""

from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from lcm import DiscreteGrid, LinSpacedGrid, Model
from lcm.regime import StateTransitionEntry
from lcm.typing import ContinuousState, DiscreteState, FloatND, StateName
from tests.solution._cliff_pullback_reference import (
    child_cliff_preimages,
    sibling_draw_preimages,
)
from tests.solution.test_nbegm_draw_conditioned_cliffs import (
    _exact,
    _params,
    _solved_seam,
    _targets,
)
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    ConsumerKind,
    gross_income,
    resources,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    subsidy as kind_indexed_subsidy,
)

type _Geometry = Literal["indexed", "derived", "invariant"]

_CUTOFFS = (Fraction(9), Fraction(6))


def _liquid_reads_shock(
    *, savings: FloatND, next_shock: DiscreteState
) -> ContinuousState:
    return savings + 0.1 * next_shock


def _liquid_reads_kind(
    *, savings: FloatND, next_kind: DiscreteState
) -> ContinuousState:
    return savings + 0.1 * next_kind


def _kind_probabilities(*, kind: DiscreteState) -> FloatND:
    return jnp.asarray(((0.5, 0.5), (0.5, 0.5)))[kind]


def _shock_probabilities(*, shock: DiscreteState) -> FloatND:
    return jnp.asarray(((0.5, 0.5), (0.5, 0.5)))[shock]


def _zero_bequest(*, liquid: ContinuousState) -> FloatND:
    return jnp.zeros_like(liquid)


@lcm.piecewise_affine(
    output="subsidy",
    variable="gross_income",
    breakpoints=(
        lcm.affine_breakpoint(threshold="fpl_cliff", kind="jump", indexed_by="shock"),
    ),
)
def _shock_indexed_subsidy(
    *,
    gross_income: FloatND,
    shock: DiscreteState,
    subsidy_low: float,
    subsidy_high: float,
    fpl_cliff: FloatND,
) -> FloatND:
    return jnp.where(gross_income < fpl_cliff[shock], subsidy_high, subsidy_low)


@lcm.piecewise_affine(
    output="subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint(threshold="fpl_cliff", kind="jump"),),
)
def _scalar_threshold_subsidy(
    *,
    gross_income: FloatND,
    subsidy_low: float,
    subsidy_high: float,
    fpl_cliff: float,
) -> FloatND:
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


def _income_derived_from_kind(
    *, liquid: ContinuousState, kind: DiscreteState, base_income: float
) -> FloatND:
    return liquid + base_income + 3.0 * kind


def _income_derived_from_shock(
    *, liquid: ContinuousState, shock: DiscreteState, base_income: float
) -> FloatND:
    return liquid + base_income + 3.0 * shock


def _model(
    *,
    geometry: _Geometry,
    geometry_state: StateName,
    law_state: StateName,
    reverse_declaration_order: bool = False,
    fixed_geometry_state: bool = False,
) -> Model:
    income: Callable[..., object] = gross_income
    subsidy: Callable[..., object] = _scalar_threshold_subsidy
    if geometry == "indexed":
        subsidy = (
            kind_indexed_subsidy if geometry_state == "kind" else _shock_indexed_subsidy
        )
    elif geometry == "derived":
        income = (
            _income_derived_from_kind
            if geometry_state == "kind"
            else _income_derived_from_shock
        )
    state_order = ("shock", "kind") if reverse_declaration_order else ("kind", "shock")
    transitions: dict[StateName, StateTransitionEntry] = {
        "kind": lcm.StochasticTransition(func=_kind_probabilities),
        "shock": lcm.StochasticTransition(func=_shock_probabilities),
    }
    if fixed_geometry_state:
        transitions[geometry_state] = {
            "alive": lcm.fixed_transition(state_name=geometry_state)
        }
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=31,
        liquid_max=30.0,
        n_consumption=31,
        liquid_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=31),
        alive_functions={
            "utility": utility,
            "gross_income": income,
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=_liquid_reads_kind if law_state == "kind" else _liquid_reads_shock,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=16),
            envelope_arithmetic="ordinary",
        ),
        constraints={},
        extra_states={
            name: DiscreteGrid(category_class=ConsumerKind) for name in state_order
        },
        extra_state_transitions={name: transitions[name] for name in state_order},
        dead_functions={"utility": _zero_bequest},
    )


def _model_params(*, geometry: _Geometry) -> dict[str, Any]:
    """Thresholds putting the child liquid cliffs at 9 and 6 (9 when invariant)."""
    fpl_cliff = jnp.asarray((11.0, 8.0)) if geometry == "indexed" else 11.0
    params = _params(fpl_cliff=fpl_cliff, law_slope=1.0, law_offset=(0.0, 0.0))
    del params["alive"]["next_liquid"]
    return params


def _covered_centres(
    *,
    model: Model,
    geometry: _Geometry,
    cell: dict[str, int],
    centres: frozenset[Fraction],
) -> frozenset[Fraction]:
    """Return the centres some finite target pair tightly straddles.

    Fails if any finite target pair straddles none of `centres`.
    """
    seam = _solved_seam(model=model, params=_model_params(geometry=geometry))
    rows = _targets(
        seam=seam,
        cell={name: jnp.asarray(code, dtype=jnp.int32) for name, code in cell.items()},
    )
    rows = rows[np.isfinite(rows).all(axis=1)]
    eps = Fraction(float(np.finfo(rows.dtype).eps))
    assert all(
        any(_exact(below) < centre < _exact(above) for centre in centres)
        for below, above in rows
    ), rows
    return frozenset(
        centre
        for centre in centres
        for below, above in rows
        if _exact(below) < centre < _exact(above)
        and _exact(above) - _exact(below) <= 64 * eps * max(Fraction(1), centre)
    )


def _shifts() -> tuple[Fraction, Fraction]:
    """Intercepts of the liquid law at the two draw nodes, as stored."""
    dtype = jnp.asarray(0.0).dtype
    return (Fraction(0), _exact(np.dtype(dtype).type(0.1)))


_ROLES = [("kind", "shock"), ("shock", "kind")]


@pytest.mark.parametrize("reverse_declaration_order", [False, True])
@pytest.mark.parametrize(("geometry_state", "law_state"), _ROLES)
@pytest.mark.parametrize("geometry", ["indexed", "derived"])
def test_cliff_varying_with_a_sibling_state_targets_every_joint_child(
    *,
    geometry: _Geometry,
    geometry_state: StateName,
    law_state: StateName,
    reverse_declaration_order: bool,
) -> None:
    """Cliffs (9, 6) by one state and law `s + other / 10` give 9, 8.9, 6, 5.9."""
    centres = sibling_draw_preimages(
        cutoffs=_CUTOFFS, slope=Fraction(1), shifts=_shifts()
    )
    model = _model(
        geometry=geometry,
        geometry_state=geometry_state,
        law_state=law_state,
        reverse_declaration_order=reverse_declaration_order,
    )
    covered = _covered_centres(
        model=model, geometry=geometry, cell={"kind": 0, "shock": 0}, centres=centres
    )
    assert covered == centres


@pytest.mark.parametrize("state", ["kind", "shock"])
@pytest.mark.parametrize("geometry", ["indexed", "derived"])
def test_cliff_varying_with_the_drawn_state_targets_each_child_row(
    *, geometry: _Geometry, state: StateName
) -> None:
    """Cliffs (9, 6) and law `s + state / 10` on the same state give 9 and 5.9."""
    centres = frozenset(
        child_cliff_preimages(cutoffs=_CUTOFFS, slope=Fraction(1), offsets=_shifts())
    )
    model = _model(geometry=geometry, geometry_state=state, law_state=state)
    covered = _covered_centres(
        model=model, geometry=geometry, cell={"kind": 0, "shock": 0}, centres=centres
    )
    assert covered == centres


@pytest.mark.parametrize(("geometry_state", "law_state"), _ROLES)
def test_cliff_invariant_across_states_targets_one_cliff_per_draw(
    *, geometry_state: StateName, law_state: StateName
) -> None:
    """A cliff at 9 for every kind and law `s + state / 10` give 9 and 8.9."""
    centres = sibling_draw_preimages(
        cutoffs=(Fraction(9),), slope=Fraction(1), shifts=_shifts()
    )
    model = _model(
        geometry="invariant", geometry_state=geometry_state, law_state=law_state
    )
    covered = _covered_centres(
        model=model,
        geometry="invariant",
        cell={"kind": 0, "shock": 0},
        centres=centres,
    )
    assert covered == centres


@pytest.mark.parametrize("source_code", [0, 1])
@pytest.mark.parametrize(("geometry_state", "law_state"), _ROLES)
@pytest.mark.parametrize("geometry", ["indexed", "derived"])
def test_cliff_varying_with_a_fixed_state_targets_the_source_cells_cliff(
    *,
    geometry: _Geometry,
    geometry_state: StateName,
    law_state: StateName,
    source_code: int,
) -> None:
    """A fixed geometry state keeps the source cliff `c`, giving `c` and `c - 0.1`."""
    centres = sibling_draw_preimages(
        cutoffs=(_CUTOFFS[source_code],), slope=Fraction(1), shifts=_shifts()
    )
    model = _model(
        geometry=geometry,
        geometry_state=geometry_state,
        law_state=law_state,
        fixed_geometry_state=True,
    )
    covered = _covered_centres(
        model=model,
        geometry=geometry,
        cell={geometry_state: source_code, law_state: 0},
        centres=centres,
    )
    assert covered == centres
