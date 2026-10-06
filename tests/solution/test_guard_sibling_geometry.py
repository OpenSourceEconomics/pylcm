"""NB-EGM refuses child cliffs that vary with any stochastic state the child carries.

The `alive` regime carries two binary states, `kind` and `shock`, each drawn as an
independent fair coin. The liquid law reads one of the draws, `next_<law state>`.
The income cliff varies with the geometry state, either through a threshold
indexed by it (`fpl_cliff = (11, 8)`) or through a derived income
`liquid + 2 + 3 * <geometry state>` against a scalar threshold 11; both put the
child liquid cliffs at 9 and 6. The expectation runs over all four child rows,
so a cliff varying with either stochastic state is not represented by the source
cell's breakpoints, and construction refuses it:

- sibling: the geometry state differs from the state the liquid law draws;
- same axis: the geometry state is the drawn state.

Construction succeeds when the cliff is the same for every kind (invariant) and
when the geometry state is carried by a fixed law (a single child node).
"""

import contextlib
from collections.abc import Callable, Iterator
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

import lcm
from lcm import DiscreteGrid, LinSpacedGrid, Model
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousState, DiscreteState, FloatND
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


@contextlib.contextmanager
def _precision(bits: int) -> Iterator[None]:
    """Build at `bits`-bit floats, restoring the suite's setting."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", bits == 64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _model(
    *,
    geometry: _Geometry,
    geometry_state: str,
    law_state: str,
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
    transitions: dict[str, object] = {
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


def _refusal(*, geometry_state: str, law_state: str) -> str:
    """The part of the refusal naming the regime, the geometry state and the draw."""
    return (
        "Regime 'alive' has a jump breakpoint 'subsidy__fpl_cliff' that varies "
        f"with the stochastic state '{geometry_state}', and its liquid law reads "
        f"the draw 'next_{law_state}'."
    )


_PRECISIONS = [pytest.param(bits, id=f"fp{bits}") for bits in (32, 64)]
_ROLES = [("kind", "shock"), ("shock", "kind")]


@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("reverse_declaration_order", [False, True])
@pytest.mark.parametrize(("geometry_state", "law_state"), _ROLES)
@pytest.mark.parametrize("geometry", ["indexed", "derived"])
def test_cliff_varying_with_a_sibling_stochastic_state_is_refused(
    *,
    geometry: _Geometry,
    geometry_state: str,
    law_state: str,
    reverse_declaration_order: bool,
    precision: int,
) -> None:
    """A cliff varying with a state the liquid law does not draw is refused."""
    with _precision(precision), pytest.raises(ModelInitializationError) as raised:
        _model(
            geometry=geometry,
            geometry_state=geometry_state,
            law_state=law_state,
            reverse_declaration_order=reverse_declaration_order,
        )
    assert _refusal(geometry_state=geometry_state, law_state=law_state) in str(
        raised.value
    )


@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize("state", ["kind", "shock"])
@pytest.mark.parametrize("geometry", ["indexed", "derived"])
def test_cliff_varying_with_the_drawn_state_is_refused(
    *, geometry: _Geometry, state: str, precision: int
) -> None:
    """A cliff varying with the state the liquid law draws is refused."""
    with _precision(precision), pytest.raises(ModelInitializationError) as raised:
        _model(geometry=geometry, geometry_state=state, law_state=state)
    assert _refusal(geometry_state=state, law_state=state) in str(raised.value)


@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize(("geometry_state", "law_state"), _ROLES)
def test_cliff_invariant_across_states_builds(
    *, geometry_state: str, law_state: str, precision: int
) -> None:
    """A cliff no state moves builds and carries both stochastic states."""
    with _precision(precision):
        model = _model(
            geometry="invariant", geometry_state=geometry_state, law_state=law_state
        )
        assert set(model.state_names(regime_name="alive")) == {
            "liquid",
            "kind",
            "shock",
        }


@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize(("geometry_state", "law_state"), _ROLES)
@pytest.mark.parametrize("geometry", ["indexed", "derived"])
def test_cliff_varying_with_a_fixed_state_builds(
    *, geometry: _Geometry, geometry_state: str, law_state: str, precision: int
) -> None:
    """A cliff varying with a state carried by a fixed law builds."""
    with _precision(precision):
        model = _model(
            geometry=geometry,
            geometry_state=geometry_state,
            law_state=law_state,
            fixed_geometry_state=True,
        )
        assert set(model.state_names(regime_name="alive")) == {
            "liquid",
            "kind",
            "shock",
        }
