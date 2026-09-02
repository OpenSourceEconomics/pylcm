"""The documented workflow runs end to end on a collective, gated-edge model.

The workflow pylcm documents is `get_params_template()` → fill → `Model(...)` →
`solve(log_level="debug")` → `simulate(log_level="debug")` → `to_dataframe()`.
A model that carries a collective regime and a gated edge is the one that
exercises every piece added for those features, and the template is the piece a
fixture supplying a flat params dict by hand never reaches — the template is
built from the finalized regimes, so a parameter the model consumes but the
template omits is invisible until a user follows the documented route.

`n_subjects` is parametrized so the ahead-of-time compiled path is covered
alongside the ordinary one; it changes when compilation happens, not what is
computed, so both must publish the same frame.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import Model
from lcm.ages import AgeGrid
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    DissolutionRegimeId,
    _make_dissolution_regimes,
)

#: Every parameter this model consumes, as `regime__function__parameter`. A
#: gated edge contributes a branch per target — the gate, one fallback
#: projector per leg, and the regime transition — but in this model none of
#: those binds a free parameter, so each branch is an empty container carrying
#: no leaf.
_EXPECTED_TEMPLATE_LEAVES = frozenset(
    {
        "married__koopmans_aggregator__discount_factor",
        "married_ir__ir_f__delta_f",
        "married_ir__ir_m__delta_m",
        "married_ir__koopmans_aggregator__discount_factor",
        "single_f__koopmans_aggregator__discount_factor",
        "single_m__koopmans_aggregator__discount_factor",
    }
)

_VALUES = {"discount_factor": _BETA, "delta_f": 0.5, "delta_m": 0.2}

_N_SUBJECTS = 3

#: Ages 0-2. Every subject reaches a terminal regime at age 2, and the default
#: `terminal_rows="first"` publishes that entry row but not the frozen ones
#: after it, so the age-3 rows are absent.
_N_LIVE_PERIODS = 3


def _make_model(*, n_subjects: int | None) -> Model:
    """The dissolution model, optionally pinned to an ahead-of-time batch size."""
    return Model(
        regimes=_make_dissolution_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        n_subjects=n_subjects,
    )


def _leaf_paths(*, node: object, prefix: tuple[str, ...] = ()) -> dict[str, object]:
    """Flatten a nested template to `regime__function__parameter` leaf paths."""
    if isinstance(node, dict):
        flattened: dict[str, object] = {}
        for key, value in node.items():
            flattened |= _leaf_paths(node=value, prefix=(*prefix, key))
        return flattened
    return {"__".join(prefix): node}


def _fill_in_place(node: dict) -> None:
    """Set every leaf of a template branch from its own parameter name.

    The walk is recursive rather than three nested loops because the template
    nests to different depths: a regime's function parameters sit two levels
    down, while a per-target regime transition adds a level and can bottom out
    in an EMPTY container when the transition binds no free parameter. A fill
    that assumed a fixed depth would try to assign to that container's key.
    """
    for key, value in node.items():
        if isinstance(value, dict):
            _fill_in_place(value)
        else:
            node[key] = _VALUES[key]


def _filled_template(model: Model) -> dict:
    """The model's own template, with each leaf set from its parameter name."""
    template = model.get_params_template()
    _fill_in_place(template)
    return template


def _initial_conditions(model: Model) -> MappingProxyType:
    """A married cohort of `_N_SUBJECTS`, one on each side of the dissolution."""
    return MappingProxyType(
        {
            "wage": jnp.array([1.0, 2.0, 3.0]),
            "age": jnp.array([0.0, 0.0, 0.0]),
            "regime_id": jnp.array(
                [model.regime_names_to_ids["married"]] * _N_SUBJECTS, dtype=jnp.int32
            ),
            "own_stakeholder": jnp.full(
                _N_SUBJECTS, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
            ),
        }
    )


def test_the_template_asks_for_exactly_the_parameters_the_model_consumes():
    """The template's leaves are the model's parameters — no more, no fewer."""
    template = _make_model(n_subjects=None).get_params_template()

    assert frozenset(_leaf_paths(node=template)) == _EXPECTED_TEMPLATE_LEAVES


def test_every_template_leaf_is_a_parameter_this_test_can_name():
    """A leaf whose name this test cannot fill would make the fill silently partial."""
    template = _make_model(n_subjects=None).get_params_template()
    leaf_names = {path.rsplit("__", 1)[-1] for path in _leaf_paths(node=template)}

    assert leaf_names <= set(_VALUES)


def test_a_gated_edge_contributes_a_branch_per_target_under_its_source():
    """The edge's own callables are addressed under the target they route to."""
    template = _make_model(n_subjects=None).get_params_template()

    assert set(template["married"]["married_ir"]) == {
        "gate",
        "leg_fallback_single_f_wage",
        "leg_fallback_single_m_wage",
        "next_regime",
    }


def test_a_branch_that_binds_no_parameter_is_an_empty_container():
    """Filling the template has to recurse: not every branch bottoms out in a leaf."""
    template = _make_model(n_subjects=None).get_params_template()

    assert template["married"]["married_ir"]["next_regime"] == {}


def test_the_filled_template_solves_to_the_same_values_as_a_flat_params_dict():
    """Following the documented route publishes the same solution, bit for bit."""
    model = _make_model(n_subjects=None)
    from_template = model.solve(
        params=_filled_template(model), log_level="debug"
    ).values
    from_flat = model.solve(
        params={"discount_factor": _BETA, "delta_f": 0.5, "delta_m": 0.2},
        log_level="debug",
    ).values

    for period, regime_to_V in from_template.items():
        for regime_name, V_arr in regime_to_V.items():
            assert (
                np.asarray(V_arr).tobytes()
                == np.asarray(from_flat[period][regime_name]).tobytes()
            ), f"period {period}, regime {regime_name}"


@pytest.mark.parametrize("n_subjects", [None, _N_SUBJECTS])
def test_the_documented_workflow_produces_a_frame(n_subjects: int | None):
    """Template → solve → simulate → dataframe runs at `log_level="debug"`."""
    model = _make_model(n_subjects=n_subjects)
    params = _filled_template(model)
    solution = model.solve(params=params, log_level="debug")
    result = model.simulate(
        params=params,
        initial_conditions=_initial_conditions(model),
        solution=solution,
        log_level="debug",
        seed=0,
    )

    df = result.to_dataframe()

    assert set(map(tuple, df[["subject_id", "period"]].to_numpy())) == {
        (subject, period)
        for subject in range(_N_SUBJECTS)
        for period in range(_N_LIVE_PERIODS)
    }


def test_the_ahead_of_time_path_routes_the_same_subjects_as_the_ordinary_one():
    """Pinning `n_subjects` changes when compilation happens, not what is routed."""
    frames = {}
    for n_subjects in (None, _N_SUBJECTS):
        model = _make_model(n_subjects=n_subjects)
        params = _filled_template(model)
        solution = model.solve(params=params, log_level="debug")
        result = model.simulate(
            params=params,
            initial_conditions=_initial_conditions(model),
            solution=solution,
            log_level="debug",
            seed=0,
        )
        frames[n_subjects] = result.to_dataframe()["regime_name"].tolist()

    assert frames[None] == frames[_N_SUBJECTS]
