"""Supplied targets agree with the law at every non-final source age.

A selector that also names the final age says nothing more, since no edge
leaves it; a genuine non-final difference is always refused. The edge-parameter
fixtures reach only terminal regimes at the final age, with unit mass out of
their last source age.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import FloatND, ScalarInt
from tests.params.test_edge_params import (
    _horizon_retirement_model,
    _mortal_model,
    _retirement_model,
)


@categorical(ordered=False)
class BoundaryId:
    source: ScalarInt
    left: ScalarInt
    right: ScalarInt


def zero() -> FloatND:
    return jnp.asarray(0.0)


def one() -> FloatND:
    return jnp.asarray(1.0)


def two() -> FloatND:
    return jnp.asarray(2.0)


def half() -> FloatND:
    return jnp.asarray(0.5)


def boundary_model(*, final_metadata: str, mutation: str = "none") -> Model:
    """Every source age is a start; every destination is terminal.

    Law support is left at 0, right at 1, and both at 2. Each case has unit mass.
    Removing left at 2 leaves right as a terminal destination; adding left at 1
    also adds a terminal destination. Thus these support mutations do not rely
    on an unintended nonterminal endpoint to produce their expected refusal.
    """
    selected = {"left": [0, 2], "right": [1, 2]}
    if mutation == "remove-nonfinal":
        selected["left"].remove(2)
    elif mutation == "add-nonfinal":
        selected["left"].append(1)
    for target, ages in selected.items():
        if final_metadata in (target, "both"):
            ages.append(3)
    return Model(
        regimes={
            "source": Regime(functions={"utility": zero}),
            "left": Regime(functions={"utility": one}),
            "right": Regime(functions={"utility": two}),
        },
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=BoundaryId,
        initial_nodes={0: "source", 1: "source", 2: "source"},
        edges={
            "source": Transition(
                targets={
                    target: tuple(sorted(ages)) for target, ages in selected.items()
                },
                law=ByAge(
                    cases={
                        0: {"left": StochasticTransition(func=one)},
                        1: {"right": StochasticTransition(func=one)},
                        2: {
                            "left": StochasticTransition(func=half),
                            "right": StochasticTransition(func=half),
                        },
                    }
                ),
            )
        },
    )


@pytest.mark.parametrize("final_metadata", ["none", "left", "right", "both"])
def test_only_final_age_selector_metadata_is_ignored(*, final_metadata, request):
    """Naming the final age in a selector leaves the edges and values unchanged."""
    instance = boundary_model(final_metadata=final_metadata)
    expected = {"left": frozenset({0, 2}), "right": frozenset({1, 2})}
    assert dict(instance.graph.edges.solve["source"]) == expected
    assert dict(instance.graph.edges.simulate["source"]) == expected
    values = instance.solve(params={"discount_factor": 0.5}, log_level="debug").values
    # The three independently started source problems have terminal continuation
    # utilities 1, 2, and (1 + 2) / 2, respectively. The source's own utility is zero.
    # All constants and arithmetic here are binary-exact at either precision.
    dtype = np.dtype(f"float{request.config.getoption('--precision')}")
    for period, expected_value in enumerate((0.5, 1.0, 0.75)):
        actual = np.asarray(values[period]["source"])
        expected = np.asarray(expected_value, dtype=dtype)
        assert actual.shape == ()
        assert actual.dtype == dtype
        assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("final_metadata", ["none", "left", "right", "both"])
@pytest.mark.parametrize("mutation", ["remove-nonfinal", "add-nonfinal"])
def test_real_nonfinal_support_mismatch_is_never_ignored(*, final_metadata, mutation):
    """Removing or adding a non-final age is refused, final-age naming or not."""
    # Both left selectors still include 0, so neither the empty-selector nor the
    # final-age-only-selector guard can accidentally satisfy this assertion.
    with pytest.raises(
        ModelInitializationError,
        match=(
            r"(?s)`Transition\.targets` of 'source' disagree.*supplied:.*"
            r"derived from"
        ),
    ):
        boundary_model(final_metadata=final_metadata, mutation=mutation)


@pytest.mark.parametrize("inclusive_stop", [62, 63, 64])
def test_mortal_fixture_exits_with_unit_mass_at_its_last_source_age(inclusive_stop):
    """The mortal fixture dies for sure out of its last source age."""
    # These are the current helper's supported 3/4/5-period test configurations.
    instance = _mortal_model(
        ages=AgeGrid(start=60, inclusive_stop=inclusive_stop, step="Y")
    )
    last_source = inclusive_stop - 1
    support = instance.graph.edges.solve["working"]
    assert last_source not in support["working"]
    assert last_source in support["dead"]
    # A 0.25 survival law would leave only 0.75 mass if its final per-target law were
    # merely masked to dead. The explicit final then branch must provide 1.0.
    values = instance.solve(
        params={"discount_factor": 0.95, "survival_probability": 0.25},
        log_level="debug",
    ).values
    assert "working" in values[0]


def test_retirement_fixture_forces_retirement_before_the_final_death():
    """The retirement fixture retires out of its last working age."""
    instance = _retirement_model()
    support = instance.graph.edges.solve
    assert 61 not in support["working"]["working"]
    assert 61 in support["working"]["retired"]
    assert 62 in support["retired"]["dead"]
    # A threshold beyond the horizon must still be overridden at source 61.
    values = instance.solve(
        params={"discount_factor": 0.95, "retirement_age": 100.0},
        log_level="debug",
    ).values
    assert "working" in values[0]


@pytest.mark.parametrize("n_periods", [2, 3, 4])
def test_horizon_fixture_forces_unit_mass_into_terminal_retirement(n_periods):
    """The horizon fixture retires for sure out of its last source age."""
    instance = _horizon_retirement_model(n_periods=n_periods)
    last_source = 58 + n_periods
    support = instance.graph.edges.solve["working"]
    assert last_source not in support.get("working", frozenset())
    assert last_source in support["retired"]
    values = instance.solve(
        params={"discount_factor": 0.95, "retirement_age": 100.0},
        log_level="debug",
    ).values
    assert "working" in values[0]
