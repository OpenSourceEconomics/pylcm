import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    TauchenAR1Process,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ScalarFloat, ScalarInt


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> ScalarFloat:
    return jnp.float32(0)


def _shock_utility(shock: ScalarFloat) -> ScalarFloat:
    return shock


def _one_probability() -> ScalarFloat:
    return jnp.float32(1)


def _next_target() -> ScalarInt:
    return RegimeId.target


def _source_is_early(age: float) -> bool:
    return age < 22


def _source_is_forced_out(age: float) -> bool:
    return age >= 65


def _target_can_work(age: float) -> bool:
    return age < 65


def _build_overlapping_model(*, coarse: bool, carry_process: bool = False) -> Model:
    process = TauchenAR1Process(n_points=3, gauss_hermite=False)
    source_states = {"shock": process} if carry_process else {}
    transition = (
        _next_target if coarse else {"target": MarkovTransition(_one_probability)}
    )
    return Model(
        regimes={
            "source": Regime(
                transition=transition,
                active=_source_is_early,
                states=source_states,
                functions={
                    "utility": _shock_utility if carry_process else _zero_utility
                },
            ),
            "target": Regime(
                transition=None,
                states={"shock": process},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )


@pytest.mark.parametrize(
    "process",
    [
        TauchenAR1Process(n_points=3, gauss_hermite=False),
        NormalIIDProcess(n_points=3, gauss_hermite=False),
    ],
    ids=["ar1", "iid"],
)
def test_activity_compatible_target_only_process_is_rejected(
    process: TauchenAR1Process | NormalIIDProcess,
) -> None:
    """A retained edge cannot introduce a stochastic process without a value."""
    with pytest.raises(
        ModelInitializationError,
        match=r"solution phase.*period 0.*source.*target.*shock",
    ):
        Model(
            regimes={
                "source": Regime(
                    transition={"target": MarkovTransition(_one_probability)},
                    active=_source_is_early,
                    functions={"utility": _zero_utility},
                ),
                "target": Regime(
                    transition=None,
                    states={"shock": process},
                    functions={"utility": _shock_utility},
                ),
            },
            ages=AgeGrid(start=20, stop=22, step="Y"),
            regime_id_class=RegimeId,
            enable_jit=False,
        )


def test_process_carried_by_source_and_target_is_accepted() -> None:
    """A source value supplies the conditioning state for the target process."""
    model = _build_overlapping_model(coarse=False, carry_process=True)

    assert model.reachability.solution.targets(period=0, source="source") == ("target",)


def test_explicit_entry_law_for_target_only_process_is_accepted() -> None:
    """An entry law can initialize a process that is absent from the source."""
    process = TauchenAR1Process(n_points=3, gauss_hermite=False)
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={
                    "shock": {"target": lambda: jnp.float32(0)},
                },
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": process},
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )

    assert model.reachability.solution.targets(period=0, source="source") == ("target",)


def test_activity_incompatible_target_only_process_is_accepted() -> None:
    """A declared target outside the adjacent activity window needs no handoff."""
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_forced_out,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                active=_target_can_work,
                states={
                    "shock": TauchenAR1Process(
                        n_points=3,
                        gauss_hermite=False,
                    )
                },
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=60, stop=68, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )

    assert (
        model.reachability.solution.periods_for_edge(source="source", target="target")
        == ()
    )


def test_coarse_activity_incompatible_target_only_process_is_accepted() -> None:
    """A coarse declaration needs no handoff outside the adjacent activity window."""
    model = Model(
        regimes={
            "source": Regime(
                transition=_next_target,
                active=_source_is_forced_out,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                active=_target_can_work,
                states={
                    "shock": TauchenAR1Process(
                        n_points=3,
                        gauss_hermite=False,
                    )
                },
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=60, stop=68, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )

    assert (
        model.reachability.solution.periods_for_edge(source="source", target="target")
        == ()
    )


def test_target_only_nonprocess_state_is_accepted() -> None:
    """A target-local continuous state does not require a source handoff."""
    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": LinSpacedGrid(start=-1, stop=1, n_points=3),
                },
                functions={"utility": _shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )

    assert model.reachability.solution.targets(period=0, source="source") == ("target",)


def test_coarse_transition_validates_each_activity_compatible_candidate() -> None:
    """A function-based transition must be valid for every retained candidate."""
    with pytest.raises(ModelInitializationError, match=r"source.*target.*shock"):
        _build_overlapping_model(coarse=True)
