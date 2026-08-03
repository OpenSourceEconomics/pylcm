import jax.numpy as jnp
import numpy as np
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


_PROCESS_SOLVE_PARAMS = {
    "discount_factor": 1.0,
    "source__shock__rho": 0.5,
    "source__shock__sigma": 0.3,
    "source__shock__mu": 0.0,
    "source__shock__n_std": 2.0,
    "target__shock__rho": 0.5,
    "target__shock__sigma": 0.3,
    "target__shock__mu": 0.0,
    "target__shock__n_std": 2.0,
}


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
    """A source value supplies the conditioning state for the target process.

    Beyond construction, `.solve()` must actually receive the target's shared
    process in its continuation: the source's period-0 value must be finite
    and reflect the target's nonzero terminal payoff, not silently treat the
    continuation as zero because the process-only target was dropped from
    the generated transition DAG.
    """
    model = _build_overlapping_model(coarse=False, carry_process=True)

    assert model.reachability.solution.targets(period=0, source="source") == ("target",)

    solution = model.solve(params=_PROCESS_SOLVE_PARAMS, log_level="debug")

    source_v = solution[0]["source"]
    assert bool(jnp.all(jnp.isfinite(source_v)))
    assert not bool(jnp.allclose(source_v, 0.0))


def test_process_only_target_matches_equivalent_target_with_inert_nonprocess_law() -> (
    None
):
    """Adding an inert non-process law to a target must not change its value.

    Two economically identical models: one where `target` carries only the
    shared process state, one where `target` also carries an inert
    non-process state whose entry law makes the old
    `flat_nested_transitions`-derived route non-empty for `target`. The
    inert state contributes exactly zero to utility, so `source`'s period-0
    continuation value must be identical either way — the existence of a
    continuation target is a property of the reachability graph, not of
    whether some unrelated state happens to have a law-of-motion entry.
    """
    process = TauchenAR1Process(
        n_points=3, gauss_hermite=False, rho=0.5, sigma=0.3, mu=0.0, n_std=2.0
    )

    def _process_only_model() -> Model:
        return Model(
            regimes={
                "source": Regime(
                    transition={"target": MarkovTransition(_one_probability)},
                    active=_source_is_early,
                    states={"shock": process},
                    functions={"utility": _shock_utility},
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

    def _shock_and_inert_utility(shock: ScalarFloat, extra: ScalarFloat) -> ScalarFloat:
        return shock + jnp.float32(0) * extra

    def _process_and_inert_law_model() -> Model:
        return Model(
            regimes={
                "source": Regime(
                    transition={"target": MarkovTransition(_one_probability)},
                    active=_source_is_early,
                    states={"shock": process},
                    state_transitions={
                        "extra": {"target": lambda: jnp.float32(0.0)},
                    },
                    functions={"utility": _shock_utility},
                ),
                "target": Regime(
                    transition=None,
                    states={
                        "shock": process,
                        "extra": LinSpacedGrid(start=0, stop=1, n_points=2),
                    },
                    functions={"utility": _shock_and_inert_utility},
                ),
            },
            ages=AgeGrid(start=20, stop=22, step="Y"),
            regime_id_class=RegimeId,
            enable_jit=False,
        )

    process_only = _process_only_model()
    process_and_inert_law = _process_and_inert_law_model()

    assert process_only.reachability.solution.targets(period=0, source="source") == (
        "target",
    )
    assert process_and_inert_law.reachability.solution.targets(
        period=0, source="source"
    ) == ("target",)

    solve_params = {"discount_factor": 1.0}
    v_process_only = process_only.solve(params=solve_params, log_level="debug")
    v_process_and_inert_law = process_and_inert_law.solve(
        params=solve_params, log_level="debug"
    )

    np.testing.assert_allclose(
        np.asarray(v_process_only[0]["source"]),
        np.asarray(v_process_and_inert_law[0]["source"]),
        atol=1e-6,
    )


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
