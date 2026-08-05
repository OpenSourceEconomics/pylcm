import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    Regime,
    TauchenAR1Process,
    categorical,
    fixed_transition,
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


def _target_only_process_model(
    process: TauchenAR1Process | NormalIIDProcess,
) -> Model:
    """Build a source whose declared target's only state is `process`."""
    return Model(
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


def test_activity_compatible_target_only_ar1_is_rejected() -> None:
    """A retained edge cannot introduce an AR(1) process without a value.

    Its next draw is conditioned on a previous value that the source neither
    carries nor supplies, so no next-period value exists.
    """
    with pytest.raises(
        ModelInitializationError,
        match=r"solution phase.*period 0.*source.*target.*shock",
    ):
        _target_only_process_model(
            TauchenAR1Process(n_points=3, gauss_hermite=False),
        )


def test_activity_compatible_target_only_iid_is_entered_at_its_own_law() -> None:
    """A target-only IID process is priced at its unconditional mean.

    An IID draw does not depend on its previous value, so the source has
    nothing to hand over and needs nothing: the entry distribution is the
    process's own. With the target's payoff equal to the shock and no
    discounting, the source's value is that distribution's mean, which is `mu`.
    A dropped continuation would publish `0.0` instead, so the nonzero `mu`
    is what makes the two distinguishable.
    """
    model = _target_only_process_model(
        NormalIIDProcess(n_points=3, gauss_hermite=False, mu=1.0, sigma=0.3, n_std=2.0),
    )

    assert model.reachability.solution.targets(period=0, source="source") == ("target",)

    solution = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="debug",
    )
    np.testing.assert_allclose(np.asarray(solution[0]["source"]), 1.0, atol=1e-6)


def test_target_only_iid_whose_law_arrives_at_runtime_is_rejected() -> None:
    """An entered process must have its law fixed at construction.

    The entry distribution is priced inside the source's Bellman equation,
    which reads only the source's own parameters, so a law the target
    parameterizes at runtime has no value the source could read. The message
    names the state and the parameters that block it.
    """
    with pytest.raises(
        ModelInitializationError,
        match=r"'shock' passes 'mu', 'n_std', 'sigma' at runtime",
    ):
        _target_only_process_model(
            NormalIIDProcess(n_points=3, gauss_hermite=False),
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


def _explicit_entry_model(process: TauchenAR1Process) -> Model:
    """Build a source that enters its target's process at the value `0.0`."""
    return Model(
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


def test_explicit_entry_law_for_target_only_process_is_accepted() -> None:
    """An entry law can initialize a process that is absent from the source.

    The declared value is placed on the target's own nodes, so with the payoff
    equal to the shock and no discounting the source's value is the entry value
    itself — here the centre node `0.0` of a symmetric AR(1) support.
    """
    model = _explicit_entry_model(
        TauchenAR1Process(
            n_points=3, gauss_hermite=False, rho=0.5, sigma=0.3, mu=0.0, n_std=2.0
        )
    )

    assert model.reachability.solution.targets(period=0, source="source") == ("target",)

    solution = model.solve(
        params={"source": {"koopmans_aggregator": {"discount_factor": 1.0}}},
        log_level="debug",
    )
    np.testing.assert_allclose(np.asarray(solution[0]["source"]), 0.0, atol=1e-6)


def test_explicit_entry_into_a_runtime_parameterized_process_is_rejected() -> None:
    """An entry law cannot place a value on a support built at runtime.

    The declared value becomes a coordinate on the target's nodes, and those
    nodes are built inside the source's Bellman equation, which reads only the
    source's own parameters. The message names the state and the parameters
    that block it.
    """
    with pytest.raises(
        ModelInitializationError,
        match=r"'shock' passes .* at runtime",
    ):
        _explicit_entry_model(TauchenAR1Process(n_points=3, gauss_hermite=False))


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


def test_target_only_nonprocess_state_without_entry_law_is_rejected() -> None:
    """A retained edge cannot invent a value for a target-only ordinary state."""
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


def test_target_only_discrete_state_on_a_nonterminal_target_is_rejected() -> None:
    """The rejection generalizes beyond a continuous, terminal target.

    `_has_valid_state_handoff` does not distinguish grid dtype or whether the
    target itself has further transitions — a discrete-grid state on a
    non-terminal target (one that transitions on to a further regime) must
    be rejected exactly like the continuous/terminal case.
    """

    @categorical(ordered=False)
    class _Outcome:
        low: ScalarInt
        high: ScalarInt

    @categorical(ordered=False)
    class _ThreeRegimeId:
        source: ScalarInt
        target: ScalarInt
        terminal: ScalarInt

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
                    transition={"terminal": MarkovTransition(_one_probability)},
                    states={"shock": DiscreteGrid(_Outcome)},
                    # Target's own outgoing (target -> terminal) law satisfies
                    # completeness; it says nothing about the incoming
                    # (source -> target) edge under test.
                    state_transitions={"shock": fixed_transition("shock")},
                    functions={"utility": _shock_utility},
                ),
                "terminal": Regime(
                    transition=None,
                    functions={"utility": _zero_utility},
                ),
            },
            ages=AgeGrid(start=20, stop=23, step="Y"),
            regime_id_class=_ThreeRegimeId,
            enable_jit=False,
        )


def test_target_only_nonprocess_state_with_entry_law_solves() -> None:
    """An explicit entry law supplies the target interpolation coordinate."""

    def _enter_shock() -> ScalarFloat:
        return jnp.float32(0.5)

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={"shock": {"target": _enter_shock}},
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

    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")
    np.testing.assert_allclose(np.asarray(solution[0]["source"]), 0.5, atol=1e-6)


def test_coarse_transition_validates_each_activity_compatible_candidate() -> None:
    """A function-based transition must be valid for every retained candidate."""
    with pytest.raises(ModelInitializationError, match=r"source.*target.*shock"):
        _build_overlapping_model(coarse=True)
