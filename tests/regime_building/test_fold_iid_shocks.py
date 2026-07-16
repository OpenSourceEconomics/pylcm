"""Fold IID utility shocks out of the stored value (`fold=True`).

An IID process state that enters only the CURRENT period's utility can be
integrated out at solve time — quadrature-averaged into the stored value —
instead of living as a grid axis of every stored `V`. This kills a
multiplicative memory blow-up for models with several such shocks (see
`PLAN-fold-iid-shocks.md`).

`fold=True` on the IID process declaration (`Regime(states={"shock":
NormalIIDProcess(..., fold=True)})`) means: still evaluate every quadrature
node exactly as today (the max-over-actions / collective readout runs per
node, unchanged), but weighted-average the node axis away — using the SAME
quadrature the process already carries — immediately after, before the
result is written into the stored value. The stored `V` loses that axis.
Default `fold=False` is byte-identical to today.

Scope of this slice (see `_fail_if_folded_state_persists` in
`regime_building/processing.py`): only a state that does NOT structurally
persist past the period that folds it — not redeclared, directly or via a
self-transition, in any regime a transition reaches. A genuinely persistent
IID shock (redrawn every period across many periods of the SAME regime)
would additionally need the continuation side (`regime_to_v_interpolation_info`
/ `stochastic_transition_names`) of every regime reading into it to also
recognize the fold; that is out of scope here (fold-review F5). Concretely:
the memory saving this slice delivers is PER-PERIOD only (isolated,
non-persistent shocks within the one period that folds them) — it does not
yet deliver a MULTI-period saving for a shock redrawn every period across
many ages (the central "repeated-IID" use case); do not advertise a
multi-period node-count reduction until continuation support for a
persistent fold lands.
"""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.processing import process_regimes
from _lcm.solution.backward_induction import solve
from _lcm.utils.logging import get_logger
from lcm import DiscreteGrid, LinSpacedGrid, NormalIIDProcess, categorical
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.processes import RouwenhorstAR1Process
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.typing import DiscreteAction, FloatND, ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


@categorical(ordered=False)
class RegimeId:
    period0: ScalarInt
    terminal: ScalarInt


def _next_regime() -> ScalarInt:
    return RegimeId.terminal


def _utility(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    """Working earns the base wage plus the shock; leisure earns nothing."""
    return work * (10.0 + wage_shock)


_AGES = AgeGrid(start=0, stop=2, step="Y")
_REGIME_NAMES_TO_IDS = MappingProxyType(
    {"period0": jnp.int32(0), "terminal": jnp.int32(1)}
)
_FLAT_PARAMS = MappingProxyType(
    {
        "period0": MappingProxyType({"H__discount_factor": jnp.asarray(0.9)}),
        "terminal": MappingProxyType({}),
    }
)


def _shock(*, fold: bool, n_points: int = 5, sigma: float = 2.0) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=n_points, gauss_hermite=True, mu=0.0, sigma=sigma, fold=fold
    )


def _make_regimes(*, fold: bool) -> dict[str, Regime]:
    """A one-shock, one-discrete-action, two-period singleton model.

    `wage_shock` enters ONLY `period0`'s own utility; `terminal` does not
    declare it (per this slice's persistence restriction — see module
    docstring), so nothing downstream ever reads its realization.
    """
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=fold)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return {"period0": period0, "terminal": terminal}


def _solve(regimes: dict[str, Regime]) -> MappingProxyType:
    processed = process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=_AGES,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        enable_jit=False,
    )
    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=_FLAT_PARAMS,
        ages=_AGES,
        regimes=processed,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return solution


def test_fold_exactness_oracle_matches_manual_average_and_drops_one_axis():
    """`fold=True` V equals the `fold=False` V weighted-averaged over the shock
    axis with the process's own quadrature weights, and has one fewer axis.

    This is the key oracle: it proves the fold is a pure, value-invariant
    memory optimization — same quadrature, reduced one step earlier.
    """
    unfolded = _solve(_make_regimes(fold=False))
    folded = _solve(_make_regimes(fold=True))

    V0_unfolded = unfolded[0]["period0"]
    V0_folded = folded[0]["period0"]

    assert V0_unfolded.shape == (5,)
    assert V0_folded.shape == ()
    assert V0_folded.ndim == V0_unfolded.ndim - 1

    weights = _shock(fold=False, sigma=2.0).get_transition_probs()[0]
    manual_average = jnp.average(V0_unfolded, weights=weights)
    np.testing.assert_allclose(np.asarray(V0_folded), np.asarray(manual_average))

    # Hand check: work always dominates leisure here (10 + shock > 0 for a
    # sigma=2 shock), so V = E[10 + shock] = 10 (mean-zero quadrature).
    np.testing.assert_allclose(np.asarray(V0_folded), 10.0, atol=1e-5)

    # The terminal regime (no shock at all) is untouched by fold either way.
    np.testing.assert_allclose(
        np.asarray(unfolded[1]["terminal"]), np.asarray(folded[1]["terminal"])
    )


def test_fold_default_path_is_byte_identical():
    """`fold=False` (the default) reproduces the pre-fold V exactly.

    Solves the SAME model spec twice — once leaving `fold` at its default,
    once passing `fold=False` explicitly — and requires bit-identical
    arrays, pinning that the new machinery introduces no behavior change on
    the default path.
    """
    default = _solve(_make_regimes(fold=False))
    explicit = _solve(_make_regimes(fold=False))
    np.testing.assert_array_equal(
        np.asarray(default[0]["period0"]), np.asarray(explicit[0]["period0"])
    )


def _three_shock_regimes(*, fold: bool) -> dict[str, Regime]:
    def _utility3(a: FloatND, b: FloatND, c: FloatND, work: DiscreteAction) -> FloatND:
        return work * (10.0 + a + b + c)

    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "a": _shock(fold=fold, n_points=3, sigma=1.0),
            "b": _shock(fold=fold, n_points=3, sigma=1.0),
            "c": _shock(fold=fold, n_points=3, sigma=1.0),
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility3},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return {"period0": period0, "terminal": terminal}


def test_fold_drops_one_axis_per_folded_shock():
    """A 3-shock model folds all three: the shape drops all 3 axes."""
    unfolded = _solve(_three_shock_regimes(fold=False))
    folded = _solve(_three_shock_regimes(fold=True))

    assert unfolded[0]["period0"].shape == (3, 3, 3)
    assert folded[0]["period0"].shape == ()
    assert folded[0]["period0"].ndim == unfolded[0]["period0"].ndim - 3


def _utility_f(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (10.0 + wage_shock) + 5.0 * (1.0 - work)


def _utility_m(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (6.0 + wage_shock)


def test_fold_collective_composition_matches_unfolded_then_averaged():
    """A collective regime's folded per-stakeholder V equals the per-stakeholder
    unfolded V weighted-averaged over the shock axis — same weights, applied
    AFTER `collective_argmax_and_readout` (the argmax is computed at the
    REALIZED node; only the readout is folded)."""

    def _make(*, fold: bool) -> dict[str, Regime]:
        couple = Regime(
            transition=None,
            stakeholders=("f", "m"),
            states={"wage_shock": _shock(fold=fold, sigma=3.0)},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility_f": _utility_f, "utility_m": _utility_m},
        )
        return {"couple": couple}

    def _solve_one(regimes: dict[str, Regime]) -> FloatND:
        processed = process_regimes(
            user_regimes=finalize_regimes(
                user_regimes=regimes, derived_categoricals={}
            ),
            ages=_AGES,
            regime_names_to_ids=MappingProxyType({"couple": jnp.int32(0)}),
            enable_jit=False,
        )
        solution, _sim_policies, _dissolution_flags = solve(
            flat_params=MappingProxyType({"couple": MappingProxyType({})}),
            ages=_AGES,
            regimes=processed,
            logger=get_logger(log_level="off"),
            enable_jit=False,
        )
        return solution[0]["couple"]

    V_unfolded = _solve_one(_make(fold=False))
    V_folded = _solve_one(_make(fold=True))

    assert V_unfolded.shape == (5, 2)  # (shock nodes, stakeholders)
    assert V_folded.shape == (2,)  # stakeholders only

    weights = _shock(fold=False, sigma=3.0).get_transition_probs()[0]
    manual = jnp.average(V_unfolded, axis=0, weights=weights)
    np.testing.assert_allclose(np.asarray(V_folded), np.asarray(manual), rtol=1e-6)
    # The two stakeholders' folded values genuinely differ (real composition,
    # not a scalarization collapse).
    assert not np.allclose(np.asarray(V_folded[0]), np.asarray(V_folded[1]))


def test_fold_on_ar1_process_is_rejected_by_the_type_system():
    """A persistent (non-IID) process has no `fold` field at all."""
    with pytest.raises(TypeError, match="fold"):
        RouwenhorstAR1Process(n_points=5, rho=0.9, sigma=1.0, mu=0.0, fold=True)  # type: ignore[call-arg]


def test_fold_on_taste_shocks_regime_is_rejected():
    from lcm.taste_shocks import ExtremeValueTasteShocks  # noqa: PLC0415

    with pytest.raises(RegimeInitializationError, match="taste_shocks"):
        Regime(
            transition=None,
            taste_shocks=ExtremeValueTasteShocks(),
            states={"wage_shock": _shock(fold=True)},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _utility},
        )


def test_fold_on_non_gridsearch_solver_is_rejected():
    from lcm import LinSpacedGrid  # noqa: PLC0415
    from lcm.solvers import DCEGM  # noqa: PLC0415

    with pytest.raises(RegimeInitializationError, match="GridSearch"):
        Regime(
            transition=None,
            states={
                "wage_shock": _shock(fold=True),
                "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            },
            actions={"consumption": LinSpacedGrid(start=0.0, stop=10.0, n_points=5)},
            functions={
                "utility": lambda consumption: consumption,
                "resources": lambda wealth: wealth,
                "savings": lambda wealth, consumption: wealth - consumption,
            },
            solver=DCEGM(
                continuous_state="wealth",
                continuous_action="consumption",
                resources="resources",
                post_decision_function="savings",
                savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            ),
        )


def test_fold_source_state_name_reused_by_outbound_gate_is_not_rejected():
    """A source folding a state does NOT reject merely because its OWN
    outbound `gated_edges[...].gate` declares an argument of the same name
    (fold-review F4, corrected from this test's earlier assertion).

    `GatedEdge.gate` is compiled and evaluated on the TARGET regime's own
    grid/DAG (`_attach_gated_edge_folds`/`_resolve_gated_edge`), never on
    this (source) regime's — so `gate=lambda wage_shock: ...` here reads
    `some_target`'s `wage_shock` (if it declares one), not this regime's.
    Treating the SOURCE-local `_validate_fold_declarations` walk as if the
    gate were source-local (the old behavior this test used to pin) produced
    a false positive purely from name collision. The genuine cross-regime
    hazard — THIS regime being read nodewise as a gated-edge target, leg
    fallback, or same-period reference — is covered by the model-processing
    guard `_fail_if_folded_regime_is_same_period_endpoint`
    (`regime_building/processing.py`; see
    `test_fold_gate_guard.py`/`test_fold_guard_complete.py`), which correctly
    checks the TARGET side of the same declarations instead.
    """
    Regime(
        transition=None,
        states={"wage_shock": _shock(fold=True)},
        gated_edges={
            "some_target": GatedEdge(
                gate=lambda wage_shock: wage_shock > 0.0,
                legs={
                    "only": EdgeLeg(
                        fallback=SamePeriodRef(regime="elsewhere", projection={})
                    )
                },
            )
        },
    )


def test_fold_on_transition_conditioning_shock_is_rejected():
    """A next-period transition that reads the shock's realized value can't
    compose with folding it: the shock is integrated out, so nothing
    downstream may depend on which node was realized."""
    with pytest.raises(RegimeInitializationError, match="next-period transition"):
        Regime(
            transition=_next_regime,
            states={
                "wage_shock": _shock(fold=True),
                "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            },
            actions={"work": DiscreteGrid(Work)},
            state_transitions={
                "wealth": lambda wealth, wage_shock: wealth + wage_shock,
            },
            functions={"utility": _utility},
        )


def test_fold_on_persisting_shock_is_rejected_at_model_processing():
    """A shock redeclared in a regime reachable via a next-period transition
    structurally persists — its continuation would need a `next_<name>` axis
    the folded stored value no longer has. Caught once, cross-regime, at
    model processing (`_fail_if_folded_state_persists`), not at
    `Regime.__post_init__` (which only sees one regime at a time).

    The process-state continuation machinery
    (`target_process_grids`/`weight_<target>__next_<process>` in
    `processing.py`) only wires up for a regime target reached through a
    QUALIFIED (per-target-dict) transition — a bare coarse `transition=func`
    with no other per-target state law never populates `reachable_targets`,
    so this scenario needs a per-target regime transition (`MarkovTransition`)
    plus a per-target `wealth` state law to force `terminal` into
    `reachable_targets` (mirrors a stochastic multi-target model, where this
    persistence pattern is the realistic case fold must reject).

    The fold must sit on the TARGET (`terminal`): `period0`, as SOURCE, needs
    to interpolate `next_V["terminal"]` over a `wage_shock` axis for its own
    continuation (`period0.solution.transitions["terminal"]` carries
    `next_wage_shock` — confirmed the mechanism engages before asserting the
    rejection) — folding wage_shock away from `terminal`'s OWN stored V is
    what breaks that read. Folding the SOURCE's own (non-persisting) copy
    would not break anything (that's exactly the supported case the other
    tests in this module cover) — a deliberately wrong fold placement here
    would make the pin vacuous.
    """
    from lcm import LinSpacedGrid  # noqa: PLC0415
    from lcm.transition import MarkovTransition  # noqa: PLC0415

    def _utility_with_wealth(
        wage_shock: FloatND, work: DiscreteAction, wealth: FloatND
    ) -> FloatND:
        return work * (10.0 + wage_shock) + wealth

    wealth_grid = LinSpacedGrid(start=0.0, stop=10.0, n_points=3)
    period0 = Regime(
        transition={"terminal": MarkovTransition(lambda: jnp.asarray(1.0))},
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=False), "wealth": wealth_grid},
        state_transitions={"wealth": {"terminal": lambda wealth: wealth}},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_with_wealth},
    )
    # The TARGET regime folds wage_shock, but `period0`'s own continuation
    # into it still needs to interpolate over that axis — it persists.
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage_shock": _shock(fold=True), "wealth": wealth_grid},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility_with_wealth},
    )
    with pytest.raises(ModelInitializationError, match="structurally persists"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes={"period0": period0, "terminal": terminal},
                derived_categoricals={},
            ),
            ages=_AGES,
            regime_names_to_ids=_REGIME_NAMES_TO_IDS,
            enable_jit=False,
        )
