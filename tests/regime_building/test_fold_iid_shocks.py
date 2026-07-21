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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.finalize import finalize_regimes
from _lcm.regime_building.max_Q_over_a import _select_fold_reducer
from _lcm.regime_building.processing import process_regimes
from _lcm.regime_building.zero_safe import zero_safe_average
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


def _make_regimes(
    *, fold: bool, n_points: int = 5, sigma: float = 2.0
) -> dict[str, Regime]:
    """A one-shock, one-discrete-action, two-period singleton model.

    `wage_shock` enters ONLY `period0`'s own utility; `terminal` does not
    declare it (per this slice's persistence restriction — see module
    docstring), so nothing downstream ever reads its realization.

    `n_points`/`sigma` are exposed because the fold's exactness is a
    FLOATING-POINT property: whether a given quadrature actually exercises the
    rounding difference between the two reduction kernels depends on the node
    values. The defaults do NOT (measured), so a test pinning exactness must
    pick a configuration that does — see
    `test_fold_is_bit_exact_against_unfolded_then_averaged`.
    """
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=fold, n_points=n_points, sigma=sigma)},
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


def _make_regimes_fold_omitted() -> dict[str, Regime]:
    """`_make_regimes`'s model with `fold` never passed to the process at all.

    Deliberately does NOT route through `_shock`, which always passes an
    explicit `fold=`: the whole point is to construct a `NormalIIDProcess`
    with no `fold` argument, so the DEFAULT is what gets exercised.
    """
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "wage_shock": NormalIIDProcess(
                n_points=5, gauss_hermite=True, mu=0.0, sigma=2.0
            )
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return {"period0": period0, "terminal": terminal}


def test_fold_default_path_is_byte_identical():
    """Omitting `fold` entirely is bit-identical to passing `fold=False`.

    Pins that `fold`'s DEFAULT really is the pre-fold path. The two branches
    must construct genuinely DIFFERENT declarations — an omitted `fold` vs an
    explicit `fold=False` — or this compares a spec with itself and pins
    nothing but determinism (fold-review J3: it previously did exactly that,
    since both branches went through `_shock`, which always passes `fold=`).
    """
    omitted = _make_regimes_fold_omitted()["period0"].states["wage_shock"]
    explicit = _make_regimes(fold=False)["period0"].states["wage_shock"]
    # Guard the guard: the default is what makes the omitted branch meaningful.
    assert omitted.fold is False
    assert explicit.fold is False
    assert omitted == explicit  # identical spec, reached two different ways

    default_V = _solve(_make_regimes_fold_omitted())
    explicit_V = _solve(_make_regimes(fold=False))
    np.testing.assert_array_equal(
        np.asarray(default_V[0]["period0"]), np.asarray(explicit_V[0]["period0"])
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


def _solve_jit(regimes: dict[str, Regime], *, enable_jit: bool) -> MappingProxyType:
    """`_solve`, but with `enable_jit` under the caller's control.

    The fold's exactness contract must hold on BOTH paths, and they are not
    the same path: the jitted core closes over the fold weights as compile-time
    constants (XLA constant-folds them), while the non-jitted core executes the
    reduction eagerly. Fold-review F1 was invisible precisely because every
    other test in this module pins only `enable_jit=False`.
    """
    processed = process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=_AGES,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        enable_jit=enable_jit,
    )
    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=_FLAT_PARAMS,
        ages=_AGES,
        regimes=processed,
        logger=get_logger(log_level="off"),
        enable_jit=enable_jit,
    )
    return solution


def _bits(x: FloatND) -> int:
    """The exact bit pattern of a float scalar, as an int.

    `assert_allclose` cannot see a 1-ULP defect; a feature whose contract is
    EXACT equality has to be pinned on the bits themselves (fold-review J4).

    Reads the array's OWN dtype — the suite runs float64 by default and
    float32 under `--precision=32`. Casting to a fixed width here would
    silently round the very last bit this helper exists to inspect.
    """
    arr = np.asarray(x)
    assert arr.shape == (), "scalar expected"
    assert arr.dtype in (np.float32, np.float64), f"unexpected dtype {arr.dtype}"
    return int(arr.view(np.uint32 if arr.dtype == np.float32 else np.uint64))


# A (n_points, sigma) whose Gauss-Hermite nodes MEASURABLY separate the two
# reduction kernels: `zero_safe_average`'s extra rounding lands on the other
# side of an ULP from `jnp.average`'s FMA-contracted one. Chosen by search to
# separate them in BOTH precisions the suite runs (float64 by default, float32
# under `--precision=32`).
#
# This matters more than it looks: the module defaults (5, 2.0) happen NOT to
# separate the kernels — their symmetric quadrature cancels exactly — so an
# exactness test built on them passes against the PRE-FIX code and pins
# nothing at all (fold-review F1/J4). `_average` is asymmetric here only
# because the sigma is large enough for the leisure floor to bind at the
# bottom node.
_DRIFT_N_POINTS = 5
_DRIFT_SIGMA = 3.0


def test_fold_is_bit_exact_against_unfolded_then_averaged():
    """The folded V is BIT-IDENTICAL to the unfolded V averaged with the same
    quadrature, on the NON-JITTED path.

    This is the fold's actual contract ("a pure, value-invariant memory
    optimization"), pinned on bit patterns rather than a tolerance.

    PROVEN fail-pre/pass-post (both `--precision=64` and `--precision=32`):
    pre-fix, `_wrap_with_fold_reduction` unconditionally used
    `zero_safe_average`, whose per-term `jnp.where` blocks XLA's FMA
    contraction, so a strictly-positive-weight fold drifted 1 ULP off the
    oracle (fold-review F1).

    Scoped to `enable_jit=False` deliberately — see
    `test_fold_jitted_matches_unfolded_then_averaged_within_one_ulp` for why
    the jitted path cannot be held to bit-equality.
    """
    kwargs = {"n_points": _DRIFT_N_POINTS, "sigma": _DRIFT_SIGMA}
    weights = _shock(fold=False, **kwargs).get_transition_probs()[0]

    unfolded_V = _solve_jit(_make_regimes(fold=False, **kwargs), enable_jit=False)[0][
        "period0"
    ]
    folded_V = _solve_jit(_make_regimes(fold=True, **kwargs), enable_jit=False)[0][
        "period0"
    ]
    oracle = jnp.average(unfolded_V, weights=weights)

    # Guard the guard #1: strictly positive weights, so this exercises the
    # all-positive path the fix is about — not the zero-weight one.
    assert bool(jnp.all(weights > 0))
    # Guard the guard #2: this configuration really does separate the two
    # kernels, so the assertion below has something to catch. The module
    # defaults do NOT — without this, a future edit to the fixture could
    # silently defang the test into passing against the pre-fix code.
    guarded = zero_safe_average(unfolded_V, axis=0, weights=weights)
    assert _bits(oracle) != _bits(guarded)

    assert _bits(folded_V) == _bits(oracle)


def test_fold_jitted_matches_unfolded_then_averaged_to_summand_scale_tolerance():
    """The JITTED fold matches the unfolded-then-averaged oracle to a SCALE-AWARE
    tolerance measured against the summand magnitude, NOT a fixed ULP count.

    Under `jit` the fold reduction is compiled INTO the surrounding solve kernel,
    and XLA's fusion/FMA/reassociation decisions there depend on the whole graph —
    not reproducible by any standalone oracle. The gap is the float32 REDUCTION
    error of the weighted summands, so the bound is summand-scale.

    Two refinements over a naive `rtol * max|summand|` (fold-round5 T2):

    1. The principled forward-error scale for a length-`n` weighted sum is the SUM
       of absolute weighted contributions `Σ_k |w_k V_k|`, not `max_k |w_k V_k|`.
       The two diverge sharply near cancellation: the round-5 review's executed
       192-node fixture had a gap of 255.6 epsilons times `max|w_k V_k|` but only
       1.42 epsilons times `Σ|w_k V_k|`, so a small fixed rtol on the max is not a
       general contract while the sum-scale one holds.
    2. The coefficient must be NODE-COUNT- and dtype-aware: the reduction accrues
       O(n) rounding steps, so a fixed node-count-independent rtol silently tightens
       or loosens as `n` grows. Use `c(n, dtype) = C * n * eps(dtype)`.

    Do NOT pin this to a fixed few-ULP count of the RESULT (fold-round4 F3): ULP is
    a result-space spacing metric and becomes unstable near CANCELLATION — the
    round-4 18-node fold differed by only ~2.62e-7 absolute yet 287,557 ULP in the
    small (~1e-5) cancelled result. This model's node values are ~10 (no
    cancellation), so the gap is at the float32 floor (here exactly 0), but the
    sum-scale node-count-aware bound is the one that also holds under cancellation.
    """
    kwargs = {"n_points": _DRIFT_N_POINTS, "sigma": _DRIFT_SIGMA}
    weights = _shock(fold=False, **kwargs).get_transition_probs()[0]

    unfolded_V = _solve_jit(_make_regimes(fold=False, **kwargs), enable_jit=True)[0][
        "period0"
    ]
    folded_V = _solve_jit(_make_regimes(fold=True, **kwargs), enable_jit=True)[0][
        "period0"
    ]
    oracle = jnp.average(unfolded_V, weights=weights)

    # atol + C * n * eps(dtype) * Σ|w_k V_k| — summand-scale, node-count- and
    # dtype-aware, stable under cancellation (fold-round5 T2). This is a
    # TOOLCHAIN-CHARACTERIZED contract, not a proved universal XLA bound: the
    # classical weighted-reduction forward error is ~gamma_n * Σ|w_k V_k| (with
    # gamma_n = n*u/(1-n*u)); `C = 16` is a conservative empirical factor that
    # additionally covers the fused-vs-materialized graph difference (products,
    # the `jnp.average` division) beyond the bare summation. It stays orders of
    # magnitude below any wrong-reducer gap (which would be O(node value), not
    # O(n * eps * Σ|wV|)). The absolute floor is dtype/value-scale aware rather
    # than a fixed 1e-6 (which is fine for float32 but needlessly loose for small
    # float64 values): scale it by eps(dtype) and the summand magnitude.
    n_nodes = int(weights.shape[0])
    eps = float(jnp.finfo(folded_V.dtype).eps)
    summand_scale = float(jnp.sum(jnp.abs(weights * unfolded_V)))
    atol = 8.0 * eps * max(summand_scale, 1.0)
    tol = atol + 16.0 * n_nodes * eps * summand_scale
    assert abs(float(folded_V) - float(oracle)) <= tol


def test_select_fold_reducer_takes_the_guard_only_when_a_weight_is_zero():
    """The zero-safe guard is bound per axis, at build time, from that axis's
    own weights.

    `zero_safe_average` costs an extra rounding (hence F1's 1-ULP drift) and
    ~3x the runtime; it protects only against `0 * ±inf = nan`, which cannot
    arise on an axis whose weights are all strictly positive. The weights are
    concrete at kernel-build time (`_validate_fold_declarations` rejects a
    runtime-parameterized process), so this is a plain Python branch — not a
    traced predicate, which could only ever decide globally.
    """
    assert (
        _select_fold_reducer(weight=jnp.array([0.25, 0.5, 0.25]), name="s")
        is jnp.average
    )
    assert (
        _select_fold_reducer(weight=jnp.array([0.0, 1.0, 0.0]), name="s")
        is zero_safe_average
    )
    # Per AXIS, not per model: each axis gets the kernel its own weights need.
    assert _select_fold_reducer(weight=jnp.array([1.0]), name="s") is jnp.average


def test_select_fold_reducer_rejects_non_concrete_weights():
    """A traced weight cannot pick a kernel at build time — fail loudly.

    `_validate_fold_declarations` is supposed to make this unreachable by
    rejecting a fold on a runtime-parameterized process. If that guarantee is
    ever bypassed, this must raise rather than silently fall back to one
    kernel for every axis.
    """

    def _build(w: FloatND) -> object:
        return _select_fold_reducer(weight=w, name="s")

    with pytest.raises(ValueError, match="not concrete at kernel-build time"):
        jax.jit(_build)(jnp.array([0.5, 0.5]))


def test_zero_weight_fold_axis_still_averages_infinities_safely():
    """The guard is still TAKEN where it is needed: a zero-weight fold node
    beside an admissible on-path `-inf` must not poison the fold average.

    The F1 fix narrows WHERE `zero_safe_average` is applied, so this pins that
    it is still applied on the zero-weight path — otherwise the fix would have
    traded a 1-ULP drift for a `nan`.
    """
    reducer = _select_fold_reducer(weight=jnp.array([0.0, 1.0, 0.0]), name="s")
    out = reducer(
        jnp.array([-jnp.inf, 4.0, jnp.inf]),
        axis=0,
        weights=jnp.array([0.0, 1.0, 0.0]),
    )
    assert reducer is zero_safe_average
    np.testing.assert_array_equal(np.asarray(out), np.float32(4.0))


def test_fold_on_persisting_shock_reached_only_via_regime_transition_is_rejected():
    """The persistence guard fires for a target reached ONLY by the per-target
    regime transition — with NO ordinary state law to carry it.

    The sibling test above deliberately adds a `wealth` law "SOLELY to force
    the target into reachable_targets". That admission was the bug
    (fold-review F2/J5): reachability was derived from ordinary state laws
    only, so WITHOUT such a law the target's process transitions were never
    built, its bundle stayed empty, the guard found no `next_wage_shock` to
    object to, and `get_period_targets` dropped the target from E[V] entirely
    — silently, since `get_period_targets` assumes an absent target "has no
    state and therefore zero value".

    PROVEN fail-pre/pass-post: pre-fix `process_regimes` returned normally
    with `period0.solution.transitions == {}`.
    """
    from lcm.transition import MarkovTransition  # noqa: PLC0415

    period0 = Regime(
        transition={"terminal": MarkovTransition(lambda: jnp.asarray(1.0))},
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=False)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
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


def test_coarse_regime_transition_does_not_fabricate_a_self_transition():
    """A coarse `transition=func`'s candidate universe is admitted as reachable
    EXCEPT the source regime itself, so it never fabricates a self-transition.

    A coarse `transition=func` emits a `next_regime` cell for EVERY regime —
    routing is decided at runtime from the returned id — so its cell keys are
    the CANDIDATE universe. Those candidates ARE admitted to `reachable_targets`
    (fold-round3 F1: omitting a genuinely-routed candidate silently drops its
    continuation), but two things keep that from fabricating a spurious
    continuation here: (1) the SOURCE regime is excluded, so no false
    `period0 -> period0` self-transition wires `period0`'s own folded
    `wage_shock` back into itself and trips a bogus "structurally persists";
    (2) process transitions are still scoped to the source's own processes, and
    `terminal` shares none, so admitting it builds nothing. This is the module's
    primary supported fold topology (shock declared and folded only in
    `period0`); it must still solve cleanly to `E[10 + shock] = 10`.

    MEASURED: the reviewer's original `reachable_targets |=
    set(next_regime_cells_by_target)` (candidates INCLUDING self) fabricates the
    self-transition and fails this model with a bogus persistence error; the
    minus-self admission does not.
    """
    solution = _solve(_make_regimes(fold=True))
    assert solution[0]["period0"].shape == ()
    np.testing.assert_allclose(np.asarray(solution[0]["period0"]), 10.0, atol=1e-5)


def test_coarse_regime_transition_to_persisting_fold_target_is_rejected():
    """A COARSE `transition=func` that can route to a folded target whose shock
    persists from the source is rejected at build time, requiring per-target
    cells (fold-round3 F1 / fold-round4 F1+F2).

    Before the coarse-candidate reachability fix, this target — reached only via
    coarse routing, folding a `wage_shock` that also lives in the source — was
    silently dropped from E[V], while the byte-identical PER-TARGET model raised
    "structurally persists". A coarse transition's actual support is unknown at
    build time, so rather than build a `next_wage_shock` edge whose persistence
    the structural guard would then judge on an UNKNOWN-support candidate (which
    would wrongly accept a real self-fold or wrongly reject a never-returned one),
    the ambiguous folded-coarse topology is rejected here with a clear
    "use per-target transitions" scope error. Declaring the per-target form then
    routes it into the exact persistence guard
    (`test_fold_on_persisting_shock_reached_only_via_regime_transition_is_rejected`).
    """
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=False)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    with pytest.raises(ModelInitializationError, match="explicit PER-TARGET cell"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes={"period0": period0, "terminal": terminal},
                derived_categoricals={},
            ),
            ages=_AGES,
            regime_names_to_ids=_REGIME_NAMES_TO_IDS,
            enable_jit=False,
        )


def _u_source_shock(source_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (10.0 + source_shock)


def _u_target_shock(target_shock: FloatND, work: DiscreteAction) -> FloatND:
    return work * (10.0 + target_shock)


def _make_target_local_fold_regimes(*, shared: bool) -> dict[str, Regime]:
    """`period0` coarse-routes to `terminal`, which folds a process.

    `shared=False`: `terminal` folds a TARGET-LOCAL `target_shock` whose name the
    source (`source_shock`) does not carry -- no `next_target_shock` edge can be
    auto-wired from the source, so the fold cannot persist across the coarse edge.
    `shared=True`: `terminal` folds the SAME name the source carries -- the
    genuinely ambiguous round-4 case.
    """
    fold_name = "source_shock" if shared else "target_shock"
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"source_shock": _shock(fold=False)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_source_shock},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={fold_name: _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_source_shock if shared else _u_target_shock},
    )
    return {"period0": period0, "terminal": terminal}


def test_coarse_candidate_folding_a_target_local_process_is_not_rejected():
    """fold-round5 F1: the active-period scope fence must key on the SOURCE's own
    process names, not every folded process in the candidate target.

    `terminal` folds a target-local `target_shock` the source never carries, so no
    `next_target_shock` continuation can persist across the coarse edge -- there is
    nothing for the persistence guard to validate and the model is unambiguous.
    Pre-fix it was rejected solely because the check ignored process provenance;
    it must now build AND solve, with the fold axis integrated out of `terminal`'s
    stored value.
    """
    processed = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_target_local_fold_regimes(shared=False),
            derived_categoricals={},
        ),
        ages=_AGES,
        regime_names_to_ids=_REGIME_NAMES_TO_IDS,
        enable_jit=False,
    )
    solution, _s, _d = solve(
        flat_params=_FLAT_PARAMS,
        ages=_AGES,
        regimes=processed,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    # The folded process leaves no axis on the terminal value.
    assert solution[1]["terminal"].shape == ()


def test_coarse_candidate_folding_a_source_carried_process_is_still_rejected():
    """fold-round5 F1 negative control: when the folded name IS carried by the
    source, persistence across the coarse edge is genuinely possible, so the
    ambiguous topology must still be rejected (round-4 behaviour preserved)."""
    with pytest.raises(ModelInitializationError, match="explicit PER-TARGET cell"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes=_make_target_local_fold_regimes(shared=True),
                derived_categoricals={},
            ),
            ages=_AGES,
            regime_names_to_ids=_REGIME_NAMES_TO_IDS,
            enable_jit=False,
        )


def test_coarse_self_transition_retains_the_self_continuation():
    """A coarse transition that returns its OWN regime keeps the self-continuation
    in E[V] (fold-round4 F1: excluding the source by name dropped it).

    `stay` is active for two periods and coarse-routes to itself over a live
    (non-folded) `wage_shock`; its next-period self-value must enter E[V]. Pre-fix
    (source excluded) the `stay` self-target was dropped and its continuation was
    zero; now `stay` is admitted and appears as its own transition target.
    """
    ages3 = AgeGrid(start=0, stop=3, step="Y")
    ids = MappingProxyType({"stay": jnp.int32(0), "done": jnp.int32(1)})
    params = MappingProxyType(
        {
            "stay": MappingProxyType({"H__discount_factor": jnp.asarray(0.9)}),
            "done": MappingProxyType({}),
        }
    )

    def _next_self() -> ScalarInt:
        return jnp.int32(0)  # always return "stay"

    stay = Regime(
        transition=_next_self,
        active=lambda age: age < 2,
        states={"wage_shock": _shock(fold=False)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 2,
        functions={"utility": lambda: 0.0},
    )
    processed = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes={"stay": stay, "done": done}, derived_categoricals={}
        ),
        ages=ages3,
        regime_names_to_ids=ids,
        enable_jit=False,
    )
    core = getattr(processed["stay"], "solution", processed["stay"])
    assert "stay" in dict(getattr(core, "transitions", {}) or {}), (
        "the coarse self-transition must retain 'stay' as its own continuation target"
    )
    solution, _s, _d = solve(
        flat_params=params,
        ages=ages3,
        regimes=processed,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    # A live self-continuation lifts the value above the one-period utility (~10).
    assert float(jnp.mean(solution[0]["stay"])) > 10.5


def test_coarse_self_fold_is_rejected():
    """A coarse transition that can return its own regime while that regime FOLDS
    a shock is rejected (fold-round4 F1: it must not silently bypass the
    persistence guard).

    `stay` is active two periods, folds `wage_shock`, and coarse-routes to itself,
    so the folded shock could persist across the self-edge. Support is unknown at
    build time, so this is rejected with the per-target scope error.
    """
    ages3 = AgeGrid(start=0, stop=3, step="Y")
    ids = MappingProxyType({"stay": jnp.int32(0), "done": jnp.int32(1)})

    def _next_self() -> ScalarInt:
        return jnp.int32(0)

    stay = Regime(
        transition=_next_self,
        active=lambda age: age < 2,
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 2,
        functions={"utility": lambda: 0.0},
    )
    with pytest.raises(ModelInitializationError, match="explicit PER-TARGET cell"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes={"stay": stay, "done": done}, derived_categoricals={}
            ),
            ages=ages3,
            regime_names_to_ids=ids,
            enable_jit=False,
        )


def test_unreachable_folded_coarse_candidate_is_rejected_with_scope_error():
    """A coarse function that never returns a folded candidate is still rejected
    with the per-target scope error, NOT a misleading 'structurally persists'
    (fold-round4 F2).

    The persistence guard is structural and cannot see that the coarse function
    always returns `stay` (so folded candidate `alt` has probability zero). Rather
    than build a spurious `next_wage_shock` edge into `alt` and let the guard
    manufacture a persistence error, the ambiguous folded-coarse topology is
    rejected up front with the actionable per-target message.
    """
    ages3 = AgeGrid(start=0, stop=3, step="Y")
    ids = MappingProxyType(
        {"src": jnp.int32(0), "stay": jnp.int32(1), "alt": jnp.int32(2)}
    )

    def _always_stay() -> ScalarInt:
        return jnp.int32(1)  # always "stay", never "alt"

    src = Regime(
        transition=_always_stay,
        active=lambda age: age < 1,
        states={"wage_shock": _shock(fold=False)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    stay = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    alt = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wage_shock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    with pytest.raises(ModelInitializationError, match="explicit PER-TARGET cell"):
        process_regimes(
            user_regimes=finalize_regimes(
                user_regimes={"src": src, "stay": stay, "alt": alt},
                derived_categoricals={},
            ),
            ages=ages3,
            regime_names_to_ids=ids,
            enable_jit=False,
        )


def test_coarse_regime_transition_to_shared_process_target_builds_continuation():
    """A coarse transition to a target that shares a NON-folded process with the
    source now BUILDS that target's continuation into E[V] — the positive half of
    the fold-round3 F1 fix — matching the per-target form value-for-value.

    `wage_shock` is live (not folded) in both regimes, so it persists
    source->target and the continuation must interpolate over it. Pre-fix the
    coarse form dropped `terminal` entirely (empty `period0` bundle, continuation
    = 0); post-fix it equals the per-target form, which never dropped it.
    """
    from lcm.transition import MarkovTransition  # noqa: PLC0415

    def _terminal() -> Regime:
        return Regime(
            transition=None,
            active=lambda age: age >= 1,
            states={"wage_shock": _shock(fold=False)},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": lambda wage_shock, work: work * (2.0 + wage_shock)},
        )

    def _period0(transition: object) -> Regime:
        return Regime(
            transition=transition,
            active=lambda age: age < 1,
            states={"wage_shock": _shock(fold=False)},
            actions={"work": DiscreteGrid(Work)},
            functions={"utility": _utility},
        )

    coarse = _solve({"period0": _period0(_next_regime), "terminal": _terminal()})
    per_target = _solve(
        {
            "period0": _period0(
                {"terminal": MarkovTransition(lambda: jnp.asarray(1.0))}
            ),
            "terminal": _terminal(),
        }
    )
    # Guard the guard: the per-target continuation is genuinely present (the
    # discounted terminal value lifts period0 above its own shock-only ~10).
    assert float(jnp.mean(per_target[0]["period0"])) > 11.0
    # The coarse form no longer drops it: value-for-value equal to per-target.
    np.testing.assert_allclose(
        np.asarray(coarse[0]["period0"]), np.asarray(per_target[0]["period0"])
    )


# --------------------------------------------------------------------------------------
# fold-review F2 (fold-only continuation): a target whose ONLY state is a
# target-local folded IID process must still enter E[V] — its stored V is a
# SCALAR (the folded axis is integrated out), so it needs an empty transition
# bundle that keeps it enumerable by `get_period_targets`, and its continuation
# must be read as that scalar (no interpolation coordinate).
# --------------------------------------------------------------------------------------


@categorical(ordered=False)
class _RouteRegimeId:
    src: ScalarInt
    folded_B: ScalarInt
    dead_C: ScalarInt


def _make_route_to_folded_target_regimes() -> dict[str, Regime]:
    """Binary-action source routes to a folded-only target B or a worthless C.

    `src` (period 0) has NO states, only a binary `work` action:

    - `work == work` (code 1) routes, per-target, to `folded_B`; its immediate
      utility is 0.
    - `work == leisure` (code 0) routes to `dead_C`; its immediate utility is
      0.5.

    `folded_B` (period 1, terminal) declares a SINGLE state: a target-local
    `NormalIIDProcess(fold=True)` the source does not carry. Its utility folds
    to the constant 1.0 (mean-zero shock), so its stored V is the scalar 1.0.
    `dead_C` (period 1, terminal) is stateless with utility 0 — genuinely
    worthless, so dropping it from E[V] is harmless.

    With `discount > 0.5` the correct choice is to route to B (continuation
    `discount * 1.0`) rather than take C's immediate 0.5. If B's folded scalar
    continuation is silently dropped, `src` wrongly prefers C — a reversed
    policy. `src` has no states, so its stored V is a single scalar equal to
    the value of the chosen action.
    """
    from lcm.transition import MarkovTransition  # noqa: PLC0415

    def _route_to_B(work: DiscreteAction) -> FloatND:
        return jnp.asarray(work, dtype=float)

    def _route_to_C(work: DiscreteAction) -> FloatND:
        return 1.0 - jnp.asarray(work, dtype=float)

    def _u_src(work: DiscreteAction) -> FloatND:
        # leisure (code 0) -> 0.5; work (code 1) -> 0.0
        return 0.5 * (1.0 - jnp.asarray(work, dtype=float))

    def _u_folded_B(bshock: FloatND, work: DiscreteAction) -> FloatND:
        # Mean-zero shock folds away; the constant 1.0 survives. `work` is
        # inert so the max-over-actions is the folded 1.0.
        return 1.0 + bshock + 0.0 * jnp.asarray(work, dtype=float)

    src = Regime(
        transition={
            "folded_B": MarkovTransition(_route_to_B),
            "dead_C": MarkovTransition(_route_to_C),
        },
        active=lambda age: age < 1,
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
    )
    folded_B = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"bshock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_folded_B},
    )
    dead_C = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return {"src": src, "folded_B": folded_B, "dead_C": dead_C}


def _solve_route(regimes: dict[str, Regime], *, discount: float) -> MappingProxyType:
    processed = process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=_AGES,
        regime_names_to_ids=MappingProxyType(
            {"src": jnp.int32(0), "folded_B": jnp.int32(1), "dead_C": jnp.int32(2)}
        ),
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(discount)}),
            "folded_B": MappingProxyType({}),
            "dead_C": MappingProxyType({}),
        }
    )
    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=processed,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    return solution


def test_folded_only_per_target_continuation_enters_expected_value():
    """A folded-only target reached by a per-target transition enters E[V].

    fold-review F2 (folded slice): `folded_B`'s only state is a target-local
    folded IID process, so its stored V is the scalar 1.0. Pre-fix, `src`'s
    transition bundle was empty for BOTH targets (no state law, no source
    process edge), so `get_period_targets` enumerated neither and E[V] was
    identically zero — `src` took `dead_C`'s immediate 0.5 (V_src = 0.5), a
    REVERSED policy. Post-fix, the folded target keeps an explicit empty bundle
    (enumerable, read as its scalar V), so routing to B yields the discounted
    continuation and `src` prefers it.

    PROVEN fail-pre/pass-post: pre-fix V_src == 0.5 (routes to worthless C);
    post-fix V_src == discount * 1.0 (routes to the value-1 folded target).
    """
    discount = 0.9
    solution = _solve_route(_make_route_to_folded_target_regimes(), discount=discount)
    # `src` has no states: a single scalar equal to the chosen action's value.
    V_src = np.asarray(solution[0]["src"])
    assert V_src.shape == ()
    # The folded target's stored V is the scalar 1.0 (shock integrated out).
    np.testing.assert_allclose(np.asarray(solution[1]["folded_B"]), 1.0, atol=1e-5)
    # Post-fix: route to B, value = discount * 1.0 = 0.9 (> C's immediate 0.5).
    np.testing.assert_allclose(V_src, discount, atol=1e-5)


def test_folded_only_per_target_target_is_enumerable_in_transitions():
    """The folded-only target keeps an (empty) bundle so it stays enumerable.

    Structural companion to the value test: pre-fix `src.solution.transitions`
    was empty (the folded-only target was dropped); post-fix it carries a
    `folded_B` key with an empty bundle (no state law / process edge needed —
    its V is scalar). `dead_C` stays absent: it is genuinely stateless and
    worthless, so the general non-folded empty-bundle hole stays deferred.
    """
    processed = process_regimes(
        user_regimes=finalize_regimes(
            user_regimes=_make_route_to_folded_target_regimes(),
            derived_categoricals={},
        ),
        ages=_AGES,
        regime_names_to_ids=MappingProxyType(
            {"src": jnp.int32(0), "folded_B": jnp.int32(1), "dead_C": jnp.int32(2)}
        ),
        enable_jit=False,
    )
    transitions = dict(processed["src"].solution.transitions)
    assert "folded_B" in transitions
    assert dict(transitions["folded_B"]) == {}
    assert "dead_C" not in transitions


# --------------------------------------------------------------------------------------
# simulate-side parity for fold-round6/round7: pylcm's simulate RE-OPTIMIZES Q over
# the grid (it does not interpolate the stored policy), so it reads the continuation
# exactly as solve does. The folded-only per-target continuation must therefore enter
# the SIMULATED argmax the same way it enters the solved E[V] — the folded target must
# stay enumerable AND be read as its scalar V (no phantom `next_<shock>` coordinate).
# --------------------------------------------------------------------------------------


def _make_route_to_folded_target_regimes_stateful() -> dict[str, Regime]:
    """`_make_route_to_folded_target_regimes` with an inert `wealth` state on `src`.

    Identical routing/values, but `src` declares a continuous `wealth` state that
    does not enter utility (`+ 0.0 * wealth`) and is not carried to any target. It
    exists only to give the forward simulation a per-subject state axis: a stateless
    SINGLETON regime with actions is a separate, pre-existing simulate limitation (a
    0-d argmax index reaches `vmapped_unravel_index`; the guard at
    `simulate._simulate_regime_in_period` only broadcasts the stateless COLLECTIVE
    case). Adding the state isolates the fold-only continuation behavior under test.

    The fold-only bug is preserved: `folded_B`'s only state is still the target-local
    folded `bshock`, and `wealth` is a plain (non-process) state the source does not
    carry into `folded_B`, so `src`'s transition bundle to `folded_B` is still empty.
    """
    from lcm import LinSpacedGrid, fixed_transition  # noqa: PLC0415
    from lcm.transition import MarkovTransition  # noqa: PLC0415

    def _route_to_B(work: DiscreteAction) -> FloatND:
        return jnp.asarray(work, dtype=float)

    def _route_to_C(work: DiscreteAction) -> FloatND:
        return 1.0 - jnp.asarray(work, dtype=float)

    def _u_src(work: DiscreteAction, wealth: FloatND) -> FloatND:
        # leisure (code 0) -> 0.5; work (code 1) -> 0.0. `wealth` is inert.
        return 0.5 * (1.0 - jnp.asarray(work, dtype=float)) + 0.0 * wealth

    def _u_folded_B(bshock: FloatND, work: DiscreteAction) -> FloatND:
        return 1.0 + bshock + 0.0 * jnp.asarray(work, dtype=float)

    src = Regime(
        transition={
            "folded_B": MarkovTransition(_route_to_B),
            "dead_C": MarkovTransition(_route_to_C),
        },
        active=lambda age: age < 1,
        states={"wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=3)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_src},
    )
    folded_B = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"bshock": _shock(fold=True)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _u_folded_B},
    )
    dead_C = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return {"src": src, "folded_B": folded_B, "dead_C": dead_C}


def _simulate_route(
    regimes: dict[str, Regime], *, discount: float
) -> tuple[MappingProxyType, object]:
    """Solve then simulate the route-to-folded-target model; return the sim result."""
    from _lcm.simulation.simulate import simulate  # noqa: PLC0415

    regime_names_to_ids = MappingProxyType(
        {"src": jnp.int32(0), "folded_B": jnp.int32(1), "dead_C": jnp.int32(2)}
    )
    processed = process_regimes(
        user_regimes=finalize_regimes(user_regimes=regimes, derived_categoricals={}),
        ages=_AGES,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=False,
    )
    flat_params = MappingProxyType(
        {
            "src": MappingProxyType({"H__discount_factor": jnp.asarray(discount)}),
            "folded_B": MappingProxyType({}),
            "dead_C": MappingProxyType({}),
        }
    )
    solution, _sim_policies, dissolution_flags = solve(
        flat_params=flat_params,
        ages=_AGES,
        regimes=processed,
        logger=get_logger(log_level="off"),
        enable_jit=False,
    )
    initial_conditions = MappingProxyType(
        {
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([0], dtype=jnp.int32),
            "wealth": jnp.array([0.0]),
        }
    )
    result = simulate(
        flat_params=flat_params,
        initial_conditions=initial_conditions,
        regimes=processed,
        regime_names_to_ids=regime_names_to_ids,
        logger=get_logger(log_level="off"),
        period_to_regime_to_V_arr=solution,
        period_to_regime_to_dissolution_flags=dissolution_flags,
        ages=_AGES,
        simulation_output_dtypes={},
        seed=0,
    )
    return solution, result


def test_folded_only_per_target_continuation_enters_simulated_value():
    """The folded-only continuation enters the SIMULATED argmax (round6/7 parity).

    Simulate re-optimizes Q over the grid, so `src`'s simulated period-0 decision
    must value the folded-only target `folded_B` (scalar V = 1.0) exactly as solve
    does: route to B for the discounted continuation `discount * 1.0 = 0.9`, which
    beats `dead_C`'s immediate 0.5.

    Pre-fix (simulate side of fold-round6 unpatched), the simulate Q read passed the
    UNSTRIPPED interpolation info for `folded_B` (whose stored V is the scalar 1.0 but
    whose `VInterpolationInfo` still lists the folded `bshock` axis), so it demanded a
    `next_bshock` coordinate the source never realises / indexed an axis the scalar V
    lacks — the simulated decision was wrong or the run crashed. Post-fix, the folded
    axis is stripped for the simulate continuation read (parity with solve), so `src`
    simulates `work == 1` (route to B) with recomputed V = discount.
    """
    discount = 0.9
    solution, result = _simulate_route(
        _make_route_to_folded_target_regimes_stateful(), discount=discount
    )
    # Sanity: the solve side already values B correctly at every `wealth` node
    # (V_src == discount; `wealth` is inert).
    np.testing.assert_allclose(np.asarray(solution[0]["src"]), discount, atol=1e-5)
    # The simulated period-0 decision must reflect the folded-only continuation:
    # route to B (work code 1), recomputed value = discount * 1.0.
    period_0 = result.raw_results["src"][0]
    np.testing.assert_array_equal(np.asarray(period_0.actions["work"]), [1])
    np.testing.assert_allclose(
        np.asarray(period_0.V_arr).reshape(-1), [discount], atol=1e-5
    )
