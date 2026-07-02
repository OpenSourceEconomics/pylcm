# Certainty-equivalent seam (issue #385) — design

**Date:** 2026-07-02 **Issue:**
[#385](https://github.com/OpenSourceEconomics/pylcm/issues/385) — Expose/generalize the
aggregator so the certainty-equivalent sees the value distribution (non-EU preferences)
**Branch:** `feat/certainty-equivalent`, based on `feat/type-local-continuation-v`
(#391)

## Goal

Make nonlinear certainty equivalents (Epstein–Zin and other recursive
non-expected-utility preferences) expressible without monkey-patching the engine. Today
the engine forms the scalar `E_next_V` as a linear probability-weighted average of
next-period values and hands only that scalar to the Bellman aggregator `H`; a nonlinear
CE such as `CE = (E[V'^(1-γ)])^(1/(1-γ))` needs the transform to hit `V'` *before* the
expectation.

Acceptance (from the issue):

- A documented way to specify a nonlinear CE.
- An Epstein–Zin example model with a test pinning its solved values.
- Default (expected-utility) behaviour and fixtures unchanged.

## Decisions

- **Seam shape:** transform pair `g` / `g⁻¹` now, shaped for extension. The engine
  dispatches on a `CertaintyEquivalent` base class; the shipped implementation is the
  transformed expectation `CE = g⁻¹(Σ_r p_r · E_w[g(V'_r)])`. A future full-distribution
  CE becomes a new subclass plus a new dispatch case in `Q_and_F.py` — no re-threading.
- **User API:** a `Regime` field, mirroring `taste_shocks`
  (`Regime(certainty_equivalent=...)`), not reserved function names in `functions`.
- **Built-ins:** `PowerCertaintyEquivalent` (Epstein–Zin) only. The generic
  `TransformedExpectation` covers other forms (e.g. risk-sensitive exponential).
- **Example:** toy version of the Atal–Fang–Karlsson–Ziebarth (2025) consumer block
  (savings + health Markov + survival) with the recursion swapped to Epstein–Zin;
  shipped as a test model with pinned values, promoted to `src/lcm_examples` with a docs
  page.

## User API

New module `src/lcm/certainty_equivalent.py`, mirroring `src/lcm/taste_shocks.py`
(frozen dataclass, `@beartype(conf=REGIME_CONF)`):

```python
class CertaintyEquivalent(ABC):
    """Base class; the engine dispatches on the concrete subclass."""


@dataclass(frozen=True, kw_only=True)
class TransformedExpectation(CertaintyEquivalent):
    transform: Callable[..., FloatND]
    """`g` — applied elementwise to next-period values before every expectation."""
    inverse: Callable[..., FloatND]
    """`g⁻¹` — applied once, after the regime-probability-weighted sum."""


class PowerCertaintyEquivalent(TransformedExpectation):
    """`CE = (E[V'^(1-risk_aversion)])^(1/(1-risk_aversion))`; runtime param `risk_aversion`."""
```

Rules:

- `Regime.certainty_equivalent: CertaintyEquivalent | None = None`.
- Phase-invariant — `Phased` is rejected with an explanation (same as
  `constraints`/`actions`).
- Rejected on terminal regimes (no continuation to aggregate).
- Regime-level only; no model-level broadcast (same as `taste_shocks`).
- `transform` and `inverse` take the value array as a reserved first argument named
  `value`. Every further signature argument is a runtime parameter, discovered into the
  params template under the pseudo-function name `"certainty_equivalent"`:
  `params[regime]["certainty_equivalent"]["risk_aversion"]`. A user function named
  `certainty_equivalent` collides and is rejected (the `taste_shocks` collision-check
  pattern in `regime_template.py`).
- Parameters come from the params template only — not from DAG function outputs (unlike
  `H`). Documented as a v1 restriction.

## Engine seam

In `get_Q_and_F` and `get_compute_intermediates`
(`src/_lcm/regime_building/Q_and_F.py`), when the regime declares a CE:

```
acc      = Σ_targets p_r · average(g(V'_r at stochastic states), joint_weights)
E_next_V = g⁻¹(acc)
Q        = H(utility, E_next_V, ...)        # unchanged
```

- `H` keeps its `E_next_V` kwarg name; under a CE it receives the certainty equivalent
  (documented). Epstein–Zin composes with the existing `H` seam:
  `H = ((1-β)·utility^ρ + β·E_next_V^ρ)^(1/ρ)` with `utility = c`.
- `certainty_equivalent is None` ⇒ byte-identical current code path: no identity
  transform enters the jaxpr, so the default path keeps its performance and its
  fixtures.
- CE parameter kwargs are pulled from `states_actions_params` by a small
  `_build_ce_kwargs`-style closure (simplified analogue of `_get_build_H_kwargs` —
  passthrough only, no DAG targets).
- Orthogonality:
  - Taste shocks smooth the max over *actions* (`max_Q_over_a` / `logsum.py`) — no code
    interaction with the state expectation.
  - #391's co-map slices V-array leaves per device; the CE transform is elementwise
    before averaging — no interaction.
  - Simulation needs no separate work: solve and simulate decision functions are built
    from the same `get_Q_and_F`.

### Threading (mirrors `has_taste_shocks`)

- `processing.py`: pass `certainty_equivalent=user_regime.certainty_equivalent` into the
  Q_and_F builders (solve + simulate function sets, and the diagnostics builder).
- `contract.py`: one field on `SolverBuildContext` recording the CE (for solver
  validation).
- `engine.py` canonical `Regime`: one field, mirroring `has_taste_shocks`.

## Solver compatibility

Euler-inversion EGM assumes expected utility; a nonlinear CE breaks it (EZ-EGM à la
Lujan is an explicit non-goal). Guard: model-build validation in `finalize.py` — a
regime with `solver=DCEGM(...)` and a non-`None` `certainty_equivalent` raises, naming
`GridSearch` as the supported solver. Rationale for the location: `finalize.py` is
barely touched by `feat/dcegm`, whereas `solvers.py` is rewritten there (+1838 lines),
and main's `DCEGM.validate` is an unconditional `NotImplementedError` stub.

## Conflict strategy (feat/dcegm, #391)

- Base branch: `feat/type-local-continuation-v` (#391); the PR retargets `main` once
  #391 lands (usual cascade). `feat/dcegm` is a sibling on top of #391.
- All new logic in new files (`src/lcm/certainty_equivalent.py`, plus an engine-side
  helper module if needed). Shared files get only small local hunks: the `E_next_V` loop
  in `Q_and_F.py`, one kwarg thread in `processing.py`, one field in `contract.py`, one
  field each on `lcm.regime.Regime` and `_lcm.engine.Regime`, the finalize guard, and
  the params template hook.
- Whichever of this branch and `feat/dcegm` merges second resolves the (small) conflicts
  via the existing pr-cascade workflow.

## Example and tests

### Test model — toy Atal, Epstein–Zin

`tests/test_models/stochastic/epstein_zin_health.py`:

- States: `wealth` (`LinSpacedGrid`, ~10 points), `health` (`DiscreteGrid`, good/bad,
  Markov via `MarkovTransition`).
- Regimes: `alive` (consumption/savings action; health-dependent survival probability
  into `dead` via a per-target stochastic regime transition) and `dead` (terminal).
- Deterministic income; ~5 periods.
- Epstein–Zin: `utility = c`; `H = ((1-β)·utility^ρ + β·E_next_V^ρ)^(1/ρ)`;
  `certainty_equivalent=PowerCertaintyEquivalent()`.
- The `dead` terminal utility is a small **positive** bequest value: the power transform
  requires `V' ≥ 0`, and with `risk_aversion > 1` a zero death-continuation gives
  `0^(1-γ) = ∞` (the classic EZ-with-mortality trap). Documented in the docs page.

### Tests (TDD, red first)

1. **API/validation:** field accepted; params template carries the
   `certainty_equivalent` pseudo-function params; terminal-regime rejection; `Phased`
   rejection; DCEGM+CE rejection at model build; user-function-name collision rejection.
1. **Reduction:** `PowerCertaintyEquivalent` with `risk_aversion = 0` (identity
   transform) reproduces the default-path solve on the same model to tight tolerance.
1. **Pinned solved values:** the EZ toy's solved V equals an independent NumPy backward
   induction written in the test (`assert_allclose`, explicit `atol`).
1. **Simulation:** simulated consumption/wealth paths equal the NumPy reference's
   forward pass.
1. **Regression:** the existing suite stays green unchanged (default fixtures
   untouched).

### Example promotion + docs

`src/lcm_examples/epstein_zin.py` plus a docs page, following the Iskhakov-2017
promotion pattern (#374): the EZ recursion and how `transform`/`inverse`/`H` map onto
it, positivity and mortality pitfalls, the template parameter location, and the
grid-search-only restriction.

## Non-goals

- Full-distribution CE callable (quantile CEs, ambiguity aversion) — enabled by the
  subclass seam, not shipped.
- EZ-EGM (DC-EGM under Epstein–Zin, Lujan) — DCEGM+CE is rejected instead.
- CE parameters as DAG function outputs.
- Model-level broadcast of `certainty_equivalent`.
- `ExponentialCertaintyEquivalent` (risk-sensitive) built-in — expressible via
  `TransformedExpectation`.
