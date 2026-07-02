# Certainty-Equivalent Seam (#385) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `Regime` declare a nonlinear certainty equivalent
`CE = g⁻¹(Σ_r p_r · E_w[g(V')])` over the next-period value distribution, with
Epstein–Zin as the shipped built-in, while the default (no CE) path stays
byte-identical.

**Architecture:** A `CertaintyEquivalent` ABC with one shipped subclass
`TransformedExpectation` (transform pair `g`/`g⁻¹`) lives in a new user-facing module; a
`Regime.certainty_equivalent` field threads it — mirroring `taste_shocks` — through
finalize validation, the params template (pseudo-function `"certainty_equivalent"`), and
into `get_Q_and_F`/`get_compute_intermediates`, where the transform wraps the two
expectation sites. DC-EGM rejects it at model build.

**Tech Stack:** JAX, dags, beartype+jaxtyping, pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-07-02-certainty-equivalent-seam-design.md`

## Global Constraints

- Branch: `feat/certainty-equivalent`, based on `feat/type-local-continuation-v` (PR
  #391). Do NOT rebase onto main.
- Conflict budget: `feat/dcegm` rewrites `Q_and_F.py`, `processing.py`, `contract.py`,
  `solvers.py`. New logic goes in NEW files; shared files get only small local hunks.
  Never touch `max_Q_over_a.py` or `solvers.py`.
- TDD: every task writes the failing test first and runs it before implementing.
- Tests: `pixi run -e tests-cpu pytest <path> -v` for single files;
  `pixi run -e tests-cpu tests -n 7` for the suite. Type check:
  `pixi run -e tests-cpu ty`. Hooks: `prek run --all-files` (NOT `pre-commit`).
- Python 3.14; no `from __future__ import annotations`; narrowest jaxtyping alias
  (`FloatND`, `Float1D`, `ScalarInt`); `# ty: ignore[rule]` never `# type: ignore`.
- Docstrings: Google style, MyST markup, imperative summary lines, PEP 257 inline field
  docstrings, no history/PR references.
- Plain pytest functions (no test classes); concrete-value assertions with explicit
  tolerances.
- `log_level` is required on every `solve()`/`simulate()` call; use `"debug"` in tests.
- Commit messages: imperative, no `feat:` prefixes (repo style), each ending with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- No dependency changes anywhere in this plan ⇒ never run `pixi lock`.

______________________________________________________________________

### Task 1: User API module `lcm.certainty_equivalent`

**Files:**

- Create: `src/lcm/certainty_equivalent.py`
- Modify: `src/lcm/__init__.py` (imports near line 93, `__all__` near line 115)
- Test: `tests/test_certainty_equivalent.py` (create)

**Interfaces:**

- Consumes: `_lcm.utils.functools.get_union_of_args`, `_lcm.beartype_conf.REGIME_CONF`,
  `lcm.exceptions.RegimeInitializationError`, `lcm.typing.FloatND`.

- Produces (later tasks rely on these exact names):

  - `CertaintyEquivalent` (ABC) with abstract property `param_names -> frozenset[str]`
  - `TransformedExpectation(CertaintyEquivalent)` with fields
    `transform: Callable[..., FloatND]`, `inverse: Callable[..., FloatND]`, property
    `param_names`
  - `PowerCertaintyEquivalent(TransformedExpectation)` — no-argument construction,
    runtime param `risk_aversion`
  - Module constant `CE_VALUE_ARG = "value"`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_certainty_equivalent.py`:

```python
"""Tests for nonlinear certainty equivalents over the continuation value."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import PowerCertaintyEquivalent, TransformedExpectation
from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND


def test_power_certainty_equivalent_transform_and_inverse_are_inverses():
    """`inverse(transform(x)) == x` for positive values."""
    ce = PowerCertaintyEquivalent()
    x = jnp.array([0.5, 1.0, 2.0, 7.5])
    roundtrip = ce.inverse(
        value=ce.transform(value=x, risk_aversion=jnp.asarray(0.5)),
        risk_aversion=jnp.asarray(0.5),
    )
    np.testing.assert_allclose(roundtrip, x, rtol=1e-6)


def test_power_certainty_equivalent_param_names():
    """The power CE declares exactly the `risk_aversion` runtime param."""
    assert PowerCertaintyEquivalent().param_names == frozenset({"risk_aversion"})


def test_transformed_expectation_param_names_union_over_both_callables():
    """`param_names` is the union of transform and inverse args minus `value`."""

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    def g_inv(value: FloatND, theta: FloatND, offset: FloatND) -> FloatND:
        return value / theta + offset

    ce = TransformedExpectation(transform=g, inverse=g_inv)
    assert ce.param_names == frozenset({"theta", "offset"})


def test_transformed_expectation_rejects_callable_without_value_arg():
    """Both callables must take the value array via an argument named `value`."""

    def g(v: FloatND) -> FloatND:
        return v

    def g_inv(value: FloatND) -> FloatND:
        return value

    with pytest.raises(RegimeInitializationError, match="value"):
        TransformedExpectation(transform=g, inverse=g_inv)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected: FAIL
at collection with
`ImportError: cannot import name 'PowerCertaintyEquivalent' from 'lcm'`.

- [ ] **Step 3: Implement the module**

Create `src/lcm/certainty_equivalent.py`:

```python
"""Nonlinear certainty equivalents over the next-period value distribution."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.utils.functools import get_union_of_args
from lcm.exceptions import RegimeInitializationError
from lcm.typing import FloatND

CE_VALUE_ARG = "value"
"""Reserved argument name through which transform callables receive values."""


class CertaintyEquivalent(ABC):
    """Base class for certainty-equivalent specifications.

    Declared on a non-terminal `Regime` via `certainty_equivalent=...`. The
    engine dispatches on the concrete subclass; `TransformedExpectation` is
    the shipped implementation. When the field is `None` (the default), the
    continuation is aggregated as the linear expectation `E[V']`. Only
    `GridSearch` supports a nonlinear certainty equivalent.
    """

    @property
    @abstractmethod
    def param_names(self) -> frozenset[str]:
        """Names of the certainty equivalent's runtime parameters."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class TransformedExpectation(CertaintyEquivalent):
    """Certainty equivalent `CE = g⁻¹(Σ_r p_r · E_w[g(V'_r)])`.

    `transform` (`g`) is applied elementwise to next-period values before
    every expectation — over stochastic state transitions and over regime
    transitions — and `inverse` (`g⁻¹`) once, after the regime-probability-
    weighted sum. Both callables take the value array as the reserved first
    argument `value`; every further signature argument becomes a runtime
    parameter under the pseudo-function name `certainty_equivalent` in the
    regime's params (`{"certainty_equivalent": {"<arg>": ...}}`).

    Combined with a user-supplied Bellman aggregator `H` this expresses
    Epstein–Zin and other transformed-expectation recursive preferences.
    The parameters are read from the params template only, not from DAG
    function outputs.
    """

    transform: Callable[..., FloatND]
    """`g` — applied elementwise to next-period values before every expectation."""

    inverse: Callable[..., FloatND]
    """`g⁻¹` — applied once, after the regime-probability-weighted sum."""

    def __post_init__(self) -> None:
        for name in ("transform", "inverse"):
            func = getattr(self, name)
            if CE_VALUE_ARG not in get_union_of_args([func]):
                msg = (
                    f"The `{name}` callable of a `TransformedExpectation` must "
                    f"take the value array via an argument named "
                    f"'{CE_VALUE_ARG}'."
                )
                raise RegimeInitializationError(msg)

    @property
    def param_names(self) -> frozenset[str]:
        """Names of the runtime parameters of `transform` and `inverse`."""
        return frozenset(
            get_union_of_args([self.transform, self.inverse]) - {CE_VALUE_ARG}
        )


def _power_transform(value: FloatND, risk_aversion: FloatND) -> FloatND:
    return value ** (1.0 - risk_aversion)


def _power_inverse(value: FloatND, risk_aversion: FloatND) -> FloatND:
    return value ** (1.0 / (1.0 - risk_aversion))


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class PowerCertaintyEquivalent(TransformedExpectation):
    """Epstein–Zin power certainty equivalent.

    `CE = (E[V'^(1 - risk_aversion)])^(1 / (1 - risk_aversion))` with the
    runtime parameter `{"certainty_equivalent": {"risk_aversion": ...}}`.
    Requires strictly positive continuation values; `risk_aversion = 1`
    (the log case) is not representable. `risk_aversion = 0` reduces to
    the linear expectation.
    """

    transform: Callable[..., FloatND] = _power_transform
    inverse: Callable[..., FloatND] = _power_inverse
```

Modify `src/lcm/__init__.py`: next to the `lcm.taste_shocks` import (line ~93) add

```python
from lcm.certainty_equivalent import (  # noqa: E402
    CertaintyEquivalent,
    PowerCertaintyEquivalent,
    TransformedExpectation,
)
```

and add `"CertaintyEquivalent"`, `"PowerCertaintyEquivalent"`,
`"TransformedExpectation"` to `__all__` (alphabetical position).

Note: if module layout ordering (helpers `_power_transform`/`_power_inverse` above
`PowerCertaintyEquivalent`) trips a linter, keep it — dataclass defaults must be bound
before the class body evaluates.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected: 4
PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/lcm/certainty_equivalent.py src/lcm/__init__.py tests/test_certainty_equivalent.py
git commit -m "Add the certainty-equivalent user API (transform-pair seam)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 2: `Regime.certainty_equivalent` field + model-build validation

**Files:**

- Modify: `src/lcm/regime.py` (import near line 34; field after `taste_shocks`, line
  ~166)
- Modify: `src/_lcm/user_regime_validation.py` (`_validate_completeness`, line ~211; new
  helper below it)
- Test: `tests/test_certainty_equivalent.py` (append)

**Interfaces:**

- Consumes: Task 1's `CertaintyEquivalent`, `PowerCertaintyEquivalent`;
  `lcm.solvers.DCEGM`.

- Produces: `Regime.certainty_equivalent: CertaintyEquivalent | None = None`;
  finalize-time `RegimeInitializationError` for (a) terminal regime with CE, (b) `DCEGM`
  solver with CE.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_certainty_equivalent.py` (imports go to the top of the file):

```python
import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.solvers import DCEGM
from lcm.typing import BoolND, ContinuousAction, ContinuousState, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility_alive(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_dead(wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def _next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _budget(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_regime() -> ScalarInt:
    return _RegimeId.dead


_WEALTH = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=0.5, stop=5.0, n_points=5)


def _make_model(*, alive_kwargs: dict, dead_kwargs: dict) -> Model:
    """Build a minimal two-regime model with extra kwargs spliced per regime."""
    alive = Regime(
        transition=_next_regime,
        states={"wealth": _WEALTH},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": _CONSUMPTION},
        constraints={"budget": _budget},
        functions={"utility": _utility_alive},
        active=lambda age: age < 41,
        **alive_kwargs,
    )
    dead = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5)},
        functions={"utility": _utility_dead},
        **dead_kwargs,
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_regime_accepts_certainty_equivalent():
    """A non-terminal grid-search regime may declare a certainty equivalent."""
    model = _make_model(
        alive_kwargs={"certainty_equivalent": PowerCertaintyEquivalent()},
        dead_kwargs={},
    )
    assert model.user_regimes["alive"].certainty_equivalent is not None


def test_terminal_regime_rejects_certainty_equivalent():
    """Terminal regimes have no continuation to aggregate."""
    with pytest.raises(RegimeInitializationError, match="[Tt]erminal"):
        _make_model(
            alive_kwargs={},
            dead_kwargs={"certainty_equivalent": PowerCertaintyEquivalent()},
        )


def test_dcegm_rejects_certainty_equivalent():
    """DC-EGM's Euler inversion assumes expected utility; the guard names GridSearch."""
    dcegm = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    with pytest.raises(RegimeInitializationError, match="GridSearch"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerCertaintyEquivalent(),
                "solver": dcegm,
            },
            dead_kwargs={},
        )


def test_certainty_equivalent_rejects_phased():
    """The certainty equivalent is phase-invariant; `Phased` is rejected."""
    with pytest.raises(RegimeInitializationError):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": Phased(
                    solve=PowerCertaintyEquivalent(),
                    simulate=PowerCertaintyEquivalent(),
                ),
            },
            dead_kwargs={},
        )
```

Note: `Phased` — confirm the import location (`from lcm import Phased`; if not exported
there, `from lcm.phased import Phased` as `src/lcm/regime.py:32` does).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected: the
4 new tests FAIL with
`TypeError: Regime.__init__() got an unexpected keyword argument 'certainty_equivalent'`
(or the beartype equivalent). Task 1's tests still PASS.

- [ ] **Step 3: Implement field and validation**

In `src/lcm/regime.py`, next to the `lcm.taste_shocks` import add:

```python
from lcm.certainty_equivalent import CertaintyEquivalent
```

After the `taste_shocks` field (line ~166) add:

```python
    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent over the next-period value distribution.

    When set, the solve aggregates the continuation as
    `g⁻¹(Σ_r p_r · E_w[g(V')])` instead of the linear expectation, and the
    transform parameters become runtime params under the pseudo-function
    name `certainty_equivalent`. Only non-terminal regimes solved by
    `GridSearch` support it.
    """
```

In `src/_lcm/user_regime_validation.py`, add `from lcm.solvers import DCEGM` to the
imports, extend `_validate_completeness` (after the existing
`error_messages.extend(...)` calls, line ~234):

```python
    error_messages.extend(_certainty_equivalent_errors(regime))
```

and add below the existing private helpers:

```python
def _certainty_equivalent_errors(regime: lcm.regime.Regime) -> list[str]:
    """Collect errors for a regime's `certainty_equivalent` declaration.

    - terminal regimes have no continuation value to aggregate
    - only `GridSearch` supports a nonlinear certainty equivalent (the
      Euler inversion in DC-EGM assumes expected utility)
    """
    if regime.certainty_equivalent is None:
        return []
    error_messages: list[str] = []
    if regime.terminal:
        error_messages.append(
            "A terminal regime cannot declare `certainty_equivalent`: there "
            "is no continuation value to aggregate."
        )
    if isinstance(regime.solver, DCEGM):
        error_messages.append(
            "The DCEGM solver does not support a nonlinear "
            "`certainty_equivalent`: the Euler inversion assumes expected "
            "utility. Use GridSearch() for this regime."
        )
    return error_messages
```

The `Phased` rejection needs no code: the field's type is `CertaintyEquivalent | None`,
so the `REGIME_CONF` beartype boundary raises `RegimeInitializationError` on a `Phased`
value. If `_validate_mapping_contents`/phase normalization fires first with a different
error type, keep beartype's error and adjust nothing — the test only asserts
`RegimeInitializationError`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected: 8
PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/lcm/regime.py src/_lcm/user_regime_validation.py tests/test_certainty_equivalent.py
git commit -m "Add Regime.certainty_equivalent with terminal- and DCEGM-rejection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 3: Params-template pseudo-function `"certainty_equivalent"`

**Files:**

- Modify: `src/_lcm/params/regime_template.py` (directly after the `taste_shocks` block,
  line ~102)
- Test: `tests/test_certainty_equivalent.py` (append)

**Interfaces:**

- Consumes: `Regime.certainty_equivalent` (Task 2), `CertaintyEquivalent.param_names`
  (Task 1).

- Produces:
  `model.get_params_template()[<regime>]["certainty_equivalent"] == {"<param>": "float", ...}`;
  flat param names `certainty_equivalent__<param>` reach `flat_param_names` (used in
  Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_certainty_equivalent.py`:

```python
from lcm.exceptions import InvalidNameError


def test_params_template_contains_certainty_equivalent_params():
    """CE params surface under the pseudo-function name `certainty_equivalent`."""
    model = _make_model(
        alive_kwargs={"certainty_equivalent": PowerCertaintyEquivalent()},
        dead_kwargs={},
    )
    template = model.get_params_template()
    assert template["alive"]["certainty_equivalent"] == {"risk_aversion": "float"}


def test_certainty_equivalent_name_collision_with_function_is_rejected():
    """A regime function named `certainty_equivalent` collides with the pseudo-function."""

    def certainty_equivalent(wealth: ContinuousState) -> FloatND:
        return wealth

    with pytest.raises(InvalidNameError, match="certainty_equivalent"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerCertaintyEquivalent(),
                "functions": {
                    "utility": _utility_alive,
                    "certainty_equivalent": certainty_equivalent,
                },
            },
            dead_kwargs={},
        )
```

Note on the collision test: `alive_kwargs` containing `functions` conflicts with
`_make_model`'s own `functions` kwarg — change `_make_model` so `alive_kwargs` is
applied via `dict(base_kwargs) | alive_kwargs` before constructing the `Regime`:

```python
def _make_model(*, alive_kwargs: dict, dead_kwargs: dict) -> Model:
    base_alive = {
        "transition": _next_regime,
        "states": {"wealth": _WEALTH},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": _CONSUMPTION},
        "constraints": {"budget": _budget},
        "functions": {"utility": _utility_alive},
        "active": lambda age: age < 41,
    }
    base_dead = {
        "transition": None,
        "states": {"wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility_dead},
    }
    alive = Regime(**(base_alive | alive_kwargs))
    dead = Regime(**(base_dead | dead_kwargs))
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=_RegimeId,
    )
```

(Apply this refactor of `_make_model` while writing this step; Task 2's tests must stay
green.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected:
`test_params_template_contains_certainty_equivalent_params` FAILS with
`KeyError: 'certainty_equivalent'`; the collision test FAILS (no error raised). Earlier
tests PASS.

- [ ] **Step 3: Implement the template hook**

In `src/_lcm/params/regime_template.py`, directly after the `taste_shocks` block (line
~102):

```python
    if user_regime.certainty_equivalent is not None:
        if "certainty_equivalent" in function_params:
            raise InvalidNameError(
                "The regime declares `certainty_equivalent`, whose parameters "
                "live under the pseudo-function name 'certainty_equivalent' in "
                "the params — this conflicts with a regime function of the "
                "same name."
            )
        function_params["certainty_equivalent"] = dict.fromkeys(
            sorted(user_regime.certainty_equivalent.param_names), "float"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected: 10
PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/_lcm/params/regime_template.py tests/test_certainty_equivalent.py
git commit -m "Surface certainty-equivalent params in the regime params template

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 4: Engine seam in `Q_and_F` + threading + toy EZ test model

**Files:**

- Create: `tests/test_models/epstein_zin_health.py`
- Modify: `src/_lcm/regime_building/Q_and_F.py` (`get_Q_and_F`,
  `get_compute_intermediates`, new private helper)
- Modify: `src/_lcm/regime_building/processing.py` (thread `certainty_equivalent`
  alongside `has_taste_shocks`)
- Modify: `src/_lcm/solution/contract.py` (`SolverBuildContext`, after
  `has_taste_shocks`, line ~53)
- Modify: `src/_lcm/engine.py` (canonical `Regime`, after `has_taste_shocks`, line ~598)
- Test: `tests/test_certainty_equivalent.py` (append)

**Interfaces:**

- Consumes: Tasks 1–3. `flat_param_names` already contains
  `certainty_equivalent__<param>` (Task 3), so the values arrive in
  `states_actions_params` at runtime — the same channel `taste_shocks__scale` uses.

- Produces:

  - `get_Q_and_F(..., certainty_equivalent: CertaintyEquivalent | None = None)` and same
    on `get_compute_intermediates`
  - `SolverBuildContext.certainty_equivalent: CertaintyEquivalent | None = None`
  - engine `Regime.certainty_equivalent: CertaintyEquivalent | None = None`
  - Test model exporting `get_model(*, certainty_equivalent=None)`,
    `get_params(*, risk_aversion, ...)`, grids/constants used by Task 5's NumPy
    reference.

- [ ] **Step 1: Write the toy Epstein–Zin test model**

Create `tests/test_models/epstein_zin_health.py`. Design notes baked into the numbers:
`income == consumption-grid start (0.5)` keeps next wealth inside `[0.5, 12]` so linear
interpolation never extrapolates (numpy references clamp, pylcm extrapolates); all value
functions are strictly positive so power transforms are safe; `survival_probs` ends in
`0.0` like `lcm_examples/mortality.py`.

```python
"""Toy Epstein–Zin savings model with health-dependent mortality.

A pared-down version of the consumer block of Atal, Fang, Karlsson &
Ziebarth (2025, JPE 133(6), doi:10.1086/734781) — savings, a two-state
health Markov chain, and health-dependent survival into a terminal `dead`
regime — with the recursion swapped to Epstein–Zin:

- `V = ((1 - β) · c^ρ + β · CE^ρ)^(1/ρ)` via a user-supplied `H`,
- `CE = (E[V'^(1-γ)])^(1/(1-γ))` via `PowerCertaintyEquivalent`.

Utility is consumption itself, so values stay in (positive) consumption
units and the power transform is well-defined. The `dead` bequest value
`sqrt(wealth)` is strictly positive at every reachable wealth. Grids are
sized so an in-test numpy backward induction on the same grids reproduces
the solve exactly: next wealth `w - c + income` stays inside the wealth
grids (`income` equals the consumption-grid lower bound), so linear
interpolation never extrapolates.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.certainty_equivalent import CertaintyEquivalent
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    Float1D,
    FloatND,
    Period,
    ScalarInt,
)

WEALTH_GRID = LinSpacedGrid(start=0.5, stop=12.0, n_points=6)
DEAD_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=12.0, n_points=25)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=5.0, n_points=7)

INCOME = 0.5
SURVIVAL_PROBS = (0.95, 0.85, 0.0)
BAD_HEALTH_SURVIVAL_FACTOR = 0.9
HEALTH_TRANSITION = ((0.8, 0.2), (0.1, 0.9))  # rows: bad, good


@categorical(ordered=False)
class EzRegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class HealthStatus:
    bad: ScalarInt
    good: ScalarInt


def utility_alive(consumption: ContinuousAction) -> FloatND:
    return consumption


def utility_dead(wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def H_epstein_zin(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND, rho: FloatND
) -> FloatND:
    return (
        (1.0 - discount_factor) * utility**rho + discount_factor * E_next_V**rho
    ) ** (1.0 / rho)


def next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction, income: FloatND
) -> ContinuousState:
    return wealth - consumption + income


def health_probs(health: DiscreteState) -> FloatND:
    return jnp.where(
        health == HealthStatus.good,
        jnp.array(HEALTH_TRANSITION[1]),
        jnp.array(HEALTH_TRANSITION[0]),
    )


def next_regime(
    health: DiscreteState, period: Period, survival_probs: Float1D
) -> FloatND:
    sp = survival_probs[period] * jnp.where(
        health == HealthStatus.good, 1.0, BAD_HEALTH_SURVIVAL_FACTOR
    )
    return jnp.array([sp, 1.0 - sp])


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def get_model(*, certainty_equivalent: CertaintyEquivalent | None = None) -> Model:
    alive = UserRegime(
        transition=MarkovTransition(next_regime),
        states={
            "wealth": WEALTH_GRID,
            "health": DiscreteGrid(HealthStatus),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": {"alive": MarkovTransition(health_probs)},
        },
        actions={"consumption": CONSUMPTION_GRID},
        constraints={"budget_constraint": budget_constraint},
        functions={"utility": utility_alive, "H": H_epstein_zin},
        certainty_equivalent=certainty_equivalent,
        active=lambda age: age < 63,
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": DEAD_WEALTH_GRID},
        functions={"utility": utility_dead},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=60, stop=63, step="Y"),
        regime_id_class=EzRegimeId,
    )


def get_params(
    *,
    risk_aversion: float | None,
    discount_factor: float = 0.9,
    rho: float = 0.5,
) -> dict:
    params: dict = {
        "alive": {
            "H": {"discount_factor": discount_factor, "rho": rho},
            "next_wealth": {"income": INCOME},
            "next_regime": {"survival_probs": jnp.array(SURVIVAL_PROBS)},
        },
    }
    if risk_aversion is not None:
        params["alive"]["certainty_equivalent"] = {"risk_aversion": risk_aversion}
    return params
```

If the params nesting is off (e.g. the regime transition or a per-target law keys
differently), print `get_model().get_params_template()` and align `get_params` with it —
the template is the source of truth. Sanity-check now:

Run:
`pixi run -e tests-cpu python -c " from tests.test_models.epstein_zin_health import get_model, get_params m = get_model() print(m.get_params_template()) m.solve(params=get_params(risk_aversion=None), log_level='debug') print('EU solve OK') "`
Expected: template printed; `EU solve OK`.

- [ ] **Step 2: Write the failing seam tests**

Append to `tests/test_certainty_equivalent.py`:

```python
from tests.test_models.epstein_zin_health import get_model, get_params


def test_nonlinear_certainty_equivalent_changes_solved_values():
    """With `risk_aversion = 2`, solved values differ from expected utility."""
    ez_model = get_model(certainty_equivalent=PowerCertaintyEquivalent())
    eu_model = get_model(certainty_equivalent=None)
    V_ez = ez_model.solve(params=get_params(risk_aversion=2.0), log_level="debug")
    V_eu = eu_model.solve(params=get_params(risk_aversion=None), log_level="debug")
    assert not np.allclose(
        np.asarray(V_ez[0]["alive"]), np.asarray(V_eu[0]["alive"]), rtol=1e-6
    )


def test_zero_risk_aversion_reduces_to_expected_utility():
    """`risk_aversion = 0` makes the power CE the linear expectation."""
    ez_model = get_model(certainty_equivalent=PowerCertaintyEquivalent())
    eu_model = get_model(certainty_equivalent=None)
    V_ez = ez_model.solve(params=get_params(risk_aversion=0.0), log_level="debug")
    V_eu = eu_model.solve(params=get_params(risk_aversion=None), log_level="debug")
    for period in V_eu:
        for regime_name in V_eu[period]:
            np.testing.assert_allclose(
                np.asarray(V_ez[period][regime_name]),
                np.asarray(V_eu[period][regime_name]),
                rtol=1e-5,
                err_msg=f"period={period}, regime={regime_name}",
            )
```

If `model.solve()` returns a container other than `dict[period][regime] -> array`, adapt
the iteration — check `tests/test_taste_shocks.py` for the access pattern actually in
use.

- [ ] **Step 3: Run tests to verify they fail for the right reason**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected:
`test_nonlinear_certainty_equivalent_changes_solved_values` FAILS on the
`assert not np.allclose(...)` (the CE is accepted but ignored, so values are equal —
exactly the missing behavior). Everything else PASSES, including the reduction test
(trivially, for now). If instead the EZ solve raises `InvalidParamsError` on the
`certainty_equivalent` params entry, that also counts as the right failure — the seam
doesn't consume them yet.

- [ ] **Step 4: Implement the seam in `Q_and_F.py`**

In `src/_lcm/regime_building/Q_and_F.py`:

Add imports:

```python
from lcm.certainty_equivalent import (
    CE_VALUE_ARG,
    CertaintyEquivalent,
    TransformedExpectation,
)
```

Add to the module's private helpers (bottom of file):

```python
def _resolve_certainty_equivalent(
    certainty_equivalent: CertaintyEquivalent | None,
) -> tuple[
    TransformedExpectation | None,
    MappingProxyType[str, str],
    MappingProxyType[str, str],
]:
    """Narrow the certainty equivalent and map its args to flat param names.

    The runtime parameters live under the pseudo-function name
    `certainty_equivalent` in the regime's flat params
    (`certainty_equivalent__<arg>`); the returned mappings let the Q-and-F
    closure pull each callable's kwargs from `states_actions_params`.

    Returns:
        Tuple of the narrowed transform-pair CE (or `None`), the transform's
        arg-to-flat-name mapping, and the inverse's arg-to-flat-name mapping.

    """
    if certainty_equivalent is None:
        return None, MappingProxyType({}), MappingProxyType({})
    if not isinstance(certainty_equivalent, TransformedExpectation):
        msg = (
            "Only `TransformedExpectation` certainty equivalents are "
            f"supported, got {type(certainty_equivalent).__name__}."
        )
        raise NotImplementedError(msg)

    def flat_names(func: Callable[..., FloatND]) -> MappingProxyType[str, str]:
        return MappingProxyType(
            {
                arg: f"certainty_equivalent__{arg}"
                for arg in get_union_of_args([func]) - {CE_VALUE_ARG}
            }
        )

    return (
        certainty_equivalent,
        flat_names(certainty_equivalent.transform),
        flat_names(certainty_equivalent.inverse),
    )
```

In `get_Q_and_F`, add the keyword-only parameter (with docstring entry):

```python
    certainty_equivalent: CertaintyEquivalent | None = None,
```

```
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
```

After `_build_H_kwargs = _get_build_H_kwargs(functions)` add:

```python
    ce, ce_transform_flat_names, ce_inverse_flat_names = (
        _resolve_certainty_equivalent(certainty_equivalent)
    )
```

Inside the `Q_and_F` closure, replace the block from
`next_V_at_stochastic_states_arr = ...` through the `E_next_V = E_next_V + ...`
accumulation and up to the `Q_arr = functions["H"](...)` call with:

```python
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **{
                    name: val
                    for name, val in next_states.items()
                    if name not in _co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )
            if ce is not None:
                next_V_at_stochastic_states_arr = ce.transform(
                    value=next_V_at_stochastic_states_arr,
                    **{
                        arg: states_actions_params[flat_name]
                        for arg, flat_name in ce_transform_flat_names.items()
                    },
                )

            # We then take the weighted average of the next value function at the
            # stochastic states to get the expected next value function.
            next_V_expected_arr = jnp.average(
                next_V_at_stochastic_states_arr,
                weights=joint_next_stochastic_states_weights,
            )
            E_next_V = (
                E_next_V + active_regime_probs[target_regime_name] * next_V_expected_arr
            )

        if ce is not None:
            E_next_V = ce.inverse(
                value=E_next_V,
                **{
                    arg: states_actions_params[flat_name]
                    for arg, flat_name in ce_inverse_flat_names.items()
                },
            )

        Q_arr = functions["H"](
            utility=U_arr,
            E_next_V=E_next_V,
            **_build_H_kwargs(states_actions_params),
        )
```

Make the *same* three edits (parameter, `_resolve_certainty_equivalent` call,
transform/inverse application) in `get_compute_intermediates` — its loop body is a
near-copy without the co-map filtering.

- [ ] **Step 5: Thread through `processing.py`, `contract.py`, `engine.py`**

All threading mirrors `has_taste_shocks`; run
`grep -n "has_taste_shocks" src/_lcm/regime_building/processing.py` to see every site.
Concretely:

1. `src/_lcm/solution/contract.py` — after the `has_taste_shocks` field (~line 53), with
   import `from lcm.certainty_equivalent import CertaintyEquivalent`:

```python
    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent declared by the regime, if any.

    `GridSearch` consumes it via the compiled Q-and-F closures; solvers
    that exploit the linear-expectation structure of the continuation
    (e.g. Euler-inversion EGM) must reject regimes that declare one.
    """
```

2. `src/_lcm/engine.py` — after `has_taste_shocks` (~line 598), same import:

```python
    certainty_equivalent: CertaintyEquivalent | None = None
    """Nonlinear certainty equivalent declared by the regime, if any."""
```

3. `src/_lcm/regime_building/processing.py`:
   - `process_regimes` loop: add
     `certainty_equivalent=user_regime.certainty_equivalent,` to the
     `_build_solution_phase(...)` call (next to `has_taste_shocks=...`, ~line 196), the
     `_build_simulation_phase(...)` call (~line 218), and the canonical `Regime(...)`
     construction (~line 235).
   - `_build_solution_phase` and `_build_simulation_phase`: add keyword-only parameter
     `certainty_equivalent: CertaintyEquivalent | None,` (plus docstring line "Nonlinear
     certainty equivalent declared by the regime, or `None`."), and forward it to every
     `_build_Q_and_F_per_period(...)` call, to the diagnostics builder call that reaches
     `get_compute_intermediates`, and to the `SolverBuildContext(...)` construction
     (near the `solver.validate(context=context)` call, ~line 388).
   - `_build_Q_and_F_per_period` (~line 1660): add keyword-only parameter
     `certainty_equivalent: CertaintyEquivalent | None = None,` (docstring entry as
     above) and forward `certainty_equivalent=certainty_equivalent,` into
     `get_Q_and_F(...)`.
   - The diagnostics builder in `src/_lcm/regime_building/diagnostics.py` (calls
     `get_compute_intermediates`, line ~107): add the same parameter and forward it;
     thread it from `_build_solution_phase`'s call site.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py -v` Expected: 12
PASSED — the divergence test now sees different values, the reduction test still sees
equal ones.

- [ ] **Step 7: Run the full suite to confirm the default path is untouched**

Run: `pixi run -e tests-cpu tests -n 7` Expected: all PASS (no fixture changes
anywhere).

- [ ] **Step 8: Commit**

```bash
git add src/_lcm/regime_building/Q_and_F.py src/_lcm/regime_building/processing.py \
        src/_lcm/regime_building/diagnostics.py src/_lcm/solution/contract.py \
        src/_lcm/engine.py tests/test_models/epstein_zin_health.py \
        tests/test_certainty_equivalent.py
git commit -m "Apply the certainty-equivalent transform around both continuation expectations

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 5: Pinned-value tests against an independent NumPy backward induction

**Files:**

- Modify: `tests/test_certainty_equivalent.py` (append reference solver + solve test)
- Create: `tests/simulation/test_simulate_certainty_equivalent.py`

**Interfaces:**

- Consumes: Task 4's toy model (`get_model`, `get_params`, `WEALTH_GRID`,
  `DEAD_WEALTH_GRID`, `CONSUMPTION_GRID`, `INCOME`, `SURVIVAL_PROBS`,
  `BAD_HEALTH_SURVIVAL_FACTOR`, `HEALTH_TRANSITION`, `HealthStatus`, `EzRegimeId`).

- Produces: nothing consumed later; this is the issue's "test pinning its solved
  values".

- [ ] **Step 1: Write the failing solve-pinning test**

Append to `tests/test_certainty_equivalent.py`:

```python
from tests.test_models.epstein_zin_health import (
    BAD_HEALTH_SURVIVAL_FACTOR,
    CONSUMPTION_GRID,
    DEAD_WEALTH_GRID,
    HEALTH_TRANSITION,
    INCOME,
    SURVIVAL_PROBS,
    WEALTH_GRID,
)


def _reference_backward_induction(
    *, risk_aversion: float, discount_factor: float, rho: float
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Independent numpy backward induction of the toy Epstein–Zin model.

    Mirrors the engine's computation order on the same grids: interpolate
    each target's V at next wealth, transform, average over health, weight
    by regime probabilities, invert, aggregate via the EZ `H`. Returns the
    per-period alive V arrays (shape `(n_wealth, n_health)`) and the
    period-0 argmax consumption (same shape).
    """
    wealth = np.linspace(WEALTH_GRID.start, WEALTH_GRID.stop, WEALTH_GRID.n_points)
    dead_wealth = np.linspace(
        DEAD_WEALTH_GRID.start, DEAD_WEALTH_GRID.stop, DEAD_WEALTH_GRID.n_points
    )
    consumption = np.linspace(
        CONSUMPTION_GRID.start, CONSUMPTION_GRID.stop, CONSUMPTION_GRID.n_points
    )
    health_transition = np.array(HEALTH_TRANSITION)
    exponent = 1.0 - risk_aversion

    def g(v: np.ndarray) -> np.ndarray:
        return v**exponent

    def g_inv(v: np.ndarray) -> np.ndarray:
        return v ** (1.0 / exponent)

    V_dead = np.sqrt(dead_wealth)
    n_decision_periods = len(SURVIVAL_PROBS)
    V_alive: dict[int, np.ndarray] = {}
    policy_c: dict[int, np.ndarray] = {}
    V_next: np.ndarray | None = None

    for period in reversed(range(n_decision_periods)):
        V_p = np.empty((len(wealth), 2))
        c_p = np.empty((len(wealth), 2))
        for iw, w in enumerate(wealth):
            for ih in range(2):
                survival = SURVIVAL_PROBS[period] * (
                    1.0 if ih == 1 else BAD_HEALTH_SURVIVAL_FACTOR
                )
                best_q, best_c = -np.inf, np.nan
                for c in consumption:
                    if c > w:
                        continue
                    w_next = w - c + INCOME
                    acc = (1.0 - survival) * g(np.interp(w_next, dead_wealth, V_dead))
                    if V_next is not None:
                        alive_vals = np.array(
                            [
                                np.interp(w_next, wealth, V_next[:, jh])
                                for jh in range(2)
                            ]
                        )
                        acc += survival * (health_transition[ih] @ g(alive_vals))
                    ce = g_inv(acc)
                    q = (
                        (1.0 - discount_factor) * c**rho + discount_factor * ce**rho
                    ) ** (1.0 / rho)
                    if q > best_q:
                        best_q, best_c = q, c
                V_p[iw, ih] = best_q
                c_p[iw, ih] = best_c
        V_alive[period] = V_p
        policy_c[period] = c_p
        V_next = V_p

    return V_alive, policy_c[0]


def test_epstein_zin_solved_values_match_numpy_reference():
    """The solved alive-V equals an independent numpy backward induction."""
    risk_aversion, discount_factor, rho = 0.5, 0.9, 0.5
    model = get_model(certainty_equivalent=PowerCertaintyEquivalent())
    solution = model.solve(
        params=get_params(
            risk_aversion=risk_aversion, discount_factor=discount_factor, rho=rho
        ),
        log_level="debug",
    )
    expected, _ = _reference_backward_induction(
        risk_aversion=risk_aversion, discount_factor=discount_factor, rho=rho
    )
    for period, expected_arr in expected.items():
        np.testing.assert_allclose(
            np.asarray(solution[period]["alive"]),
            expected_arr,
            rtol=5e-5,
            err_msg=f"period={period}",
        )
```

Details that matter:

- Feasibility uses `c > w: continue` — bare `>`, no epsilon: the only exactly-equal grid
  pair is `c = w = 0.5` (identical literals), which both the engine
  (`consumption <= wealth`) and the reference treat as feasible.

- `HealthStatus` is `ordered=True` with `bad = 0`, `good = 1`; the reference's `ih == 1`
  is `good`. The solved alive-V axis order is the `states` declaration order
  `(wealth, health)` — if the assertion fails with transposed shapes, check
  `solution[period]["alive"].shape` and transpose the expected array, once, with a
  comment.

- `rtol=5e-5` covers float32 (engine) vs float64 (reference); tighten to the tolerance
  style used in `tests/test_taste_shocks.py` if that file pins a different convention.

- [ ] **Step 2: Run to verify current status**

Run:
`pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py::test_epstein_zin_solved_values_match_numpy_reference -v`
Expected: PASS if Task 4's seam is correct — this test is the acceptance gate, not a
red-first test (the red driver was Task 4's divergence test). If it FAILS, debug the
seam against the reference before proceeding; do not loosen the tolerance beyond 1e-3
without understanding why.

- [ ] **Step 3: Write the simulation test**

Create `tests/simulation/test_simulate_certainty_equivalent.py`:

```python
"""Simulation under a nonlinear certainty equivalent."""

import jax.numpy as jnp
import numpy as np

from lcm import PowerCertaintyEquivalent
from tests.test_certainty_equivalent import _reference_backward_induction
from tests.test_models.epstein_zin_health import (
    EzRegimeId,
    HealthStatus,
    get_model,
    get_params,
)


def test_simulated_period0_consumption_matches_reference_policy():
    """Period-0 consumption equals the reference argmax at the initial states."""
    risk_aversion, discount_factor, rho = 0.5, 0.9, 0.5
    model = get_model(certainty_equivalent=PowerCertaintyEquivalent())
    params = get_params(
        risk_aversion=risk_aversion, discount_factor=discount_factor, rho=rho
    )
    initial_wealth = jnp.array([2.8, 7.4, 12.0])  # on-grid nodes
    initial_health = jnp.array([HealthStatus.bad, HealthStatus.good, HealthStatus.good])
    result = model.simulate(
        params=params,
        initial_conditions={
            "wealth": initial_wealth,
            "health": initial_health,
            "regime_id": jnp.full(3, EzRegimeId.alive),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    df = result.to_dataframe(use_labels=False)
    period0 = df.query("period == 0").sort_index()

    _, policy_c = _reference_backward_induction(
        risk_aversion=risk_aversion, discount_factor=discount_factor, rho=rho
    )
    wealth_grid = np.linspace(0.5, 12.0, 6)
    expected = np.array(
        [
            # Nearest-node lookup, not `searchsorted`: linspace values and
            # float literals can differ in the last bit.
            policy_c[int(np.argmin(np.abs(wealth_grid - w))), h]
            for w, h in [(2.8, 0), (7.4, 1), (12.0, 1)]
        ]
    )
    np.testing.assert_allclose(period0["consumption"].to_numpy(), expected, rtol=1e-5)
```

If the DataFrame's shape differs (column names, index structure, subject ordering),
mirror the access pattern from `tests/simulation/test_simulate_taste_shocks.py` — assert
against the same three subjects in their initial-conditions order.

- [ ] **Step 4: Run both test files**

Run:
`pixi run -e tests-cpu pytest tests/test_certainty_equivalent.py tests/simulation/test_simulate_certainty_equivalent.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_certainty_equivalent.py tests/simulation/test_simulate_certainty_equivalent.py
git commit -m "Pin Epstein-Zin solve and simulation against a numpy reference

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 6: Example model + docs page

**Files:**

- Create: `src/lcm_examples/epstein_zin.py`
- Create: `docs/examples/epstein_zin.md`
- Modify: `docs/examples/index.md` (add entry next to `mortality`)

**Interfaces:**

- Consumes: the public API from Tasks 1–4 only (`lcm` top-level imports).

- Produces: importable `lcm_examples.epstein_zin.get_model` / `get_params`.

- [ ] **Step 1: Create the example module**

`src/lcm_examples/epstein_zin.py` is the toy from Task 4 at example scale. Copy
`tests/test_models/epstein_zin_health.py` and change ONLY:

- The module docstring: drop the sentence about numpy-reference sizing; add one sentence
  pointing to `docs/examples/epstein_zin.md`.
- Grids and horizon:

```python
WEALTH_GRID = LinSpacedGrid(start=0.5, stop=40.0, n_points=100)
DEAD_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=40.0, n_points=100)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=10.0, n_points=100)

INCOME = 0.5
SURVIVAL_PROBS = tuple(np.linspace(0.97, 0.0, 20))
```

with `AgeGrid(start=60, stop=80, step="Y")` and `active=lambda age: age < 80`.
(`import numpy as np` at the top; keep everything else `jnp`.)

- `get_model` defaults to the Epstein–Zin configuration:
  `certainty_equivalent: CertaintyEquivalent | None = PowerCertaintyEquivalent()`
  (import `PowerCertaintyEquivalent` from `lcm`).
- Follow the import/structure conventions of `src/lcm_examples/mortality.py` (public
  `get_model`/`get_params` first if that file orders them so).

Smoke-check:

Run:
`pixi run -e tests-cpu python -c " from lcm_examples.epstein_zin import get_model, get_params m = get_model() m.solve(params=get_params(risk_aversion=5.0), log_level='progress') print('example solves') "`
Expected: `example solves` (a few seconds on CPU).

Note the deliberate asymmetry: the example keeps `risk_aversion=5.0 > 1` working because
next wealth stays ≥ `INCOME = 0.5` and the dead bequest `sqrt(wealth)` is strictly
positive there — call this out in the docs page as the design rule.

- [ ] **Step 2: Write the docs page**

Create `docs/examples/epstein_zin.md`. Before writing, open `docs/examples/mortality.md`
and mirror its exact structure (front matter, heading levels, how code is included or
quoted). Content to cover, in this order:

1. **The model.** One paragraph: toy consumer block in the spirit of Atal, Fang,
   Karlsson & Ziebarth (2025, *JPE* 133(6), doi:10.1086/734781) — savings, a two-state
   health Markov chain, health-dependent survival into a terminal `dead` regime — with
   Epstein–Zin preferences.

1. **The recursion and how it maps onto pylcm.** Display math:

   ```{math}
   V_t = \Bigl[(1-\beta)\,c_t^{\rho} + \beta\,\mathrm{CE}_t^{\rho}\Bigr]^{1/\rho},
   \qquad
   \mathrm{CE}_t = \Bigl(\mathbb{E}_t\bigl[V_{t+1}^{\,1-\gamma}\bigr]\Bigr)^{1/(1-\gamma)}
   ```

   and the mapping table: `utility` returns $c_t$; `H` implements the outer aggregator
   (its `E_next_V` argument receives the certainty equivalent);
   `certainty_equivalent=PowerCertaintyEquivalent()` implements $\mathrm{CE}_t$ with
   runtime param `{"certainty_equivalent": {"risk_aversion": γ}}`; the expectation runs
   jointly over the health Markov chain and the alive/dead regime transition, i.e.
   $\mathrm{CE} = g^{-1}\bigl(\sum_r p_r\,\mathbb{E}_w[g(V'_r)]\bigr)$ with
   $g(v) = v^{1-\gamma}$.

1. **Pitfalls.**

   - Positivity: power transforms need $V' > 0$; keep values in consumption units
     (utility = $c$, not $\log c$) and give death a strictly positive bequest — with
     $\gamma > 1$, a zero continuation yields $0^{1-\gamma} = \infty$.
   - Targets without a value contribution: a reachable target regime with no states
     contributes $0$ to the transformed sum, which equals $p_r \cdot g(0)$ only when
     $g(0)=0$ — model death with an explicit wealth state and bequest utility instead.
   - `risk_aversion = 1` (log CE) is not representable by the power pair.
   - Solver restriction: only `GridSearch`; `DCEGM` + `certainty_equivalent` is rejected
     at model build.

1. **Run it.** Code block constructing the model, solving, simulating a few subjects,
   and one `plotly.graph_objects` figure comparing simulated mean wealth paths for
   `risk_aversion=0.5` vs `5.0` (mirror the plotting style of the neighboring example
   pages; grey default + one accent color).

Add the page to `docs/examples/index.md` exactly the way `mortality.md` is listed (same
list/toctree syntax, alphabetical or thematic position next to it).

- [ ] **Step 3: Build the docs**

Run: `pixi run -e docs build-docs` Expected: build succeeds; `epstein_zin` page present
in the output (check the build log for the page name; warnings about it are failures to
fix).

- [ ] **Step 4: Commit**

```bash
git add src/lcm_examples/epstein_zin.py docs/examples/epstein_zin.md docs/examples/index.md
git commit -m "Add an Epstein-Zin lifecycle example with docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 7: Full verification, push, PR

**Files:** none new.

- [ ] **Step 1: Run the verification battery**

```bash
prek run --all-files
pixi run -e tests-cpu ty
pixi run -e tests-cpu tests -n 7
```

Expected: hooks pass (re-stage and re-run if formatters modify files), `ty` clean (fix
any new diagnostics with the narrowest change; `# ty: ignore[rule]` only with a reason),
full suite green. `pyproject.toml` was never touched, so no `pixi lock`.

- [ ] **Step 2: Push and open the PR against the #391 branch**

```bash
git push -u origin feat/certainty-equivalent
```

Write the PR body to a temp file in the scratchpad, covering: the seam formula, the
`taste_shocks`-mirroring API, the byte-identical default path, the DCEGM rejection +
finalize-location rationale, the toy-Atal EZ pinning tests, and `Closes #385`. End the
body with the standard Claude Code footer. Then:

```bash
gh pr create --repo OpenSourceEconomics/pylcm \
  --base feat/type-local-continuation-v \
  --head feat/certainty-equivalent \
  --title "Certainty-equivalent seam for non-EU preferences (Epstein-Zin)" \
  --body-file <scratchpad>/pr_body.md
```

IMPORTANT: `gh pr edit` is blocked by a hook — for any later PR metadata change use
`gh api repos/OpenSourceEconomics/pylcm/pulls/<n> -X PATCH -f base=... -F body=@file`.
When #391 merges into main, retarget this PR's base to `main` the same way.

- [ ] **Step 3: Report**

Post the PR URL and note the two coordination points for the team: (a) retarget to
`main` after #391 lands; (b) whichever of this PR and `feat/dcegm` (#390) merges second
resolves small conflicts in `Q_and_F.py`/`processing.py`/`contract.py` via the
pr-cascade workflow, and `feat/dcegm`'s real `DCEGM.validate` should keep rejecting
regimes whose `SolverBuildContext.certainty_equivalent` is not `None`.
