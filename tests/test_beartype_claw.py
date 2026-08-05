"""The beartype claw is live on the entire `lcm` package.

The claw uses `INTERNAL_CONF`, so type violations in internal helpers
surface as beartype's own `BeartypeCallHintViolation`. User-facing
constructors (`Model`, `Regime`, `MarkovTransition`, every grid and shock,
`@categorical`, `as_leaf`) carry their own explicit `@beartype(conf=...)`
decorators that map violations to the relevant project exception
(`ModelInitializationError`, `RegimeInitializationError`,
`GridInitializationError`, `InvalidParamsError`); those decorators stack
on top of the claw and win at the user boundary.

Each `test_claw_checks_*` test calls an internal function with one argument
of the wrong type, chosen so the call would return cleanly if the function
were *not* instrumented — the violation is what proves the claw is
installed. Each `test_*_with_bad_arg_raises_project_exception` test
confirms that an ill-typed argument to a public constructor surfaces as
the project exception, not as `BeartypeCallHintViolation`.
"""

import ast
import importlib
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype.door import is_bearable
from beartype.roar import BeartypeCallHintViolation
from jaxtyping import Int, Scalar

import _lcm
import lcm
from _lcm.egm.upper_envelope.query import _Dyadic, _exact_sign_of_sum
from _lcm.engine import _build_regime_sharding
from _lcm.optimization.golden_section import maximize_golden_section
from _lcm.regime_building.max_Q_over_a import get_argmax_and_max_Q_over_a
from _lcm.simulation.simulate import _compute_starting_periods
from _lcm.solution.diagnostics import _log_per_period_stats
from _lcm.state_action_space import _validate_all_states_present
from _lcm.transition_checks import _validate_regime_transition_probs
from lcm import AgeGrid, LinSpacedGrid, Model
from lcm.exceptions import (
    GridInitializationError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.koopmans_aggregation import LinearAggregator
from lcm.regime import Regime as UserRegime


def test_claw_checks_lcm_simulation() -> None:
    """Type-violating arguments to internal `_lcm.simulation` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _compute_starting_periods(
            initial_ages=np.array([25.0]),  # ty: ignore[invalid-argument-type]
            ages=AgeGrid(start=25, stop=75, step="Y"),
        )


def test_claw_checks_lcm_solution() -> None:
    """Type-violating arguments to internal `_lcm.solution` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _log_per_period_stats(
            logger="not a logger",  # ty: ignore[invalid-argument-type]
            diagnostic_rows=[],
            mins=jnp.array([]),
            maxs=jnp.array([]),
            means=jnp.array([]),
        )


def test_claw_checks_lcm_transition_checks() -> None:
    """Type-violating arguments to `_lcm.transition_checks` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _validate_regime_transition_probs(
            regime_transition_probs={"working": jnp.array([1.0])},  # ty: ignore[invalid-argument-type]
            active_regimes_next_period=("working",),
            regime_name="working",
            age=50.0,
            next_age=51.0,
        )


def test_claw_checks_lcm_state_action_space() -> None:
    """Type-violating arguments to `_lcm.state_action_space` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _validate_all_states_present(
            provided_states="",  # ty: ignore[invalid-argument-type]
            required_state_names=set(),
        )


def test_claw_checks_lcm_engine() -> None:
    """Type-violating arguments to `_lcm.engine` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        _build_regime_sharding(
            grids=MappingProxyType({}),
            n_devices="not an int",  # ty: ignore[invalid-argument-type]
        )


def test_claw_checks_lcm_regime() -> None:
    """Type-violating arguments to `lcm.regime` helpers raise."""
    with pytest.raises(BeartypeCallHintViolation):
        LinearAggregator()(
            utility=np.array([1.0]),  # ty: ignore[invalid-argument-type]
            CE=jnp.array([1.0]),
            discount_factor=jnp.array([0.95]),
        )


def test_claw_allows_with_signature_wrapper_over_named_param_function() -> None:
    """A `with_signature` wrapper over a function with named parameters stays
    callable under the package claw.

    `get_argmax_and_max_Q_over_a` returns a `dags.with_signature` wrapper
    around a function whose parameters — `next_regime_to_V_arr` plus the
    `**states_actions_params` it expands — are named explicitly rather than
    being a bare `*args, **kwargs` forwarder. The claw decorates that
    wrapper. The wrapper advertises a permissive forwarder via its
    `__annotations__`, so the claw must enforce nothing against the
    synthetic, annotation-free `__signature__`; otherwise every call fails
    because each parameter is checked against the `inspect.Parameter.empty`
    sentinel.
    """

    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[str, jnp.ndarray],  # noqa: ARG001
        action: jnp.ndarray,
        state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return action, action >= state

    argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=("action",),
        state_names=("state",),
    )

    argmax, maximum = argmax_and_max_Q_over_a(
        next_regime_to_V_arr=MappingProxyType({"working": jnp.arange(3.0)}),
        action=jnp.array([0.0, 1.0, 2.0]),
        state=jnp.array(0.0),
    )

    assert int(argmax) == 2
    assert float(maximum) == 2.0


def _fori_loop_body_index_annotations() -> list[tuple[str, str, str]]:
    """Every `fori_loop` body index annotation in the clawed packages.

    Derived by scanning the packages rather than listed, so a new `fori_loop`
    is covered the day it is written instead of the day someone remembers to
    extend an enumeration.

    Returns `(module, body function, annotation source)` triples. Bodies with an
    unannotated index (bare lambdas) are skipped: there is nothing for the claw
    to enforce against.
    """
    annotations = []
    for package in (_lcm, lcm):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            # Explicit encoding: the default is the locale's, which is cp1252 on
            # the Windows runner, and the sources this scans contain UTF-8
            # em-dashes. Without it the test fails with a `UnicodeDecodeError`
            # that has nothing to do with what it is checking.
            tree = ast.parse(path.read_text(encoding="utf-8"))
            bodies = {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            }
            module = ".".join(
                (package.__name__, *path.relative_to(root).with_suffix("").parts)
            ).removesuffix(".__init__")
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not ast.unparse(node.func).endswith("fori_loop"):
                    continue
                # `fori_loop(lower, upper, body, init)` — the body is third.
                body_arg = node.args[2]
                if not isinstance(body_arg, ast.Name):
                    continue
                index = bodies[body_arg.id].args.args[0]
                if index.annotation is None:
                    continue
                annotations.append((module, body_arg.id, ast.unparse(index.annotation)))
    return annotations


def test_every_fori_loop_body_index_annotation_admits_a_python_int() -> None:
    """A `fori_loop` body must accept the EAGER index, which is a plain `int`.

    `fori_loop` called with static Python-int bounds really loops in Python
    under `jax.disable_jit()` and hands the body an `int`; only under trace is
    the index a tracer. The claw is registered unconditionally, so an
    array-only annotation turns every eager call into a type violation — a
    deterministic failure with no numerical symptom, which is how it survived a
    green suite until a jitted-vs-eager agreement test became the first eager
    caller of the exact-arithmetic kernel in `upper_envelope.query`.

    This is the class, not the witness: the sites are discovered by scanning.
    """
    array_only = Int[Scalar, ""]
    assert not is_bearable(0, array_only), (
        "self-check: the predicate must be able to FAIL, otherwise this test "
        "passes vacuously for every annotation"
    )

    sites = _fori_loop_body_index_annotations()
    assert len(sites) >= 1, (
        f"the scan found {len(sites)} annotated `fori_loop` bodies; the known "
        "one lives in `_lcm.optimization.golden_section`, so a smaller count "
        "means the scan broke, not that the code is clean. The second site, in "
        "`_lcm.egm.upper_envelope.query`, went away with the round-13 exact "
        "kernel: it carries arrays through a `lax.scan` and annotates no loop "
        "index at all."
    )

    rejected = []
    for module, body, annotation in sites:
        namespace = vars(importlib.import_module(module))
        hint = eval(annotation, namespace)  # noqa: S307
        if not is_bearable(0, hint):
            rejected.append(f"{module}.{body}: {annotation}")
    assert not rejected, (
        "these `fori_loop` body indices reject the eager Python `int`: "
        + "; ".join(rejected)
    )


def test_the_exact_sign_kernel_runs_eagerly() -> None:
    """`_exact_sign_of_sum` is reachable with the claw live and jit disabled.

    This test exists because a `jax.Array`-only annotation on a loop index made
    every EAGER call raise a beartype violation, and the exact sign kernel is
    reached eagerly only from `test_jitted_solve_matches_the_eager_solve`. The
    round-13 rewrite carries arrays through a `lax.scan` instead of an index
    through a `fori_loop`, so the original hazard is gone; the eager reachability
    guard is kept because that is what caught it.
    """
    terms = _Dyadic(
        mantissa=jnp.array([0.5, -0.5, 0.5]),
        exponent=jnp.array([1, 1, -29], dtype=jnp.int32),
    )
    with jax.disable_jit():
        sign = _exact_sign_of_sum(terms)
    assert float(sign) == 1.0


def test_golden_section_runs_eagerly() -> None:
    """`maximize_golden_section` is reachable with the claw live and jit off."""
    with jax.disable_jit():
        result = maximize_golden_section(
            lambda x: -((x - 0.25) ** 2),
            lower=jnp.array([0.0, -1.0]),
            upper=jnp.array([1.0, 2.0]),
            iterations=8,
        )
    assert np.allclose(np.asarray(result.x), 0.25, atol=1e-2)


def test_regime_with_bad_arg_raises_project_exception() -> None:
    """A bad `Regime` argument surfaces as `RegimeInitializationError`."""
    with pytest.raises(RegimeInitializationError):
        UserRegime(
            transition=None,
            states={"wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3)},
            functions="not a mapping",  # ty: ignore[invalid-argument-type]
        )


def test_model_with_bad_arg_raises_project_exception() -> None:
    """A bad `Model` argument surfaces as `ModelInitializationError`."""
    with pytest.raises(ModelInitializationError):
        Model(
            ages=AgeGrid(start=25, stop=75, step="Y"),
            regimes="not a mapping",  # ty: ignore[invalid-argument-type]
            regime_id_class=int,
        )


def test_linspaced_grid_with_bad_arg_raises_project_exception() -> None:
    """A bad `LinSpacedGrid` argument surfaces as `GridInitializationError`."""
    with pytest.raises(GridInitializationError):
        LinSpacedGrid(
            start="not a number",  # ty: ignore[invalid-argument-type]
            stop=10.0,
            n_points=3,
        )
