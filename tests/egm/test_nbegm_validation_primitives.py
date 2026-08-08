"""The NB-EGM smoothness gate rejects staircase and selection primitives.

The JAXPR gate exists to catch piecewise logic hidden inside an undecorated
helper, where the AST of the declared piece shows nothing. A helper that floors,
reduces with `max`/`min`, or takes an `argmax` is a case boundary the model never
declared, and NB-EGM would run its Euler inversion straight across it.
"""

import jax.numpy as jnp
import pytest

from _lcm.egm.nbegm_validation import find_ast_violations, find_jaxpr_violations


def _bracket(income):
    return jnp.floor(income / 1000.0)


def _cap(income):
    return jnp.max(jnp.array([income, 1000.0]))


def _pick(income):
    return jnp.argmax(jnp.array([income, 1000.0])) * 1.0


@pytest.mark.parametrize(
    ("func", "prim"),
    [(_bracket, "floor"), (_cap, "reduce_max"), (_pick, "argmax")],
)
def test_jaxpr_gate_rejects_a_staircase_or_selection_primitive(func, prim) -> None:
    """A hidden `floor` / `reduce_max` / `argmax` is reported by its primitive name."""
    violations = find_jaxpr_violations(func, abstract_args=(1.0,), mode="smooth_user")
    assert any(f"`{prim}`" in violation for violation in violations)


def test_jaxpr_gate_accepts_a_genuinely_smooth_formula() -> None:
    """A smooth polynomial passes the primitive gate."""

    def smooth(income):
        return 0.3 * income + 0.01 * income**2

    assert find_jaxpr_violations(smooth, abstract_args=(1.0,), mode="smooth_user") == []


def _indexes_an_array_parameter(schedule):
    return schedule[2]


def test_jaxpr_gate_reports_an_untraceable_piece_as_a_violation() -> None:
    """A piece the build-time fills cannot trace is a loud message, not a raw error."""
    violations = find_jaxpr_violations(
        _indexes_an_array_parameter, abstract_args=(1.0,), mode="smooth_user"
    )
    assert any("could not be traced" in violation for violation in violations)


def test_jaxpr_gate_warns_instead_of_reporting_under_assume_declared() -> None:
    """`probe_failure='assume_declared'` downgrades a tracing failure to a warning."""
    with pytest.warns(UserWarning, match="could not be traced"):
        violations = find_jaxpr_violations(
            _indexes_an_array_parameter,
            abstract_args=(1.0,),
            mode="smooth_user",
            probe_failure="assume_declared",
        )
    assert violations == []


def _floors(x):
    return jnp.floor(x)


def _argmins(x):
    return jnp.argmin(x)


def _compares(x):
    return jnp.greater(x, 1.0)


@pytest.mark.parametrize("piece", [_floors, _argmins, _compares])
def test_ast_gate_rejects_a_staircase_or_comparison_call(piece) -> None:
    """The AST gate names a piecewise call written directly in the piece."""
    violations = find_ast_violations(piece, mode="smooth_user")
    assert any("piecewise function" in violation for violation in violations)
