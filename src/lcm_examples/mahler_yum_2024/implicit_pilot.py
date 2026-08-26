"""Mahler-Yum pilot for the implicit outer derivative.

Wires `continuous_outer_optimum` — the custom-JVP outer optimum — into the
REAL paper-mode model: the pilot objective is the period kernel's own exact
adjuster-node solve `Q(f, theta) = V_adjuster(f; theta)` at the capture
period (36, the first period whose optimal effort is interior and off-node),
read at a small set of pilot state cells. The AD tangent `df*/dtheta` from the implicit
JVP must agree with a Richardson-extrapolated central difference of the
primal search, within the finite-difference method's own uncertainty.

Two scope facts, stated rather than hidden:

- The continuation entering the capture period is solved at the BASELINE
  theta and held fixed, so the pilot differentiates the per-period outer
  step — exactly the derivative the custom JVP contributes to a chain rule —
  not the full all-periods dV/dtheta.
- The pilot parameter is the effort-cost elasticity (`effort_elasticity`),
  chosen because it moves the interior effort optimum smoothly and enters
  the inner solve only through the flow utility.

The per-cell objective evaluates the full-state-space node solve once per
requested cell value via `jax.vmap` and reads the diagonal, so it stays a
genuine JAX-transformable function of `(f, theta)` — the forward-mode JVP
rule differentiates the actual nested inner solve, not a surrogate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.optimization.implicit_outer_derivative import (
    ImplicitOptimumDiagnostics,
    OwnerProvenance,
    continuous_outer_optimum,
    implicit_optimum_diagnostics,
)
from lcm.typing import FloatND

if TYPE_CHECKING:
    import _lcm.solution.nnbegm as _nnbegm
    from lcm.solvers import AdaptiveOuterMesh

PILOT_PERIOD = 36
_THETA_SUBSTRING = "effort_elasticity"


class _StopAfterCaptureError(Exception):
    pass


@dataclass
class PilotProblem:
    """The captured period kernel plus everything a node solve needs."""

    kernel: _nnbegm._NNBEGMPeriodKernel
    kernel_kwargs: dict
    adjuster_cores: Mapping[str, Callable]
    theta_key: str
    theta_baseline: float
    owner_provenance: Callable[[FloatND, FloatND], OwnerProvenance] | None = None
    """Optional compact complete witness; absent capture paths fail closed."""


def capture_pilot_problem(
    *, period: int = PILOT_PERIOD, mesh: AdaptiveOuterMesh | None = None
) -> PilotProblem:
    """Solve the paper model down to `period` and capture its kernel inputs.

    The backward induction is stopped by exception the moment the capture
    period's kernel is entered, so only the later periods are solved; the
    captured continuation objects are exactly what the production solve
    would hand this period.
    """
    import _lcm.solution.nnbegm as _nnbegm  # noqa: PLC0415
    from lcm import LinSpacedGrid  # noqa: PLC0415
    from lcm.solvers import AdaptiveOuterMesh  # noqa: PLC0415
    from lcm_examples.mahler_yum_2024 import (  # noqa: PLC0415
        START_PARAMS,
        create_inputs,
    )
    from lcm_examples.mahler_yum_2024.paper import (  # noqa: PLC0415
        adapt_params_to_paper_mode,
        create_mahler_yum_model,
    )

    if mesh is None:
        mesh = AdaptiveOuterMesh(
            initial_grid=LinSpacedGrid(start=0.0, stop=1.0, n_points=9),
            max_nodes=33,
            max_refinement_rounds=3,
            value_atol=1e-3,
            value_rtol=1e-3,
            golden_iterations=24,
        )
    captured: dict = {}
    original_call = _nnbegm._NNBEGMPeriodKernel.__call__  # noqa: SLF001

    def capturing_call(self: object, **kw: object) -> object:
        if kw["period"] == period:
            captured["kernel"] = self
            captured["kwargs"] = kw
            raise _StopAfterCaptureError
        return cast("Callable[..., object]", original_call)(self, **kw)

    # enable_jit=False keeps the kernel cores traceable: AOT-compiled
    # cores reject JAX transformations, and the pilot objective must be
    # vmapped (per-cell f) and forward-differentiated (the JVP rule).
    model = create_mahler_yum_model(
        implementation="paper", outer_search=mesh, enable_jit=False
    )
    model_params, _ = create_inputs(
        seed=0, n_simulation_subjects=10, params=START_PARAMS
    )
    params = adapt_params_to_paper_mode(model_params)
    setattr(  # noqa: B010 (deliberate white-box patch, restored in finally)
        _nnbegm._NNBEGMPeriodKernel,  # noqa: SLF001
        "__call__",
        capturing_call,
    )
    try:
        model.solve(params=params, log_level="off")
        msg = f"period {period} was never entered by the NNBEGM kernel"
        raise RuntimeError(msg)
    except _StopAfterCaptureError:
        pass
    finally:
        setattr(  # noqa: B010
            _nnbegm._NNBEGMPeriodKernel,  # noqa: SLF001
            "__call__",
            original_call,
        )
    kernel = captured["kernel"]
    kw = captured["kwargs"]
    regime_params = kw["flat_params"][kernel.regime_name]
    theta_keys = [key for key in regime_params if _THETA_SUBSTRING in key]
    if len(theta_keys) != 1:
        msg = (
            f"expected exactly one flat param containing {_THETA_SUBSTRING!r}, "
            f"found {theta_keys}"
        )
        raise RuntimeError(msg)
    return PilotProblem(
        kernel=kernel,
        kernel_kwargs=kw,
        adjuster_cores=_nnbegm._subcores(  # noqa: SLF001
            compiled_cores=kw["compiled_cores"], role="adjuster"
        ),
        theta_key=theta_keys[0],
        theta_baseline=float(regime_params[theta_keys[0]]),
    )


def node_value(
    problem: PilotProblem, node: FloatND | float, theta: FloatND | float
) -> FloatND:
    """The adjuster branch's exact value surface at outer node `node`.

    This is the SAME conditional inner solve the production mesh driver
    dispatches per node — with the pilot parameter overridden to `theta` —
    returning the full-state-space `V_arr`.
    """
    kw = problem.kernel_kwargs
    regime_name = problem.kernel.regime_name
    base = kw["flat_params"]
    flat_params = MappingProxyType(
        {
            **dict(base),
            regime_name: MappingProxyType(
                {**base[regime_name], problem.theta_key: jnp.asarray(theta)}
            ),
        }
    )
    result = problem.kernel._solve_adjuster_node(  # noqa: SLF001
        node=jnp.asarray(node),
        adjuster_cores=problem.adjuster_cores,
        state_action_space=kw["state_action_space"],
        next_regime_to_V_arr=kw["next_regime_to_V_arr"],
        next_regime_to_continuation=kw["next_regime_to_continuation"],
        flat_params=flat_params,
        period=kw["period"],
        ages=kw["ages"],
    )
    return result.V_arr


def select_pilot_cells(
    problem: PilotProblem, *, n_cells: int = 2, n_probe: int = 9
) -> np.ndarray:
    """Flat state-cell indices whose baseline effort optimum is interior.

    A coarse probe sweep locates cells whose argmax over the probe nodes is
    strictly interior and finite everywhere — the cells on which the
    implicit derivative is defined and the diagnostics should come back
    resolved. Chosen indices are spread across the qualifying set.
    """
    probes = np.linspace(0.0, 1.0, n_probe)
    probe_solve = jax.jit(
        lambda node: node_value(problem, node, problem.theta_baseline)
    )
    values = np.stack(
        [np.asarray(probe_solve(jnp.asarray(node))).reshape(-1) for node in probes],
        axis=-1,
    )
    finite = np.isfinite(values).all(axis=-1)
    best = values.argmax(axis=-1)
    interior = finite & (best > 0) & (best < n_probe - 1)
    candidates = np.flatnonzero(interior)
    if candidates.size < n_cells:
        msg = (
            f"only {candidates.size} interior pilot cells found "
            f"(need {n_cells}); the capture period is boundary-pinned"
        )
        raise RuntimeError(msg)
    positions = np.linspace(0, candidates.size - 1, n_cells).round().astype(int)
    return candidates[positions]


def build_pilot_objective(
    problem: PilotProblem, cell_indices: np.ndarray
) -> Callable[[FloatND, FloatND], FloatND]:
    """`Q(f, theta)` per pilot cell, as one JAX-transformable callable.

    `f` carries one value per pilot cell; each is solved by the full node
    solve under `jax.vmap` and the matching cell is read off the diagonal.
    """
    idx = jnp.asarray(np.asarray(cell_indices))

    def objective(f_arr: FloatND, theta: FloatND) -> FloatND:
        def one(node: FloatND) -> FloatND:
            return jnp.reshape(node_value(problem, node, theta), (-1,))[idx]

        return jnp.diagonal(jax.vmap(one)(f_arr))

    # Deliberately not jitted. Forward differentiation already traces the
    # objective at every mesh node into one graph; an additional jit nests a
    # compiled full-state solve inside that trace. Peak memory scales with
    # `n_mesh`, so the pilot keeps the mesh small.
    return objective


@dataclass
class PilotReport:
    """Everything section 19.3's acceptance decision needs, per pilot cell."""

    cell_indices: np.ndarray
    theta_key: str
    theta_baseline: float
    f_star: np.ndarray
    q_f: np.ndarray
    """`Q_f(f*)` per cell — the first-order residual; ~0 at a smooth interior
    optimum, sign-definite and material at a kink (the stationarity screen)."""
    q_ff: np.ndarray
    ad_tangent: np.ndarray
    fd_h: np.ndarray
    fd_h2: np.ndarray
    fd_richardson: np.ndarray
    fd_error_estimate: np.ndarray
    diagnostics: ImplicitOptimumDiagnostics
    no_valid_derivative_reason: tuple[str, ...]
    """Per-cell empty string when certified, otherwise a compact refusal reason."""


def no_valid_derivative_reasons(
    diagnostics: ImplicitOptimumDiagnostics,
) -> tuple[str, ...]:
    """Translate compact Boolean diagnostic fields into per-cell refusal text."""
    unresolved = np.asarray(diagnostics.unresolved, dtype=bool)
    shape = unresolved.shape
    fields = {
        name: np.broadcast_to(np.asarray(getattr(diagnostics, name), dtype=bool), shape)
        for name in (
            "at_lower_bound",
            "at_upper_bound",
            "flat_curvature",
            "basin_tie",
            "nonstationary",
            "owner_missing",
            "owner_incomplete",
            "owner_unresolved",
            "owner_primary_tie",
            "owner_changed",
        )
    }
    labels = (
        ("at_lower_bound", "outer optimum is at the lower bound"),
        ("at_upper_bound", "outer optimum is at the upper bound"),
        ("flat_curvature", "outer curvature is flat"),
        ("basin_tie", "outer basins are value-tied"),
        ("nonstationary", "outer optimum is nonstationary or formula-changing"),
        ("owner_missing", "complete owner provenance is unavailable"),
        ("owner_incomplete", "owner provenance is incomplete"),
        ("owner_unresolved", "an exact owner/status fact is unresolved"),
        ("owner_primary_tie", "the exact primary owner value is tied"),
        ("owner_changed", "the composite owner signature changes in the neighborhood"),
    )
    reasons: list[str] = []
    for index in np.ndindex(shape or ()):
        if not bool(unresolved[index]):
            reasons.append("")
            continue
        active = [label for name, label in labels if bool(fields[name][index])]
        reasons.append("; ".join(active) or "local derivative is unresolved")
    return tuple(reasons)


def run_pilot(
    problem: PilotProblem,
    cell_indices: np.ndarray,
    *,
    n_mesh: int = 9,
    polish_iterations: int = 28,
    relative_step: float = 1e-2,
) -> PilotReport:
    """AD (implicit JVP) vs Richardson-extrapolated central differences.

    `fd_error_estimate` is the standard Richardson error proxy
    `|D(h/2) - D(h)| / 3`; the acceptance band belongs to the caller (the
    test), per section 19.3: AD is rejected on disagreement, not FD.
    """
    objective = build_pilot_objective(problem, cell_indices)
    n = len(cell_indices)
    bounds = (jnp.zeros(n), jnp.ones(n))
    theta0 = jnp.asarray(problem.theta_baseline)
    step = relative_step * max(1.0, abs(problem.theta_baseline))

    def solve_at(theta: FloatND) -> tuple[FloatND, FloatND, FloatND]:
        return continuous_outer_optimum(
            objective, theta, bounds, n_mesh, polish_iterations
        )

    f_star, _value, margin = solve_at(theta0)
    ad = jax.jacfwd(lambda t: solve_at(t)[0])(theta0)
    ones = jnp.ones_like(f_star)
    q_f = jax.jvp(lambda f: objective(f, theta0), (f_star,), (ones,))[1]
    _, q_ff = jax.jvp(
        lambda f: jax.jvp(lambda g: objective(g, theta0), (f,), (ones,))[1],
        (f_star,),
        (ones,),
    )

    f_plus_h = solve_at(theta0 + step)[0]
    f_minus_h = solve_at(theta0 - step)[0]
    half_step = step / 2.0
    f_plus_h2 = solve_at(theta0 + half_step)[0]
    f_minus_h2 = solve_at(theta0 - half_step)[0]
    fd_h = (np.asarray(f_plus_h) - np.asarray(f_minus_h)) / (2.0 * step)
    fd_h2 = (np.asarray(f_plus_h2) - np.asarray(f_minus_h2)) / (2.0 * half_step)
    richardson = (4.0 * fd_h2 - fd_h) / 3.0
    diagnostics = implicit_optimum_diagnostics(
        objective,
        theta=theta0,
        f_star=f_star,
        basin_margin=margin,
        bounds=bounds,
        n_mesh=n_mesh,
        polish_iterations=polish_iterations,
        owner_provenance=problem.owner_provenance,
        require_owner_certificate=True,
        reoptimized_owner_points=(
            (f_plus_h, theta0 + step),
            (f_minus_h, theta0 - step),
            (f_plus_h2, theta0 + half_step),
            (f_minus_h2, theta0 - half_step),
        ),
    )
    return PilotReport(
        cell_indices=np.asarray(cell_indices),
        theta_key=problem.theta_key,
        theta_baseline=problem.theta_baseline,
        f_star=np.asarray(f_star),
        q_f=np.asarray(q_f),
        q_ff=np.asarray(q_ff),
        ad_tangent=np.asarray(ad),
        fd_h=fd_h,
        fd_h2=fd_h2,
        fd_richardson=richardson,
        fd_error_estimate=np.abs(fd_h2 - fd_h) / 3.0,
        diagnostics=diagnostics,
        no_valid_derivative_reason=no_valid_derivative_reasons(diagnostics),
    )
