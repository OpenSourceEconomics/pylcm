"""The DC-EGM solver (discrete-continuous endogenous grid method).

`DCEGM` configures one regime's Euler-inversion solve. Its
`build_period_kernels` returns one `PeriodKernel` per period — a non-jitted
adapter that wraps the shared jitted EGM step (deduped by function identity, so
periods sharing a core reuse one compiled program), calls it with the DC-EGM
argument layout, and assembles a `KernelResult` (value array, continuation
carry, published simulation policy) outside JIT.

The kernel-building imports (`jax`, `build_egm_step_functions`) are
function-local so the public `lcm.solvers` façade stays a thin re-export that
pulls in no numerical engine modules.
"""

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    SolutionKernels,
    Solver,
    SolverBuildContext,
)
from _lcm.typing import (
    EGMStepFunction,
    FlatParams,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.exceptions import RegimeInitializationError
from lcm.typing import (
    ActionName,
    FloatND,
    FunctionName,
    StateName,
)

# A non-concave candidate chain folds into at least two resource-increasing runs,
# so a smaller fold capacity could never publish one.
_MIN_ENVELOPE_MAX_RUNS: int = 2


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class DCEGM(Solver):
    """Configuration of the DC-EGM solver for one regime.

    DC-EGM inverts the Euler equation on an exogenous end-of-period
    (post-decision) grid instead of searching a dense grid for the continuous
    action. It requires a specific model structure — exactly one continuous
    (*Euler*) state and one continuous action, a declared resources function
    `R` with consumption recovery `c = R - A`, a post-decision function `A`,
    and an `inverse_marginal_utility` regime function — which is validated at
    `Model` construction time.

    Forward simulation works but is *grid-restricted*: `simulate` recomputes
    the argmax over the regime's gridded continuous action against the
    stored value function, rather than interpolating the exact EGM policy.
    Simulated continuous actions therefore live on the action grid, and with
    taste shocks the simulated choice frequencies follow the grid-restricted
    choice-specific values, not exactly the solve's choice probabilities.
    The budget constraint the solve enforces intrinsically
    (`continuous_action <= resources - savings_grid lower bound`) is applied
    as a feasibility mask during simulation.

    """

    continuous_state: StateName
    """Name of the Euler continuous state (e.g. `"wealth"`).

    Its transition must consume the post-decision function and reach the
    state and the continuous action only through it.
    """

    continuous_action: ActionName
    """Name of the continuous action (e.g. `"consumption"`)."""

    resources: FunctionName
    """Name of the resources function `R` in `Regime.functions`.

    Resources are what consumption is paid out of; the endogenous grid lives
    in R-space. Required even in the classic case, where it is the identity
    (e.g. `"resources": lambda wealth: wealth`). Must not depend on the
    continuous action and must be non-decreasing in the continuous state.
    In an `NEGM` regime with a declared `outer_cost`, the function of this
    name is composed by pylcm at model build
    (`<resources>_before_outer_cost - <outer_cost>`) and must not be defined
    by the regime.
    """

    post_decision_function: FunctionName
    """Name of the post-decision function in `Regime.functions`.

    The end-of-period state (e.g. savings), satisfying
    `post_decision = resources - continuous_action`.
    """

    savings_grid: ContinuousGrid
    """Exogenous end-of-period grid; its lower bound is the borrowing limit.

    The endogenous grid inherits this grid's spacing, and the published value
    function is interpolated linearly between endogenous points — so this grid
    controls where the solution is accurate. With sharply curved utility (e.g.
    CRRA), cluster the nodes toward the borrowing limit (`LogSpacedGrid`, or
    an `IrregSpacedGrid` clustered at the low end): a uniform grid
    under-resolves the value function near the limit, and that interpolation
    error compounds across periods.
    """

    upper_envelope: Literal["exact", "fues", "rfc", "ltm", "mss"] = "exact"
    """Upper-envelope refinement backend removing dominated Euler candidates.

    `"exact"` is pylcm's own construction and the default. The other four are
    faithful ports of the method columns of Dobrescu & Shanker 2024, kept for
    method comparison and speed; each carries the accepted limitation of its
    lineage, so prefer the default unless you are reproducing a method.

    - `"exact"`: the exact segment envelope. Splits the candidate chain into
      x-monotone runs and partitions resources at the live abscissae. Every run
      covering a cell covers all of it, so the envelope there is the maximum of
      full lines — convex, with owners in increasing slope order — and its owner
      sequence is resolved per cell. A branch owning only a subinterval survives,
      and a crossing landing exactly on a node separates the two policies. All
      structural decisions use a certified sign of the value difference, exact
      for the represented inputs and invariant to a common value level, and every
      live branch is certified against the owners rather than only the runner-up;
      an undecidable comparison, a branch escaping certification, or a chain
      folding into more than `envelope_max_runs` runs poisons the row rather than
      publishing a guess. Costs several times the fast scans (see
      `envelope_max_runs`).
    - `"fues"`: the Fast Upper-Envelope Scan — a sequential scan that inserts
      exact segment-crossing points. Fastest, but shares the fast-scan lineage's
      accepted limitation at *exact* endogenous-grid coincidence across branches
      (pointwise-node reduction can bridge a coincident-node crossing; endpoint
      crossings are snapped within a fixed band). Prefer `"mss"` when a model can
      realize exact coincidence and needs it resolved exactly.
    - `"rfc"`: the Rooftop-Cut algorithm — a parallel dominance test that only
      deletes points (a kink lands between retained points, recovered by the
      Hermite carry read) and generalizes to multidimensional grids.
    - `"ltm"`: the local-upper-bound brute method — an `O(K^2)` dense segment
      scan that evaluates the envelope at every candidate abscissa (the
      quadratic baseline of Dobrescu & Shanker 2024; a kink lands between
      output nodes, recovered by the downstream read).
    - `"mss"`: HARK's EGM upper envelope — a left-to-right sweep that keeps the
      max-value branch at every abscissa *and* inserts the exact
      segment-crossing point, so it tracks the FUES envelope tightly (the `MSS`
      method of Dobrescu & Shanker 2024). It resolves exact coincident-node
      interval ownership that the fast scans miss, but samples winners only at
      candidate abscissae: a branch owning just an interior subinterval is
      dropped, a crossing landing exactly on a node is not emitted, and its
      tie band scales with the value level rather than the compared margin.
      Use `"exact"` unless you are reproducing the published method.
    """

    envelope_max_runs: int = 24
    """Fold capacity of the `"exact"` upper envelope.

    The candidate chain is split into maximal resource-increasing runs; this
    bounds how many such runs a cell may fold into. It is a validated capacity,
    not an assumed bound — one discrete action can fold arbitrarily often, so a
    chain exceeding it poisons the row and surfaces through the solve loop's NaN
    diagnostics instead of silently dropping a branch.

    Three costs scale differently in this value, and the difference matters when
    tuning it:

    - **memory** is linear. Ownership is resolved per cell, so the capacity sets
      how wide each cell's working set is; the published row is row-sized
      whatever the capacity, but the intermediates that produce it are not. What
      bounds the peak is `envelope_cell_batch_size`, which caps how many cells
      are in flight at once.
    - **certified comparisons** are linear. Clearing a branch that owns nothing
      takes one exact comparison, at the breakpoint where the envelope's slope
      brackets the branch's own — not one against every rival.
    - **ordinary arithmetic** is quadratic: the owner walk can open one piece per
      run, and each step reads every run covering the cell.

    `24` covers the fold counts pylcm's own NEGM and DC-EGM models realize, with
    headroom; lower it for a model known to stay concave to buy back some speed.
    The quadratic term is the one that pays for it, and it is also what dominates
    XLA compile time for the solve kernel on GPU.
    """

    envelope_cell_batch_size: int | None = None
    """How many node cells the `"exact"` envelope resolves in parallel.

    Node cells are independent — each is resolved from the links covering it
    alone — so this partitions the work and can never change a published value or
    policy. What it sets is how much of that work is in flight, and so the
    working set: an intermediate of this value times `envelope_max_runs` per row,
    with the rows themselves mapped over, so the peak carries the product of all
    three.

    `None` resolves the cells one at a time and is the floor on the working set.
    An integer trades that memory for parallelism across cells, which pays when a
    single cell leaves the device idle and costs both memory and time when it
    does not — so measure on the model at hand rather than raising it on
    principle. A chain with fewer cells than the batch size resolves in one step
    either way, so small models pay nothing for the knob.
    """

    fues_jump_thresh: float = 2.0
    """Segment-switch threshold on `|ΔA / ΔR|` in the FUES scan."""

    fues_n_points_to_scan: int | None = None
    """Number of points the FUES forward scan inspects after a candidate.

    `None` (the default) scans exhaustively — every other candidate. That is the
    only width proven correct when more than the window's worth of off-segment
    candidates interleave between two points of one segment: a bounded window
    then misses the segment's continuation and silently accepts the dominated
    interlopers. A finite value keeps the cheaper bounded scan for models known
    to stay within the window, trading that correctness guarantee for speed.
    """

    fues_scan_unroll: int = 1
    """Loop-unroll factor for the FUES candidate `lax.scan`.

    Passed to `jax.lax.scan(..., unroll=fues_scan_unroll)` in the full-envelope
    scan of `refine_envelope`. The bracket path slices that same full row, so
    there is no separate streaming scan to unroll. The scan is sequential and
    latency-bound on accelerators; unrolling `k` iterations into one loop body
    trades compile time and code size for fewer loop-carry round trips, which can
    cut the per-row exec wall on GPU. `1` (no unroll) is the default; the refined
    envelope is numerically identical across values, so this is a pure
    performance knob.
    """

    rfc_jump_thresh: float = 2.0
    """Segment-switch threshold on `|Δc / ΔR|` in the rooftop cut."""

    rfc_search_radius: int = 10
    """Number of neighbors on each side the rooftop-cut dominance test inspects."""

    refined_grid_factor: float = 2.0
    """Headroom factor sizing the refined (NaN-padded) envelope arrays.

    The refined row holds one slot per envelope point, and every kink costs two
    (the outgoing owner's reading, then the incoming one). How often a row kinks
    is a property of the candidate cloud rather than of the grid, so this is
    headroom, not a bound: a row needing more slots than it has reports overflow
    and is NaN-poisoned rather than silently truncated. The default is sized for
    the `"exact"` upper envelope, which emits every ownership change — including
    the interior-only and node-aligned ones the fast scans miss — and therefore
    kinks more often than they do.
    """

    n_constrained_points: int = 20
    """Number of closed-form points on the credit-constrained segment."""

    stochastic_node_batch_size: int = 0
    """Block size for splaying the child stochastic-node expectation.

    The continuation expectation runs over the product of the child regime's
    stochastic process nodes — a single mesh, not a per-grid axis, so it gets
    its own solve-level knob rather than a per-grid `batch_size`. A positive
    value below the mesh length processes that expectation in `lax.map` blocks
    instead of one fused vmap, shedding the dominant `egm_step` working buffer
    (which carries this node axis); `0` keeps the fused vmap. Like the savings
    grid's `batch_size`, this is a memory knob only — the solved value function
    is identical to the unsplayed solve.
    """

    def __post_init__(self) -> None:
        _fail_if_savings_grid_is_stochastic(self.savings_grid)
        _fail_if_refined_grid_factor_too_small(self.refined_grid_factor)
        _fail_if_fues_jump_thresh_non_positive(self.fues_jump_thresh)
        _fail_if_n_constrained_points_too_few(self.n_constrained_points)
        _fail_if_fues_n_points_to_scan_too_few(self.fues_n_points_to_scan)
        _fail_if_fues_scan_unroll_too_few(self.fues_scan_unroll)
        _fail_if_envelope_max_runs_too_few(self.envelope_max_runs)
        _fail_if_envelope_cell_batch_size_non_positive(self.envelope_cell_batch_size)
        _fail_if_rfc_jump_thresh_non_positive(self.rfc_jump_thresh)
        _fail_if_rfc_search_radius_too_few(self.rfc_search_radius)
        _fail_if_stochastic_node_batch_size_negative(self.stochastic_node_batch_size)

    @property
    def requires_continuation(self) -> bool:
        """DC-EGM inverts the Euler equation against its targets' marginals."""
        return True

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one DC-EGM period adapter per period and the carry template.

        The standalone `validate_dcegm_regimes` model-contract check (run during
        regime processing) guarantees the regime is non-terminal, so the regime
        transition probability function exists. Periods sharing one EGM-step core
        reuse a single jitted core, and therefore a single compiled program.
        Numerical-builder imports are function-local so the public `lcm.solvers`
        façade stays a thin re-export that pulls in no engine modules.
        """

        from _lcm.egm.step import build_egm_step_functions  # noqa: PLC0415

        assert context.compute_regime_transition_probs is not None  # noqa: S101
        egm_step, egm_carry_template, egm_carry_targets = build_egm_step_functions(
            solver=self,
            regime_name=context.regime_name,
            user_regimes=context.user_regimes,
            functions=context.functions,
            constraints=context.constraints,
            transitions=context.transitions,
            stochastic_transition_names=context.stochastic_transition_names,
            compute_regime_transition_probs=context.compute_regime_transition_probs,
            regime_to_v_interpolation_info=context.regime_to_v_interpolation_info,
            period_to_regime_v_interp=context.period_to_regime_v_interp,
            period_to_regime_grid_signature=context.period_to_regime_grid_signature,
            solution_reachability=context.solution_reachability,
            flat_param_names=context.flat_param_names,
            regime_to_flat_param_names=context.regime_to_flat_param_names,
            state_action_space=context.state_action_space,
            has_taste_shocks=context.has_taste_shocks,
        )
        if context.enable_jit:
            jitted_by_id: dict[int, EGMStepFunction] = {}
            for func in egm_step.values():
                if id(func) not in jitted_by_id:
                    jitted_by_id[id(func)] = jax.jit(func)
            egm_step = MappingProxyType(
                {period: jitted_by_id[id(func)] for period, func in egm_step.items()}
            )
        period_kernels = MappingProxyType(
            {
                period: _DCEGMPeriodKernel(
                    core=core,
                    regime_name=context.regime_name,
                    carry_targets=egm_carry_targets,
                    transition_target_names=tuple(context.transitions),
                )
                for period, core in egm_step.items()
            }
        )
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_template=egm_carry_template,
        )


@dataclass(frozen=True, kw_only=True)
class _DCEGMPeriodKernel:
    """The DC-EGM period adapter — wraps one EGM-step core.

    Closes over the regime name, its carry targets, and the names of
    its transition targets (to union their params). Calling it inverts the Euler
    equation on the savings grid and returns a `KernelResult` carrying the value
    function, the continuation a parent interpolates, and the published off-grid
    simulation policy.
    """

    core: Callable
    """The shared jitted EGM-step core (`id`-deduped across periods)."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    carry_targets: frozenset[RegimeName]
    """The carry keys the EGM core reads; the rolling carry is filtered to these."""

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    def cores(self) -> Mapping[str, Callable]:
        """Return the single EGM-step core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _DCEGMPeriodKernel:
        """Bind the regime's and its carry targets' fixed params into the core.

        A DC-EGM source carrying into a *different* target regime evaluates that
        target's resources / transition functions in its per-asset-node solve,
        reading the target's fixed params. The core threads its `**kwargs`
        straight into the per-combo pool those captured functions read, so
        binding the union of the regime's and its carry targets' fixed params
        restores the values removed from the live `flat_params` for all of them
        at once.
        """
        egm_fixed = dict(fixed_flat_params.get(self.regime_name, MappingProxyType({})))
        for target_name in self.transition_target_names:
            for key, value in fixed_flat_params.get(
                target_name, MappingProxyType({})
            ).items():
                egm_fixed.setdefault(key, value)
        if not egm_fixed:
            return self
        return replace(self, core=functools.partial(self.core, **egm_fixed))

    def build_lower_args(
        self,
        *,
        core_key: str = "main",  # noqa: ARG002
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: states, carries, EGM params."""
        return {
            **dict(state_action_space.states),
            "next_regime_to_continuation": _carry_subset(
                next_regime_to_continuation=next_regime_to_continuation,
                carry_targets=self.carry_targets,
            ),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **self._egm_kernel_params(flat_params=flat_params),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Run the DC-EGM step and assemble the `KernelResult`."""
        V_arr, egm_carry, sim_policy = compiled_cores["main"](
            **state_action_space.states,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=_carry_subset(
                next_regime_to_continuation=next_regime_to_continuation,
                carry_targets=self.carry_targets,
            ),
            **self._egm_kernel_params(flat_params=flat_params),
            period=jnp.int32(period),
            age=ages.values[period],
        )
        return KernelResult(
            V_arr=V_arr, continuation=egm_carry, simulation_policy=sim_policy
        )

    def _egm_kernel_params(self, *, flat_params: FlatParams) -> dict[str, object]:
        """Flat params fed into the DC-EGM core: the source's plus its targets'.

        A DC-EGM source carrying into a *different* target regime evaluates that
        target's resources / transition functions in its per-asset-node solve,
        reading the target's params (e.g. a pension factor the source never
        reads). These are model-level shared values, so the target's
        `flat_params` entry carries the right value; union them in. The core
        threads its `**kwargs` into the per-combo pool, and its captured
        functions read only the keys they need, so a target's extra params are
        harmless to the source functions that do not. Mirrors the fixed-param
        binding done at model build (`_partial_fixed_params_into_regimes`) for
        the free-param path.
        """
        params: dict[str, object] = dict(flat_params[self.regime_name])
        for target_name in self.transition_target_names:
            for key, value in flat_params.get(
                target_name, MappingProxyType({})
            ).items():
                params.setdefault(key, value)
        return params


def _carry_subset(
    *,
    next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
    carry_targets: frozenset[RegimeName],
) -> MappingProxyType[RegimeName, EGMCarry]:
    """Return the carries a regime's EGM core actually reads.

    Each core only ever indexes `next_regime_to_continuation[target]` for its
    carry targets, so the full all-regimes mapping is needlessly large.
    Filtering to that subset keeps the core's carry pytree input
    minimal — only this subset is passed per call rather than every regime's
    carry at once.

    Iterates the source mapping's key order (stable across rolls) so the
    filtered pytree structure matches between lowering and call. Membership is
    tested defensively because a target need not publish a carry in every model.
    """
    return MappingProxyType(
        {
            name: next_regime_to_continuation[name]
            for name in next_regime_to_continuation
            if name in carry_targets
        }
    )


def _fail_if_savings_grid_is_stochastic(savings_grid: ContinuousGrid) -> None:
    if isinstance(savings_grid, _ContinuousStochasticProcess):
        msg = (
            "DCEGM.savings_grid must be a deterministic continuous grid, not a "
            f"stochastic process ({type(savings_grid).__name__}). The savings "
            "grid is the exogenous end-of-period grid; it carries no transition."
        )
        raise RegimeInitializationError(msg)


def _fail_if_refined_grid_factor_too_small(refined_grid_factor: float) -> None:
    # `not (x > 1.0)` rejects NaN too — `nan <= 1.0` is False, so a bare
    # `<= 1.0` guard would admit a non-finite factor that later sizes the
    # refined envelope arrays and corrupts the scatter.
    if not (math.isfinite(refined_grid_factor) and refined_grid_factor > 1.0):
        msg = (
            f"DCEGM.refined_grid_factor must be a finite value greater than 1.0, "
            f"got {refined_grid_factor}. It is the headroom factor sizing the "
            "refined envelope arrays; a value at or below 1.0 leaves no room "
            "for the constrained points and overflows the scatter."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_jump_thresh_non_positive(fues_jump_thresh: float) -> None:
    # `not (x > 0.0)` rejects NaN too: `nan <= 0.0` is False, so the segment-
    # switch comparison would silently misbehave on a non-finite threshold.
    if not (math.isfinite(fues_jump_thresh) and fues_jump_thresh > 0.0):
        msg = (
            f"DCEGM.fues_jump_thresh must be a finite positive value, got "
            f"{fues_jump_thresh}. It is the segment-switch threshold on "
            "`|ΔA / ΔR|` in the FUES scan."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_jump_thresh_non_positive(rfc_jump_thresh: float) -> None:
    # `not (x > 0.0)` rejects NaN too: `nan <= 0.0` is False, so the segment-
    # switch comparison would silently misbehave on a non-finite threshold.
    if not (math.isfinite(rfc_jump_thresh) and rfc_jump_thresh > 0.0):
        msg = (
            f"DCEGM.rfc_jump_thresh must be a finite positive value, got "
            f"{rfc_jump_thresh}. It is the segment-switch threshold on "
            "`|Δc / ΔR|` in the rooftop cut."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_search_radius_too_few(rfc_search_radius: int) -> None:
    if rfc_search_radius < 1:
        msg = (
            f"DCEGM.rfc_search_radius must be at least 1, got "
            f"{rfc_search_radius}. The rooftop-cut dominance test must inspect "
            "at least one neighbor on each side of a candidate."
        )
        raise RegimeInitializationError(msg)


def _fail_if_n_constrained_points_too_few(n_constrained_points: int) -> None:
    if n_constrained_points < 2:  # noqa: PLR2004
        msg = (
            f"DCEGM.n_constrained_points must be at least 2, got "
            f"{n_constrained_points}. The credit-constrained segment needs at "
            "least two closed-form points to interpolate between."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_n_points_to_scan_too_few(fues_n_points_to_scan: int | None) -> None:
    # `None` requests the exhaustive scan; only an explicit finite width is bounded.
    if fues_n_points_to_scan is not None and fues_n_points_to_scan < 1:
        msg = (
            f"DCEGM.fues_n_points_to_scan must be at least 1, got "
            f"{fues_n_points_to_scan}. The FUES forward scan must inspect at "
            "least one point after each candidate."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_scan_unroll_too_few(fues_scan_unroll: int) -> None:
    if fues_scan_unroll < 1:
        msg = (
            f"DCEGM.fues_scan_unroll must be at least 1, got "
            f"{fues_scan_unroll}. It is the `lax.scan` unroll factor for the "
            "FUES candidate scan; 1 means no unrolling."
        )
        raise RegimeInitializationError(msg)


def _fail_if_envelope_max_runs_too_few(envelope_max_runs: int) -> None:
    if envelope_max_runs < _MIN_ENVELOPE_MAX_RUNS:
        msg = (
            f"DCEGM.envelope_max_runs must be at least {_MIN_ENVELOPE_MAX_RUNS}, "
            f"got {envelope_max_runs}. It is the fold capacity of the exact "
            "upper envelope; a non-concave candidate chain folds into at "
            "least two resource-increasing runs."
        )
        raise RegimeInitializationError(msg)


def _fail_if_envelope_cell_batch_size_non_positive(
    envelope_cell_batch_size: int | None,
) -> None:
    if envelope_cell_batch_size is not None and envelope_cell_batch_size < 1:
        msg = (
            f"DCEGM.envelope_cell_batch_size must be at least 1, got "
            f"{envelope_cell_batch_size}. It is how many node cells the exact "
            "upper envelope resolves in parallel; use None to resolve them one "
            "at a time."
        )
        raise RegimeInitializationError(msg)


def _fail_if_stochastic_node_batch_size_negative(
    stochastic_node_batch_size: int,
) -> None:
    if stochastic_node_batch_size < 0:
        msg = (
            f"DCEGM.stochastic_node_batch_size must be non-negative, got "
            f"{stochastic_node_batch_size}. It is the block size for splaying the "
            "child stochastic-node expectation into `lax.map` blocks; 0 keeps the "
            "fused vmap."
        )
        raise RegimeInitializationError(msg)
