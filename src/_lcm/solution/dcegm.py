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
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.continuation_target import _union_fixed_params, _union_free_params
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    SolverModelContext,
)
from _lcm.typing import (
    EGMStepFunction,
    FlatParams,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.exceptions import (
    ExactAffineKernelUnavailableError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.typing import (
    ActionName,
    FloatND,
    FunctionName,
    StateName,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ExactEnvelope:
    """Certified segment-envelope backend configuration.

    The native exact-affine library must be available when a regime selects this
    backend. The backend is exact for the finite candidate set supplied to it;
    discretization of a continuous constrained branch remains controlled by
    `DCEGM.n_constrained_points`.
    """

    max_runs: int = 24
    """Maximum resource-increasing runs folded into one node cell."""

    cell_batch_size: int | None = None
    """Number of independent node cells resolved in parallel; `None` is serial."""

    def __post_init__(self) -> None:
        _fail_if_envelope_max_runs_too_few(self.max_runs)
        _fail_if_envelope_cell_batch_size_non_positive(self.cell_batch_size)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class FUESEnvelope:
    """Fast Upper-Envelope Scan configuration."""

    jump_thresh: float = 2.0
    """Segment-switch threshold on `|ΔA / ΔR|`."""

    n_points_to_scan: int | None = None
    """Forward-scan width; `None` performs the exhaustive scan."""

    scan_unroll: int = 1
    """Loop-unroll factor for the sequential candidate scan."""

    def __post_init__(self) -> None:
        _fail_if_fues_jump_thresh_non_positive(self.jump_thresh)
        _fail_if_fues_n_points_to_scan_too_few(self.n_points_to_scan)
        _fail_if_fues_scan_unroll_too_few(self.scan_unroll)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class RFCEnvelope:
    """Rooftop-Cut upper-envelope configuration."""

    jump_thresh: float = 2.0
    """Segment-switch threshold on `|Δc / ΔR|`."""

    search_radius: int = 10
    """Neighbors inspected on each side of a candidate."""

    def __post_init__(self) -> None:
        _fail_if_rfc_jump_thresh_non_positive(self.jump_thresh)
        _fail_if_rfc_search_radius_too_few(self.search_radius)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class LTMEnvelope:
    """Local-upper-bound brute-force envelope configuration."""


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class MSSEnvelope:
    """HARK-style left-to-right segment envelope configuration."""


type EnvelopeConfig = (
    ExactEnvelope | FUESEnvelope | RFCEnvelope | LTMEnvelope | MSSEnvelope
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
    `R` with consumption recovery `c = R - A`, and a post-decision function `A`.
    An analytical `inverse_marginal_utility` function is optional: when it is
    absent, pylcm differentiates `utility` and numerically inverts the marginal
    utility with an expanding bracket. Structural and per-period applicability
    checks run during `Model(...)` through the solver's staged validation hooks.

    A solve can publish the off-grid EGM policy as an inspection artifact when
    `return_simulation_policy=True`. Collection and host transfer are skipped
    otherwise, except when fresh simulation has an internal policy-read consumer.
    No shipped envelope currently satisfies the conservative off-grid read gate,
    so ordinary simulation recomputes the action on its grid.

    Otherwise — and whenever `simulate` is handed user-supplied value arrays,
    which carry no policy — `simulate` recomputes the argmax over the regime's
    gridded continuous action against the stored value function. Simulated
    continuous actions then live on the action grid, and with taste shocks the
    simulated choice frequencies follow the grid-restricted choice-specific
    values rather than exactly the solve's choice probabilities. On both paths
    the budget constraint the solve enforces intrinsically
    (`continuous_action <= resources - savings_grid lower bound`) is applied
    as a feasibility mask during simulation.

    """

    continuous_state: StateName = ""
    """Name of the Euler continuous state (e.g. `"wealth"`).

    Its transition must consume the post-decision function and reach the
    state and the continuous action only through it. May be omitted only when
    the solver is attached to a `ConsumptionSavingsRegime`, which owns the role.
    """

    continuous_action: ActionName = ""
    """Name of the continuous action (e.g. `"consumption"`).

    May be omitted only when the solver is attached to a
    `ConsumptionSavingsRegime`.
    """

    resources: FunctionName = ""
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

    post_decision_function: FunctionName = ""
    """Name of the post-decision function in `Regime.functions`.

    The end-of-period state (e.g. savings), satisfying
    `post_decision = resources - continuous_action`.
    """

    savings_grid: ContinuousGrid
    """Exogenous end-of-period grid; its lower bound is the borrowing limit.

    The endogenous grid inherits this grid's spacing, so it controls where the
    solution is accurate. Value reads use a slope-limited cubic Hermite
    interpolant when marginal-utility slopes are available; policy reads remain
    piecewise linear. Reads extrapolate along the nearest segment below support
    and use the endpoint above support. With sharply curved utility (e.g. CRRA),
    cluster nodes toward the borrowing limit: interpolation error there compounds
    across periods.
    """

    envelope: EnvelopeConfig = field(default_factory=ExactEnvelope)
    """Typed upper-envelope backend configuration.

    Backend-specific controls live on the selected frozen configuration object,
    so controls for an inactive algorithm cannot be supplied accidentally:

    - `ExactEnvelope`: certified ownership of the represented candidates. It
      requires pylcm's native exact-affine library and fails during `Model(...)`
      when that library is unavailable.
    - `FUESEnvelope`: Fast Upper-Envelope Scan.
    - `RFCEnvelope`: Rooftop-Cut algorithm.
    - `LTMEnvelope`: quadratic local-upper-bound baseline.
    - `MSSEnvelope`: HARK-style left-to-right segment sweep.

    `ExactEnvelope` certifies the envelope of the candidates actually supplied;
    it does not turn the sampled credit-constrained branch into a continuous
    exact representation.
    """

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
    """Resolution of the sampled credit-constrained branch.

    The current period evaluates the analytical constrained value as a floor,
    but the continuation carry handed to a parent contains only these sampled
    constrained candidates. Increasing the count can therefore improve values
    and policies in earlier periods at the cost of larger envelope rows, longer
    compilation, and more execution work. `ExactEnvelope` is exact only for this
    supplied finite candidate set, not for the unsampled continuous branch.
    """

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
        _fail_if_n_constrained_points_too_few(self.n_constrained_points)
        _fail_if_stochastic_node_batch_size_negative(self.stochastic_node_batch_size)

    @property
    def requires_continuation(self) -> bool:
        """DC-EGM inverts the Euler equation against its targets' marginals."""
        return True

    def validate_model(self, *, context: SolverModelContext) -> None:
        """Validate the user-level DC-EGM contract for this regime."""
        missing_roles = tuple(
            name
            for name, value in (
                ("continuous_state", self.continuous_state),
                ("continuous_action", self.continuous_action),
                ("resources", self.resources),
                ("post_decision_function", self.post_decision_function),
            )
            if not value
        )
        if missing_roles:
            msg = (
                f"DCEGM regime {context.regime_name!r} has no value for "
                f"{', '.join(missing_roles)}. Supply those fields on DCEGM, or "
                "attach the solver to ConsumptionSavingsRegime so the regime's "
                "canonical consumption-savings roles are bound automatically."
            )
            raise ModelInitializationError(msg)

        from _lcm.egm.validation import validate_dcegm_regime  # noqa: PLC0415

        validate_dcegm_regime(
            regime_name=context.regime_name,
            user_regime=context.user_regimes[context.regime_name],
            user_regimes=context.user_regimes,
            solution_reachability=context.solution_reachability,
        )

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Validate capabilities required by the selected envelope backend.

        Model semantics are checked earlier by :meth:`validate_model`, without
        consulting machine-local capabilities. Reaching this build-stage hook
        therefore means the regime satisfies the DC-EGM contract; only then is
        it meaningful to ask whether this installation can execute its selected
        backend.
        """
        if not isinstance(self.envelope, ExactEnvelope):
            return

        from _lcm.egm.upper_envelope._exact_affine import ffi  # noqa: PLC0415

        if not ffi.kernel_available_for_current_backend():
            msg = (
                f"Regime {context.regime_name!r} selects ExactEnvelope, but "
                "the native exact-affine library is unavailable or unloadable. "
                "Install a pylcm build carrying the library or build it with "
                "`pixi run build-exact-affine`; select another envelope only "
                "when its documented approximation contract is acceptable."
            )
            raise ExactAffineKernelUnavailableError(msg)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one DC-EGM period adapter per period and the carry template.

        The solver's model-stage and build-stage validation hooks guarantee the
        regime is non-terminal, so the regime-transition probability function
        exists. Periods sharing one EGM-step core reuse a single jitted core, and
        therefore a single compiled program. Numerical-builder imports are
        function-local so the public `lcm.solvers` façade stays a thin re-export
        that pulls in no engine modules.
        """

        from _lcm.egm.step import build_egm_step_functions  # noqa: PLC0415

        assert context.compute_regime_transition_probs is not None  # noqa: S101
        assert context.koopmans_aggregator is not None  # noqa: S101
        egm_step, egm_carry_template, egm_stateful_targets = build_egm_step_functions(
            solver=self,
            regime_name=context.regime_name,
            user_regimes=context.user_regimes,
            functions=context.functions,
            koopmans_aggregator=context.koopmans_aggregator,
            constraints=context.constraints,
            transitions=context.transitions,
            transition_laws=context.transition_laws,
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
                    stateful_targets=egm_stateful_targets,
                    transition_target_names=tuple(context.transitions),
                )
                for period, core in egm_step.items()
            }
        )
        return SolutionKernels(
            period_kernels=period_kernels,
            continuation_spec=EGMContinuationSpec(
                template=egm_carry_template,
                layout=self.egm_continuation_layout,
            ),
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

    stateful_targets: frozenset[RegimeName]
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
        egm_fixed = _union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
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
                stateful_targets=self.stateful_targets,
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
                stateful_targets=self.stateful_targets,
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
        return _union_free_params(
            flat_params=flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )


def _carry_subset(
    *,
    next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
    stateful_targets: frozenset[RegimeName],
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
            if name in stateful_targets
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
            f"FUESEnvelope.jump_thresh must be a finite positive value, got "
            f"{fues_jump_thresh}. It is the segment-switch threshold on "
            "`|ΔA / ΔR|` in the FUES scan."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_jump_thresh_non_positive(rfc_jump_thresh: float) -> None:
    # `not (x > 0.0)` rejects NaN too: `nan <= 0.0` is False, so the segment-
    # switch comparison would silently misbehave on a non-finite threshold.
    if not (math.isfinite(rfc_jump_thresh) and rfc_jump_thresh > 0.0):
        msg = (
            f"RFCEnvelope.jump_thresh must be a finite positive value, got "
            f"{rfc_jump_thresh}. It is the segment-switch threshold on "
            "`|Δc / ΔR|` in the rooftop cut."
        )
        raise RegimeInitializationError(msg)


def _fail_if_rfc_search_radius_too_few(rfc_search_radius: int) -> None:
    if rfc_search_radius < 1:
        msg = (
            f"RFCEnvelope.search_radius must be at least 1, got "
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
            f"FUESEnvelope.n_points_to_scan must be at least 1, got "
            f"{fues_n_points_to_scan}. The FUES forward scan must inspect at "
            "least one point after each candidate."
        )
        raise RegimeInitializationError(msg)


def _fail_if_fues_scan_unroll_too_few(fues_scan_unroll: int) -> None:
    if fues_scan_unroll < 1:
        msg = (
            f"FUESEnvelope.scan_unroll must be at least 1, got "
            f"{fues_scan_unroll}. It is the `lax.scan` unroll factor for the "
            "FUES candidate scan; 1 means no unrolling."
        )
        raise RegimeInitializationError(msg)


def _fail_if_envelope_max_runs_too_few(envelope_max_runs: int) -> None:
    if envelope_max_runs < _MIN_ENVELOPE_MAX_RUNS:
        msg = (
            f"ExactEnvelope.max_runs must be at least {_MIN_ENVELOPE_MAX_RUNS}, "
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
            f"ExactEnvelope.cell_batch_size must be at least 1, got "
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
