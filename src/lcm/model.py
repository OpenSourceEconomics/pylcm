"""Collection of classes that are used by the user to define the model and grids."""

import logging
import operator
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

import jax
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.roar import BeartypeCallHintViolation

from _lcm.beartype_conf import MODEL_CONF, PARAMS_CONF
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import EGMPolicyRead, NNBEGMPolicyRead
from _lcm.grids import DiscreteGrid
from _lcm.model_processing import (
    _validate_param_types,
    build_regimes_and_template,
    fail_if_nonpositive_taste_shock_scale,
    validate_model_inputs,
)
from _lcm.pandas_utils import (
    convert_series_in_params,
    has_series,
    initial_conditions_from_dataframe,
)
from _lcm.params.processing import (
    broadcast_to_template,
    cast_params_to_canonical_dtypes,
    materialize_granular_transition_params,
)
from _lcm.persistence.snapshots import (
    _save_simulate_snapshot,
    _save_solve_snapshot,
)
from _lcm.reachability import ModelReachability
from _lcm.regime_building.broadcast import (
    merge_model_slots,
    prune_broadcast_variables,
    validate_model_slots,
)
from _lcm.regime_building.finalize import (
    FinalizedUserRegime,
    finalize_regimes,
)
from _lcm.regime_building.fixed_process_laws import bind_fixed_process_laws
from _lcm.regime_building.processing import (
    Regime,
    compute_active_periods_by_regime,
    prepare_model_structure,
)
from _lcm.simulation.compile import compile_all_simulation_phases
from _lcm.simulation.initial_conditions import (
    canonicalize_initial_conditions,
    pad_initial_conditions_to_multiple,
    validate_initial_conditions,
)
from _lcm.simulation.result_metadata import _get_output_dtypes
from _lcm.simulation.simulate import simulate
from _lcm.solution.artifacts import (
    build_solution_result,
    fingerprint_flat_params,
)
from _lcm.solution.backward_induction import (
    _build_base_state_action_spaces,
    _reject_edge_fold_state_param_collisions,
    solve,
)
from _lcm.solution.contract import BackwardInductionResult
from _lcm.solution.fingerprint import fingerprint_model, project_solution_params
from _lcm.solution.model_authority import (
    ReplayCellDescriptor,
    SolutionAuthority,
    _replay_model_context_from_state_action_space,
    _state_action_space_for_period,
    bind_generated_solution_authority,
    build_solution_authority,
    snapshot_solution_authority,
)
from _lcm.solution.preconditions import (
    check_pareto_weights,
    check_solver_params,
)
from _lcm.solution.replay_validation import (
    validate_egm_sim_policy,
    validate_nested_egm_sim_policy,
    validate_nnbegm_sim_policy,
)
from _lcm.solution.result_snapshot import (
    snapshot_artifact_store,
    snapshot_artifact_template_declaration,
    snapshot_omissions,
    snapshot_solution_metadata,
    snapshot_value_store,
)
from _lcm.solution.validate_V import contains_nan
from _lcm.transition_checks import validate_transitions
from _lcm.typing import (
    FlatParams,
    FunctionName,
    ParamsTemplate,
    PeriodToRegimeToDissolutionFlags,
    PeriodToRegimeToSimulationPolicy,
    PeriodToRegimeToVArr,
    RegimeName,
    RegimeNamesToIds,
)
from _lcm.utils.containers import (
    ensure_containers_are_immutable,
    ensure_containers_are_mutable,
    get_field_names_and_values,
)
from _lcm.utils.logging import (
    LogLevel,
    get_logger,
    raise_or_warn,
    validation_enabled,
    validation_raises,
)
from lcm.ages import AgeGrid
from lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from lcm.exceptions import (
    InvalidInitialConditionsError,
    InvalidSimulationInputError,
    InvalidValueFunctionError,
    UnsupportedOperationError,
)
from lcm.koopmans_aggregation import LinearAggregator
from lcm.regime import Regime as UserRegime
from lcm.result import SimulationResult
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    PYLCM_VERSION,
    SIMULATION_POLICY,
    SOLUTION_SCHEMA_VERSION,
    SOLVER_API_VERSION,
    SOLVER_DIAGNOSTICS,
    ArtifactAuthority,
    ArtifactChannel,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    OmissionReason,
    PersistencePolicy,
    ReplayMode,
    ReplayReader,
    ReplayRouteRequirements,
    ReplayRouteSnapshot,
    ResultRetention,
    SimulationBuildContext,
    SolutionMetadata,
    SolutionResult,
    SolutionSource,
    ValueArraySchema,
    ValueStore,
    _canonicalize_artifact_payload,
    _replay_route_identity,
    _same_exact_artifact_contract,
)
from lcm.typing import (
    UserFacingParamsTemplate,
    UserFunction,
    UserInitialConditions,
    UserParams,
)

if TYPE_CHECKING:
    _SolutionResultBoundary: TypeAlias = SolutionResult  # noqa: UP040
    _ArtifactStoreBoundary: TypeAlias = ArtifactStore  # noqa: UP040
else:
    # Caller-visible result containers are validated and snapshotted explicitly;
    # runtime annotation traversal must not inspect hostile or lazy contents first.
    _SolutionResultBoundary = object
    _ArtifactStoreBoundary = object


def _same_exactly_typed(*, actual: object, expected: object) -> bool:
    """Compare trusted metadata without admitting equal values of another type."""
    return _same_exact_artifact_contract(
        actual=actual,
        expected=expected,
    )


type _PeriodToRegimeToReplayReader = MappingProxyType[
    int, MappingProxyType[RegimeName, ReplayReader]
]


def _built_in_policy_payload_defect(  # noqa: PLR0911
    *,
    supplied: object,
    descriptor: ReplayCellDescriptor,
    period: int,
) -> str | None:
    """Return a model-authority defect for one built-in simulation policy."""
    policy_read = descriptor.route
    if not isinstance(policy_read, EGMPolicyRead | NNBEGMPolicyRead):
        return None
    if descriptor.payload_type is None or type(supplied) is not descriptor.payload_type:
        expected_name = getattr(
            descriptor.payload_type,
            "__name__",
            repr(descriptor.payload_type),
        )
        return (
            f"expected exact payload type {expected_name}, got "
            f"{type(supplied).__name__}"
        )
    if isinstance(policy_read, EGMPolicyRead):
        if not isinstance(supplied, EGMSimPolicy) or descriptor.egm_node_count is None:
            return "model authority lacks the EGM node count"
        return validate_egm_sim_policy(
            policy=supplied,
            policy_read=policy_read,
            period=period,
            expected_node_count=descriptor.egm_node_count,
        )
    if policy_read.replay_policy_is_nested:
        if (
            not isinstance(supplied, NestedEGMSimPolicy)
            or descriptor.egm_node_count is None
            or descriptor.adaptive_outer_nodes is None
            or descriptor.expected_replay_capability is None
        ):
            return "model authority lacks a nested replay descriptor"
        return validate_nested_egm_sim_policy(
            policy=supplied,
            policy_read=policy_read,
            period=period,
            expected_node_count=descriptor.egm_node_count,
            expected_outer_nodes=descriptor.adaptive_outer_nodes,
            expected_replay_capability=descriptor.expected_replay_capability,
        )
    if (
        not isinstance(supplied, NNBEGMSimPolicy)
        or descriptor.expected_replay_capability is None
    ):
        return "model authority lacks a finite replay descriptor"
    return validate_nnbegm_sim_policy(
        policy=supplied,
        policy_read=policy_read,
        period=period,
        expected_replay_capability=descriptor.expected_replay_capability,
    )


def _materialize_artifact_projection(
    *,
    store: _ArtifactStoreBoundary,
    key: ArtifactKey,
    authority: SolutionAuthority,
    required_only: bool = False,
) -> MappingProxyType[int, MappingProxyType[RegimeName, object]]:
    """Materialize one consumed replay projection into an immutable snapshot."""
    projected: dict[int, dict[RegimeName, object]] = {}
    for ref in store:
        if ref.key != key:
            continue
        artifact_authority = authority.artifacts.get(ref)
        if artifact_authority is None:
            raise InvalidSimulationInputError(
                f"Artifact {ref!r} has no model-built materialization authority."
            )
        if required_only and not artifact_authority.required:
            continue
        try:
            materialized = store._materialize_from_template_snapshot(  # noqa: SLF001
                ref,
                template_snapshot=snapshot_artifact_template_declaration(
                    artifact_authority
                ),
            )
            if key == SIMULATION_POLICY:
                payload_defect = _built_in_policy_payload_defect(
                    supplied=materialized,
                    descriptor=authority.replay[ref],
                    period=ref.period,
                )
                if payload_defect is not None:
                    raise InvalidSimulationInputError(
                        f"Artifact {ref!r} mismatched_payload: {payload_defect}"
                    )
            canonical = _canonicalize_artifact_payload(
                payload=materialized,
                authority=artifact_authority,
            )
        except (TypeError, ValueError) as error:
            raise InvalidSimulationInputError(
                f"Artifact {ref!r} mismatched_payload: cannot be canonicalized: {error}"
            ) from error
        projected.setdefault(ref.period, {})[ref.regime] = canonical
    return MappingProxyType(
        {
            period: MappingProxyType(regime_to_payload)
            for period, regime_to_payload in sorted(projected.items())
        }
    )


class Model:
    """A model which is created from a regime.

    Upon initialization, internal regimes will be created which contain all
    the functions needed to solve and simulate the model.

    """

    description: str | None = None
    """Description of the model."""

    ages: AgeGrid
    """Age grid for the model."""

    n_periods: int
    """Number of periods in the model."""

    regime_names_to_ids: RegimeNamesToIds
    """Immutable mapping from regime names to integer indices."""

    stakeholder_names_to_ids: MappingProxyType[str, int]
    """Immutable mapping from stakeholder names to integer role codes.

    One vocabulary for the whole model, covering every collective regime's
    stakeholders, so a role means the same thing wherever it is read. Empty for
    a model with no collective regime. This is the vocabulary
    `initial_conditions["own_stakeholder"]` is written in, and the one the
    published `own_stakeholder` column is labelled from."""

    user_regimes: MappingProxyType[RegimeName, FinalizedUserRegime]
    """The finalized regimes: plain `lcm.regime.Regime` instances, complete
    (Koopmans aggregator injected, completeness validated), with model-level slots
    merged in and broadcast variables pruned, still in user vocabulary."""

    pruned_variables: MappingProxyType[RegimeName, frozenset[str]]
    """Per regime, the broadcast states and actions pruned because no root
    computation of either phase reads them (directly or through a law of
    motion toward a reachable target that keeps them)."""

    reachability: ModelReachability
    """Static solution and simulation regime graphs."""

    _regimes: MappingProxyType[RegimeName, Regime]
    """Canonical, processed regimes used by solve and simulate.

    Private: the canonical form is engine-internal. User code should read
    `user_regimes` (the boundary form supplied to the constructor).
    """

    enable_jit: bool = True
    """Whether to JIT-compile the functions of the internal regimes."""

    fixed_params: UserParams
    """Parameters fixed at model initialization."""

    n_subjects: int | None = None
    """Expected simulate population size; enables AOT compile of simulate functions.

    Dispatch by call shape:

    - `None`: purely lazy behaviour, no AOT.
    - First `simulate(...)` with `actual_n == n_subjects`: AOT-compiles all
      simulate functions for the chunk shape (`subject_batch_size`, clamped to
      the population, or the whole population when unbatched), blocking before
      solve runs, and caches them.
    - Subsequent `simulate(...)` with the same population and chunk shape:
      reuses the cached compiled programs.
    - `simulate(...)` with a mismatching population size: warns once per size
      and falls back to the runtime-traced path.

    Param-shape contract: the cache is keyed on the chunk shape. The shapes
    and dtypes of `flat_params` leaves at the first matching call become
    part of the AOT signature; subsequent calls must keep them stable. MSM-
    style estimation (varying values, fixed shapes) is the target use case;
    construct a fresh `Model` whenever a param array's shape or dtype changes.
    """

    _params_template: ParamsTemplate
    """Template for the model parameters."""

    _simulate_compile_cache: dict[int, MappingProxyType[RegimeName, Regime]]
    """AOT-compiled `regimes` keyed by chunk shape (`subject_batch_size`, or the
    full population when unbatched)."""

    _warned_n_subjects: set[int]
    """Mismatching `actual_n_subjects` already warned about (one warning each)."""

    _simulate_compile_lock: threading.Lock
    """Serialises mutations of `_simulate_compile_cache` and
    `_warned_n_subjects`.

    The check-then-set on each container is held under this lock. The
    consequent `log.warning` call sits outside the lock so concurrent
    simulate() calls don't serialise on logging I/O.
    """

    @beartype(conf=MODEL_CONF)
    def __init__(
        self,
        *,
        description: str = "",
        ages: AgeGrid,
        regimes: Mapping[RegimeName, UserRegime],
        regime_id_class: type,
        enable_jit: bool = True,
        fixed_params: UserParams = MappingProxyType({}),
        derived_categoricals: Mapping[FunctionName, DiscreteGrid] = MappingProxyType(
            {}
        ),
        functions: Mapping[str, object] = MappingProxyType({}),
        constraints: Mapping[str, object] = MappingProxyType({}),
        states: Mapping[str, object] = MappingProxyType({}),
        state_transitions: Mapping[str, object] = MappingProxyType({}),
        actions: Mapping[str, object] = MappingProxyType({}),
        koopmans_aggregator: UserFunction = LinearAggregator(),
        certainty_equivalent: CertaintyEquivalent = LinearExpectation(),
        n_subjects: int | None = None,
    ) -> None:
        """Initialize the Model.

        Args:
            regimes: Mapping of regime names to user-provided `Regime`
                instances. Stored as `self.user_regimes` after merging in
                any model-level `derived_categoricals`; the canonical
                processed form is exposed as `self._regimes`.
            ages: Age grid for the model.
            description: Description of the model.
            regime_id_class: Dataclass mapping regime names to integer indices.
            enable_jit: Whether to JIT-compile the functions of the internal
                regimes.
            fixed_params: Parameters that can be fixed at model initialization.
            derived_categoricals: Categorical grids for DAG function outputs
                not in states/actions. Broadcast to all regimes (merged with
                each regime's own `derived_categoricals`). Raises if a regime
                already has a conflicting entry.
            functions: Model-level functions, merged into every regime under
                the exactly-one-level rule (a name is defined at model level
                or regime level, never both; a regime-level `None` masks the
                model entry).
            constraints: Model-level constraints; same merge rule.
            states: Model-level states; same merge rule. Broadcast states are
                pruned per regime by DAG reachability (see
                `pruned_variables`). `distributed=True` is legal only here.
            state_transitions: Model-level laws of motion; same merge rule.
            actions: Model-level actions; same merge rule and pruning.
            koopmans_aggregator: How every non-terminal regime combines current
                utility with the certainty equivalent into `Q`. Same
                all-or-nothing rule as `certainty_equivalent` below; terminal
                regimes never receive it.
            certainty_equivalent: How every non-terminal regime aggregates its
                continuation lottery. Unlike the mapping slots above this is a
                single value, so the rule is all-or-nothing rather than
                per-name: declare it here, or in every regime that has a
                continuation, never some of each. Terminal regimes never
                receive it.
            n_subjects: Expected simulate batch size; if set, the first matching
                `simulate(...)` call AOT-compiles all simulate functions for
                batch shape `n_subjects` before backward induction starts.
                `None` keeps the purely lazy behaviour.

        """
        self.description = description
        self.ages = ages
        self.n_periods = ages.n_periods
        self.fixed_params = ensure_containers_are_immutable(fixed_params)
        self.n_subjects = n_subjects
        self._simulate_compile_cache = {}
        self._warned_n_subjects = set()
        self._simulate_compile_lock = threading.Lock()
        # In-memory result provenance. Kept in pickle state so a model and a
        # result round-tripped together remain compatible, but deliberately not
        # presented as a durable model-content fingerprint.
        self._solution_model_instance_id = uuid.uuid4().hex
        self._solution_authorities: dict[str, SolutionAuthority] = {}

        # The single canonical activity schedule: every regime's `active`
        # predicate is evaluated exactly once, here, and threaded through
        # pruning, validation, and model-structure preparation below. Its
        # `.active` predicate is unaffected by slot merging/finalization, so
        # the raw `regimes` argument is the correct — and only — evaluation
        # point.
        active_periods_by_regime = compute_active_periods_by_regime(
            ages=ages, user_regimes=regimes
        )

        model_slots = {
            "functions": functions,
            "constraints": constraints,
            "states": states,
            "state_transitions": state_transitions,
            "actions": actions,
        }
        validate_model_slots(model_slots=model_slots)
        merged_regimes, broadcast_variables = merge_model_slots(
            user_regimes=regimes,
            model_slots=model_slots,
        )
        pruned_regimes, self.pruned_variables = prune_broadcast_variables(
            user_regimes=merged_regimes,
            broadcast_variables=broadcast_variables,
            koopmans_aggregator=koopmans_aggregator,
            ages=ages,
            active_periods_by_regime=active_periods_by_regime,
        )
        finalized_regimes = finalize_regimes(
            user_regimes=pruned_regimes,
            derived_categoricals=derived_categoricals,
            koopmans_aggregator=koopmans_aggregator,
            certainty_equivalent=certainty_equivalent,
        )
        # A process law named in `fixed_params` means exactly what the same
        # value passed to the process constructor means, so it is bound into
        # the grid here — before validation, structure preparation, and
        # entry-law synthesis, all of which ask whether a process's law is
        # known. What no process could take stays a runtime parameter and
        # reaches `build_regimes_and_template` unchanged.
        (
            self.user_regimes,
            residual_fixed_params,
            params_consumed_by_binder,
        ) = bind_fixed_process_laws(
            user_regimes=finalized_regimes,
            fixed_params=self.fixed_params,
        )
        validate_model_inputs(
            n_periods=self.n_periods,
            user_regimes=self.user_regimes,
            regime_id_class=regime_id_class,
            n_subjects=n_subjects,
            broadcast_variables=broadcast_variables,
            ages=self.ages,
            active_periods_by_regime=active_periods_by_regime,
        )
        self.regime_names_to_ids = MappingProxyType(
            dict(
                sorted(
                    get_field_names_and_values(regime_id_class).items(),
                    key=operator.itemgetter(1),
                )
            )
        )
        prepared_structure = prepare_model_structure(
            user_regimes=self.user_regimes,
            ages=self.ages,
            active_periods_by_regime=active_periods_by_regime,
        )
        self.reachability = prepared_structure.reachability
        self._regimes, self._params_template = build_regimes_and_template(
            ages=self.ages,
            user_regimes=self.user_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=residual_fixed_params,
            params_already_consumed=params_consumed_by_binder,
            prepared_structure=prepared_structure,
        )
        self.stakeholder_names_to_ids = next(
            (regime.stakeholder_names_to_ids for regime in self._regimes.values()),
            MappingProxyType({}),
        )
        self.enable_jit = enable_jit
        self.simulation_output_dtypes = _get_output_dtypes(
            user_regimes=self.user_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
        )

    def __repr__(self) -> str:
        """Summarize the model; mention pruning when any regime was pruned."""
        n_pruned = sum(1 for names in self.pruned_variables.values() if names)
        pruned_part = (
            f", {n_pruned} regimes with pruned variables (see `.pruned_variables`)"
            if n_pruned
            else ""
        )
        return (
            f"Model(n_regimes={len(self.user_regimes)}, "
            f"n_periods={self.n_periods}{pruned_part})"
        )

    def __getstate__(self) -> dict[str, object]:
        """Return a copy of `__dict__` with per-process AOT compile state removed.

        Drops `_simulate_compile_lock` (a `threading.Lock`, not pickleable),
        `_simulate_compile_cache` (compiled XLA programs that can't survive
        a process boundary), and `_warned_n_subjects` (its companion set).
        `__setstate__` restores all three to their fresh state.
        """
        state = self.__dict__.copy()
        state.pop("_simulate_compile_lock", None)
        state.pop("_simulate_compile_cache", None)
        state.pop("_warned_n_subjects", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore transient state and backfill legacy solution identity."""
        self.__dict__.update(state)
        if "_solution_model_instance_id" not in state:
            self._solution_model_instance_id = uuid.uuid4().hex
        if "_solution_authorities" not in state:
            self._solution_authorities = {}
        self._simulate_compile_cache = {}
        self._warned_n_subjects = set()
        self._simulate_compile_lock = threading.Lock()

    def get_params_template(self) -> UserFacingParamsTemplate:
        """Get a human-readable params template.

        Return a nested dict showing which parameters each function in each
        regime expects.

        """
        mutable = ensure_containers_are_mutable(self._params_template)
        return cast("UserFacingParamsTemplate", _readable_template(mutable))

    @beartype(conf=PARAMS_CONF)
    def solve(
        self,
        *,
        params: UserParams,
        log_level: LogLevel,
        retention: ResultRetention = ResultRetention.VALUES_AND_REPLAY,
        max_compilation_workers: int | None = None,
        log_path: str | Path | None = None,
        log_keep_n_latest: int = 3,
    ) -> SolutionResult:
        """Solve the model into a labelled, model-authoritative result.

        The default keeps replay artifacts so every built-in solver decision can be
        replayed by ``simulate(solution=result)``. ``retention`` affects only
        artifacts kept after the solve; continuations required during backward
        induction are always produced and consumed. Solver diagnostics remain
        governed solely by ``log_level``.

        An in-memory result is bound to this model instance and the exact canonical
        parameter values used here. Metadata also carries a durable model fingerprint;
        after save/load, an equivalent fresh model validates that fingerprint instead
        of the producing instance token.

        Args:
            params: Model parameters compatible with ``get_params_template()``.
            log_level: Verbosity and runtime-validation policy.
            retention: Post-solve artifacts to retain.
            max_compilation_workers: Maximum threads for parallel XLA compilation.
            log_path: Optional directory for diagnostic snapshots.
            log_keep_n_latest: Maximum snapshots to retain on disk.

        Returns:
            An immutable labelled result containing values, metadata, retained replay
            and diagnostic artifacts, plus explicit artifact-omission reasons.
        """
        log = get_logger(log_level=log_level)
        flat_params = self._process_params(params)
        validate_transitions(
            regimes=self._regimes,
            flat_params=flat_params,
            ages=self.ages,
            logger=log,
        )
        return self._solve_from_flat_params(
            flat_params=flat_params,
            params=params,
            log=log,
            retention=retention,
            max_compilation_workers=max_compilation_workers,
            log_path=log_path,
            log_keep_n_latest=log_keep_n_latest,
        )

    def _solve_from_flat_params(
        self,
        *,
        flat_params: FlatParams,
        params: UserParams,
        log: logging.Logger,
        retention: ResultRetention,
        max_compilation_workers: int | None,
        log_path: str | Path | None,
        log_keep_n_latest: int,
    ) -> SolutionResult:
        """Build the canonical public result from processed parameters.

        The solution authority is derived from the model and the canonical
        parameters before the solve; the solve's generated replay facts are
        bound into it afterwards.
        """
        declared_authority = build_solution_authority(
            regimes=self._regimes,
            flat_params=flat_params,
            ages=self.ages,
        )
        retain_all_persistable = retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
        persistable_artifact_refs = (
            frozenset(
                ref
                for ref, artifact_authority in declared_authority.artifacts.items()
                if artifact_authority.applicable
                and artifact_authority.descriptor.persistence
                is PersistencePolicy.MODEL_VERIFIABLE
            )
            if retain_all_persistable
            else frozenset()
        )
        internal_result = self._solve_compiled(
            flat_params=flat_params,
            params=params,
            log=log,
            log_path=log_path,
            log_keep_n_latest=log_keep_n_latest,
            max_compilation_workers=max_compilation_workers,
            retain_dissolution_flags=retention.retains_replay,
            retain_replay=retention is ResultRetention.VALUES_AND_REPLAY,
            retain_all_artifacts=retain_all_persistable,
            persistable_artifact_refs=persistable_artifact_refs,
            collect_solver_diagnostics=True,
        )
        params_fingerprint = fingerprint_flat_params(
            project_solution_params(flat_params=flat_params, regimes=self._regimes)
        )
        model_fingerprint = fingerprint_model(
            ages=self.ages,
            regimes=self._regimes,
            user_regimes=self.user_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            flat_params=flat_params,
        )
        authority = bind_generated_solution_authority(
            authority=declared_authority,
            internal_result=internal_result,
            regimes=self._regimes,
            flat_params=flat_params,
        )
        if retention.retains_replay and any(
            descriptor.adaptive_outer_nodes is not None
            for descriptor in authority.replay.values()
        ):
            self._solution_authorities[params_fingerprint] = (
                snapshot_solution_authority(authority)
            )
        return build_solution_result(
            internal_result=internal_result,
            retention=retention,
            regimes=self._regimes,
            user_regimes=self.user_regimes,
            n_periods=self.n_periods,
            model_instance_id=self._solution_model_instance_id,
            params_fingerprint=params_fingerprint,
            model_fingerprint=model_fingerprint,
            authority=authority,
        )

    def _solve_compiled(
        self,
        *,
        flat_params: FlatParams,
        params: UserParams,
        log: logging.Logger,
        log_path: str | Path | None,
        log_keep_n_latest: int,
        max_compilation_workers: int | None,
        retain_dissolution_flags: bool = False,
        retain_replay: bool = True,
        retain_all_artifacts: bool = False,
        persistable_artifact_refs: frozenset[ArtifactRef] = frozenset(),
        collect_solver_diagnostics: bool = False,
    ) -> BackwardInductionResult:
        """Run backward induction, persisting a diagnostic snapshot when warranted.

        Returns the named backward-induction outputs: value-function arrays,
        each regime's published per-period simulation policy, and the
        per-period, per-COLLECTIVE-regime dissolution-flag arrays. Simulation
        policies are retained only when `retain_replay` is true, and only for
        regimes whose declared simulation route reads one.
        The dissolution flags are empty for models without collective regimes,
        and for a collective model whose gates never read `D_target` unless
        `retain_dissolution_flags` asks for them. With `log_path` set, a
        snapshot is written at `log_level="debug"` (every solve) and at
        `"warning"` / `"progress"` whenever the returned solution contains
        NaN. `_enforce_retention` caps the snapshot count at
        `log_keep_n_latest`.
        """
        check_solver_params(regimes=self._regimes, flat_params=flat_params)
        check_pareto_weights(
            regimes=self._regimes, flat_params=flat_params, ages=self.ages
        )
        try:
            internal_result = solve(
                flat_params=flat_params,
                ages=self.ages,
                regimes=self._regimes,
                logger=log,
                enable_jit=self.enable_jit,
                collect_solver_diagnostics=collect_solver_diagnostics,
                max_compilation_workers=max_compilation_workers,
                retain_dissolution_flags=retain_dissolution_flags,
                retain_replay=retain_replay,
                retain_all_artifacts=retain_all_artifacts,
                persistable_artifact_refs=persistable_artifact_refs,
            )
        except InvalidValueFunctionError as exc:
            if log_path is not None and exc.partial_solution is not None:
                snap_dir = _save_solve_snapshot(
                    model=self,
                    params=params,
                    period_to_regime_to_V_arr=exc.partial_solution,  # ty: ignore[invalid-argument-type]
                    log_path=Path(log_path),
                    log_keep_n_latest=log_keep_n_latest,
                )
                exc.add_note(f"Snapshot saved to {snap_dir}")
            raise
        if (
            log_path is not None
            and validation_enabled(log)
            and (
                validation_raises(log) or contains_nan(internal_result.value_functions)
            )
        ):
            _save_solve_snapshot(
                model=self,
                params=params,
                period_to_regime_to_V_arr=internal_result.value_functions,
                log_path=Path(log_path),
                log_keep_n_latest=log_keep_n_latest,
            )
        return internal_result

    def _resolve_simulate_regimes(
        self,
        *,
        actual_n_subjects: int,
        compile_batch_size: int,
        log: logging.Logger,
    ) -> MappingProxyType[RegimeName, Regime]:
        """Return regimes to use for simulate; AOT cache when matching.

        Dispatch by `n_subjects` and batch-shape match:

        - `n_subjects is None`: return the original `regimes`
          (purely lazy path).
        - `actual_n_subjects != n_subjects`: warn once per mismatching size,
          return the original `regimes`.
        - `actual_n_subjects == n_subjects`: return the regimes compiled for
          `compile_batch_size` (the chunk shape; caller must have populated the
          cache before calling).
        """
        if self.n_subjects is None:
            return self._regimes
        if actual_n_subjects != self.n_subjects:
            with self._simulate_compile_lock:
                already_warned = actual_n_subjects in self._warned_n_subjects
                if not already_warned:
                    self._warned_n_subjects.add(actual_n_subjects)
            if not already_warned:
                log.warning(
                    "simulate called with n_subjects=%d but model declared "
                    "n_subjects=%d; falling back to runtime compile.",
                    actual_n_subjects,
                    self.n_subjects,
                )
            return self._regimes
        with self._simulate_compile_lock:
            return self._simulate_compile_cache[compile_batch_size]

    def _resolve_solution_result(
        self, *, solution: _SolutionResultBoundary, flat_params: FlatParams
    ) -> tuple[
        PeriodToRegimeToVArr,
        PeriodToRegimeToSimulationPolicy,
        PeriodToRegimeToDissolutionFlags,
        _PeriodToRegimeToReplayReader,
    ]:
        """Resolve one model-authoritative result into engine replay inputs."""
        solution = self._snapshot_solution_envelope(solution=solution)
        metadata = solution.metadata
        authority, values, solution = self._check_solution_result_structure(
            solution=solution,
            metadata=metadata,
            flat_params=flat_params,
        )
        policies, dissolution_flags = self._check_solution_result_artifacts(
            solution=solution,
            authority=authority,
            values=values,
        )
        external_readers = self._build_external_replay_readers(
            solution=solution,
            metadata=metadata,
            authority=authority,
            flat_params=flat_params,
        )
        return (
            values,
            policies,
            dissolution_flags,
            external_readers,
        )

    def _check_solution_result_structure(
        self,
        *,
        solution: _SolutionResultBoundary,
        metadata: SolutionMetadata,
        flat_params: FlatParams,
    ) -> tuple[
        SolutionAuthority,
        PeriodToRegimeToVArr,
        _SolutionResultBoundary,
    ]:
        """Validate outer structure, then create the one canonical payload snapshot."""
        expected_fingerprint = fingerprint_flat_params(
            project_solution_params(flat_params=flat_params, regimes=self._regimes)
        )
        expected_model_fingerprint = fingerprint_model(
            ages=self.ages,
            regimes=self._regimes,
            user_regimes=self.user_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            flat_params=flat_params,
        )
        self._check_solution_result_metadata(
            metadata=metadata,
            expected_fingerprint=expected_fingerprint,
            expected_model_fingerprint=expected_model_fingerprint,
        )
        source_authority = self._solution_authorities.get(expected_fingerprint)
        if source_authority is None:
            source_authority = build_solution_authority(
                regimes=self._regimes,
                flat_params=flat_params,
                ages=self.ages,
            )
        try:
            authority = snapshot_solution_authority(source_authority)
        except (BeartypeCallHintViolation, TypeError, ValueError) as error:
            raise InvalidSimulationInputError(
                f"Model solution authority cannot be snapshotted: {error}"
            ) from error
        expected_descriptors = dict(authority.artifact_descriptors)
        if not _same_exact_artifact_contract(
            actual=metadata.artifact_descriptors,
            expected=expected_descriptors,
        ):
            raise InvalidSimulationInputError(
                "SolutionResult artifact descriptors differ from model authority."
            )
        expected_coverage = set(authority.values)
        value_store = solution.values  # noqa: PD011
        expected_periods = {period for period, _regime_name in expected_coverage}
        actual_periods = set(value_store)
        if actual_periods != expected_periods:
            missing = tuple(sorted(expected_periods - actual_periods))
            unexpected = tuple(sorted(actual_periods - expected_periods))
            msg = (
                "SolutionResult value period coverage is incompatible with this "
                f"model: missing={missing}, unexpected={unexpected}."
            )
            raise InvalidSimulationInputError(msg)
        actual_coverage = {
            (period, regime_name)
            for period, regime_to_value in value_store.items()
            for regime_name in regime_to_value
        }
        self._check_solution_result_coverage(
            actual_coverage=actual_coverage,
            expected_coverage=expected_coverage,
            label="value",
        )
        self._check_solution_result_coverage(
            actual_coverage=set(metadata.value_schemas),
            expected_coverage=expected_coverage,
            label="value schema",
        )
        self._check_solution_result_artifact_coordinates(
            solution=solution,
            metadata=metadata,
            expected_coverage=expected_coverage,
            authority=authority,
        )
        self._check_solution_result_present_artifact_semantics(
            solution=solution,
            metadata=metadata,
            authority=authority,
        )
        self._check_solution_result_omission_semantics(
            solution=solution,
            metadata=metadata,
            authority=authority,
        )
        try:
            solution = SolutionResult(
                values=solution.values,
                metadata=metadata,
                retained_continuations=snapshot_artifact_store(
                    store=solution.retained_continuations,
                    authorities=authority.artifacts,
                ),
                replay_artifacts=snapshot_artifact_store(
                    store=solution.replay_artifacts,
                    authorities=authority.artifacts,
                ),
                auxiliary_artifacts=snapshot_artifact_store(
                    store=solution.auxiliary_artifacts,
                    authorities=authority.artifacts,
                ),
                diagnostics=snapshot_artifact_store(
                    store=solution.diagnostics,
                    authorities=authority.artifacts,
                ),
                omissions=solution.omissions,
            )
        except (BeartypeCallHintViolation, TypeError, ValueError) as error:
            raise InvalidSimulationInputError(
                f"SolutionResult artifact payloads cannot be detached: {error}"
            ) from error
        try:
            values = cast("ValueStore", solution.values).materialize()
        except (TypeError, ValueError) as error:
            raise InvalidSimulationInputError(
                f"SolutionResult values cannot be materialized: {error}"
            ) from error
        self._check_solution_value_schemas(
            metadata=metadata,
            authority=authority,
            expected_coverage=expected_coverage,
            values=values,
        )
        return authority, values, solution

    @staticmethod
    def _snapshot_solution_envelope(
        *, solution: _SolutionResultBoundary
    ) -> _SolutionResultBoundary:
        """Own exact result stores and metadata before any lazy callback can run."""
        if type(solution) is not SolutionResult:
            msg = "SolutionResult has the wrong exact container type."
            raise InvalidSimulationInputError(msg)
        supplied_metadata = solution.metadata
        supplied_values = solution.values  # noqa: PD011
        supplied_retained_continuations = solution.retained_continuations
        supplied_replay_artifacts = solution.replay_artifacts
        supplied_auxiliary_artifacts = solution.auxiliary_artifacts
        supplied_diagnostics = solution.diagnostics
        supplied_omissions = solution.omissions
        if type(supplied_metadata) is not SolutionMetadata:
            msg = "SolutionResult metadata has the wrong exact container type."
            raise InvalidSimulationInputError(msg)
        try:
            snapshot = SolutionResult(
                values=snapshot_value_store(cast("ValueStore", supplied_values)),
                metadata=snapshot_solution_metadata(supplied_metadata),
                retained_continuations=snapshot_artifact_store(
                    store=supplied_retained_continuations
                ),
                replay_artifacts=snapshot_artifact_store(
                    store=supplied_replay_artifacts
                ),
                auxiliary_artifacts=snapshot_artifact_store(
                    store=supplied_auxiliary_artifacts
                ),
                omissions=snapshot_omissions(supplied_omissions),
                diagnostics=snapshot_artifact_store(store=supplied_diagnostics),
            )
        except (BeartypeCallHintViolation, TypeError, ValueError) as error:
            raise InvalidSimulationInputError(
                f"SolutionResult envelope cannot be snapshotted: {error}"
            ) from error
        return snapshot

    def _check_solution_result_metadata(  # noqa: C901, PLR0912
        self,
        *,
        metadata: SolutionMetadata,
        expected_fingerprint: str,
        expected_model_fingerprint: str,
    ) -> None:
        """Reject provenance, version, and regime metadata mismatches."""
        expected_solver_types = {
            regime_name: (
                f"{type(user_regime.solver).__module__}."
                f"{type(user_regime.solver).__qualname__}"
            )
            for regime_name, user_regime in self.user_regimes.items()
        }
        expected_solver_identities = {
            regime_name: user_regime.solver.identity
            for regime_name, user_regime in self.user_regimes.items()
        }
        expected_replay_routes = {
            regime_name: _replay_route_identity(regime.simulation.replay_route)
            for regime_name, regime in self._regimes.items()
        }
        metadata_defects: list[str] = []
        route_plugin_mismatches = tuple(
            regime_name
            for regime_name, regime in self._regimes.items()
            if regime.simulation.external_replay_route is not None
            and not _same_exact_artifact_contract(
                actual=regime.simulation.external_replay_route.plugin_identity,
                expected=expected_solver_identities[regime_name],
            )
        )
        if route_plugin_mismatches:
            metadata_defects.append(
                "replay routes are owned by a different solver plugin at "
                f"{route_plugin_mismatches}"
            )
        if type(metadata.source) is not SolutionSource:
            metadata_defects.append("source has the wrong exact type")
        elif metadata.source is SolutionSource.IN_MEMORY and not (
            _same_exactly_typed(
                actual=metadata.model_instance_id,
                expected=self._solution_model_instance_id,
            )
        ):
            metadata_defects.append("model_instance_id does not match this Model")
        elif type(metadata.model_instance_id) is not str:
            metadata_defects.append("model_instance_id has the wrong exact type")
        if not _same_exactly_typed(
            actual=metadata.model_fingerprint,
            expected=expected_model_fingerprint,
        ):
            metadata_defects.append("model_fingerprint does not match this model")
        if not _same_exactly_typed(
            actual=metadata.params_fingerprint,
            expected=expected_fingerprint,
        ):
            metadata_defects.append(
                "params_fingerprint does not match the canonical simulation params"
            )
        if not _same_exactly_typed(
            actual=metadata.pylcm_version,
            expected=PYLCM_VERSION,
        ):
            metadata_defects.append(
                f"pylcm_version={metadata.pylcm_version!r} (expected {PYLCM_VERSION!r})"
            )
        if not _same_exactly_typed(
            actual=metadata.solver_api_version, expected=SOLVER_API_VERSION
        ):
            metadata_defects.append(
                "solver_api_version="
                f"{metadata.solver_api_version} "
                f"(expected {SOLVER_API_VERSION})"
            )
        if not _same_exactly_typed(
            actual=metadata.solution_schema_version,
            expected=SOLUTION_SCHEMA_VERSION,
        ):
            metadata_defects.append(
                "solution_schema_version="
                f"{metadata.solution_schema_version} "
                f"(expected {SOLUTION_SCHEMA_VERSION})"
            )
        if not _same_exactly_typed(actual=metadata.n_periods, expected=self.n_periods):
            metadata_defects.append(
                f"n_periods={metadata.n_periods} (expected {self.n_periods})"
            )
        if not _same_exactly_typed(
            actual=metadata.regime_names,
            expected=tuple(self._regimes),
        ):
            metadata_defects.append(
                f"regime_names={metadata.regime_names!r} "
                f"(expected {tuple(self._regimes)!r})"
            )
        if type(metadata.retention) is not ResultRetention:
            metadata_defects.append("retention has the wrong exact type")
        if not _same_exact_artifact_contract(
            actual=metadata.solver_types,
            expected=expected_solver_types,
        ):
            metadata_defects.append("solver_types do not match this model")
        if not _same_exact_artifact_contract(
            actual=metadata.solver_identities,
            expected=expected_solver_identities,
        ):
            metadata_defects.append("solver plugin identities or versions do not match")
        if not _same_exact_artifact_contract(
            actual=metadata.replay_routes,
            expected=expected_replay_routes,
        ):
            metadata_defects.append("replay route identities or versions do not match")
        if metadata_defects:
            msg = (
                "SolutionResult metadata is incompatible with this model: "
                + "; ".join(metadata_defects)
                + "."
            )
            raise InvalidSimulationInputError(msg)

    @staticmethod
    def _check_solution_result_artifact_coordinates(
        *,
        solution: _SolutionResultBoundary,
        metadata: SolutionMetadata,
        expected_coverage: set[tuple[int, RegimeName]],
        authority: SolutionAuthority,
    ) -> None:
        """Reject malformed coordinates, versions, and channels."""
        named_stores = (
            ("retained_continuations", solution.retained_continuations),
            ("replay_artifacts", solution.replay_artifacts),
            ("auxiliary_artifacts", solution.auxiliary_artifacts),
            ("diagnostics", solution.diagnostics),
        )
        present_stores = tuple(store for _name, store in named_stores)
        # Snapshot coordinate sets once; duplicate detection never needs payloads.
        present_ref_sets = tuple(set(store) for store in present_stores)
        present_refs = set().union(*present_ref_sets)
        omission_refs = set(solution.omissions)
        diagnostic_present_refs = {
            ref for ref in solution.diagnostics if ref.key == SOLVER_DIAGNOSTICS
        }
        diagnostic_omission_refs = {
            ref
            for ref, reason in solution.omissions.items()
            if ref.key == SOLVER_DIAGNOSTICS and reason is OmissionReason.NOT_PERSISTED
        }
        artifact_present_refs = present_refs - diagnostic_present_refs
        artifact_omission_refs = omission_refs - diagnostic_omission_refs
        unexpected = tuple(
            sorted(
                ref
                for ref in present_refs | omission_refs
                if (ref.period, ref.regime) not in expected_coverage
            )
        )
        overlap = tuple(sorted(present_refs & omission_refs))
        duplicated = tuple(
            sorted(
                ref
                for ref in present_refs
                if sum(ref in refs for refs in present_ref_sets) > 1
            )
        )
        standard_type_ids = {
            SIMULATION_POLICY.type_id,
            DISSOLUTION_FLAG.type_id,
            EGM_CONTINUATION.type_id,
            SOLVER_DIAGNOSTICS.type_id,
        }
        exact_standard_keys = {
            SIMULATION_POLICY,
            DISSOLUTION_FLAG,
            EGM_CONTINUATION,
            SOLVER_DIAGNOSTICS,
        }
        wrong_versions = tuple(
            sorted(
                ref
                for ref in present_refs | omission_refs
                if ref.key.type_id in standard_type_ids
                and ref.key not in exact_standard_keys
            )
        )
        wrong_channels = tuple(
            sorted(
                (store_name, ref)
                for store_name, store in named_stores
                for ref in store
                if ref.key in {SIMULATION_POLICY, DISSOLUTION_FLAG}
                and authority.replay.get(ref) is not None
                and store_name != authority.replay[ref].channel
            )
        )
        channel_to_store = {
            ArtifactChannel.CONTINUATION: "retained_continuations",
            ArtifactChannel.REPLAY: "replay_artifacts",
            ArtifactChannel.AUXILIARY: "auxiliary_artifacts",
            ArtifactChannel.DIAGNOSTIC: "diagnostics",
        }
        custom_wrong_channels = tuple(
            sorted(
                (store_name, ref)
                for store_name, store in named_stores
                for ref in store
                if ref in authority.artifacts
                and store_name
                != channel_to_store[authority.artifacts[ref].descriptor.channel]
            )
        )
        described_refs = set(metadata.artifact_descriptors)
        undeclared = tuple(
            sorted(
                ref
                for ref in artifact_present_refs | artifact_omission_refs
                if ref not in described_refs
            )
        )
        missing_accounting = tuple(
            sorted(described_refs - (artifact_present_refs | artifact_omission_refs))
        )
        if (
            unexpected
            or overlap
            or duplicated
            or wrong_versions
            or wrong_channels
            or custom_wrong_channels
            or undeclared
            or missing_accounting
        ):
            msg = (
                "SolutionResult artifact coordinates are incompatible: "
                f"unexpected={unexpected}, refs both present and omitted={overlap}, "
                f"refs in multiple stores={duplicated}, "
                f"wrong schema versions={wrong_versions}, "
                f"wrong channels={wrong_channels + custom_wrong_channels}, "
                f"undeclared={undeclared}, missing accounting={missing_accounting}."
            )
            raise InvalidSimulationInputError(msg)

    @staticmethod
    def _check_solution_result_present_artifact_semantics(
        *,
        solution: _SolutionResultBoundary,
        metadata: SolutionMetadata,
        authority: SolutionAuthority,
    ) -> None:
        """Require each present model artifact to be applicable and selected."""
        present_refs = set(solution.retained_continuations)
        present_refs.update(solution.replay_artifacts)
        present_refs.update(solution.auxiliary_artifacts)
        defects: list[str] = []
        for ref in sorted(present_refs):
            artifact_authority = authority.artifacts.get(ref)
            if artifact_authority is None:
                defects.append(f"{ref!r} has no model authority")
                continue
            descriptor = artifact_authority.descriptor
            if not artifact_authority.applicable:
                defects.append(f"{ref!r} is not applicable")
                continue
            selected = (
                metadata.retention is ResultRetention.VALUES_AND_REPLAY
                and descriptor.channel is ArtifactChannel.REPLAY
            ) or (
                metadata.retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
                and descriptor.persistence is PersistencePolicy.MODEL_VERIFIABLE
            )
            if not selected:
                defects.append(
                    f"{ref!r} is not selected by retention {metadata.retention.value!r}"
                )
        if defects:
            raise InvalidSimulationInputError(
                "SolutionResult present artifacts are incompatible with model "
                "authority: " + "; ".join(defects) + "."
            )

    @staticmethod
    def _check_solution_result_omission_semantics(
        *,
        solution: _SolutionResultBoundary,
        metadata: SolutionMetadata,
        authority: SolutionAuthority,
    ) -> None:
        """Require every omission to agree with model authority and retention."""
        defects: list[str] = []
        for ref, reason in solution.omissions.items():
            artifact_authority = authority.artifacts.get(ref)
            if artifact_authority is None:
                if (
                    ref.key == SOLVER_DIAGNOSTICS
                    and reason is OmissionReason.NOT_PERSISTED
                ):
                    continue
                defects.append(f"{ref!r} has no model authority")
                continue
            descriptor = artifact_authority.descriptor
            selected = (
                metadata.retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS
                or (
                    metadata.retention is ResultRetention.VALUES_AND_REPLAY
                    and descriptor.channel is ArtifactChannel.REPLAY
                )
            )
            if not artifact_authority.applicable:
                expected = OmissionReason.NOT_APPLICABLE
            elif not selected:
                expected = OmissionReason.NOT_REQUESTED
            elif descriptor.persistence is PersistencePolicy.NOT_PERSISTED:
                expected = OmissionReason.NOT_PERSISTED
            elif artifact_authority.required:
                defects.append(
                    f"{ref!r} is a required selected model-verifiable artifact"
                )
                continue
            else:
                expected = OmissionReason.UNSUPPORTED
            if reason is not expected:
                defects.append(
                    f"{ref!r} has reason {reason.value!r}, expected {expected.value!r}"
                )
        if defects:
            raise InvalidSimulationInputError(
                "SolutionResult omissions are incompatible with model authority: "
                + "; ".join(defects)
                + "."
            )

    @staticmethod
    def _check_solution_result_coverage(
        *,
        actual_coverage: set[tuple[int, RegimeName]],
        expected_coverage: set[tuple[int, RegimeName]],
        label: str,
    ) -> None:
        """Reject missing or unexpected coordinates in one result store."""
        if actual_coverage != expected_coverage:
            missing = tuple(sorted(expected_coverage - actual_coverage))
            unexpected = tuple(sorted(actual_coverage - expected_coverage))
            msg = (
                f"SolutionResult {label} coverage is incompatible with this model: "
                f"missing={missing}, unexpected={unexpected}."
            )
            raise InvalidSimulationInputError(msg)

    def _check_solution_value_schemas(
        self,
        *,
        metadata: SolutionMetadata,
        authority: SolutionAuthority,
        expected_coverage: set[tuple[int, RegimeName]],
        values: PeriodToRegimeToVArr,
    ) -> None:
        """Check the canonical value snapshot against model-owned schemas."""
        schemas = metadata.value_schemas
        schema_defects: list[str] = []
        for period, regime_name in sorted(expected_coverage):
            value = values[period][regime_name]
            schema = schemas[(period, regime_name)]
            descriptor = authority.values[(period, regime_name)]  # noqa: PD011
            if not isinstance(value, descriptor.payload_type):
                schema_defects.append(
                    f"({period}, {regime_name!r}) payload type="
                    f"{type(value).__name__!r}, expected="
                    f"{descriptor.payload_type.__name__!r}"
                )
                continue
            if type(schema) is not ValueArraySchema:
                schema_defects.append(
                    f"({period}, {regime_name!r}) schema type="
                    f"{type(schema).__name__!r}, expected='ValueArraySchema'"
                )
                continue
            if tuple(value.shape) != descriptor.shape or not _same_exactly_typed(
                actual=schema.shape, expected=descriptor.shape
            ):
                schema_defects.append(
                    f"({period}, {regime_name!r}) shape={tuple(value.shape)!r}, "
                    f"schema={schema.shape!r}, expected={descriptor.shape!r}"
                )
            if str(value.dtype) != descriptor.dtype or not _same_exactly_typed(
                actual=schema.dtype, expected=descriptor.dtype
            ):
                schema_defects.append(
                    f"({period}, {regime_name!r}) dtype={str(value.dtype)!r}, "
                    f"schema={schema.dtype!r}, expected={descriptor.dtype!r}"
                )
            if not _same_exactly_typed(
                actual=schema.axis_names, expected=descriptor.axis_names
            ):
                schema_defects.append(
                    f"({period}, {regime_name!r}) axis_names="
                    f"{schema.axis_names!r}, expected={descriptor.axis_names!r}"
                )
        if schema_defects:
            msg = "SolutionResult value schemas are incompatible: " + "; ".join(
                schema_defects
            )
            raise InvalidSimulationInputError(msg)

    def _check_solution_result_artifacts(
        self,
        *,
        solution: _SolutionResultBoundary,
        authority: SolutionAuthority,
        values: PeriodToRegimeToVArr,
    ) -> tuple[
        PeriodToRegimeToSimulationPolicy,
        PeriodToRegimeToDissolutionFlags,
    ]:
        """Require every replay artifact the labelled solution's routes consume."""
        policies = _materialize_artifact_projection(
            store=solution.replay_artifacts,
            key=SIMULATION_POLICY,
            authority=authority,
        )
        dissolution_flags = _materialize_artifact_projection(
            store=solution.replay_artifacts,
            key=DISSOLUTION_FLAG,
            authority=authority,
            required_only=True,
        )
        self._check_solution_result_replay_policies(
            solution=solution,
            authority=authority,
            policies=policies,
            values=values,
        )
        self._check_solution_result_dissolution_flags(
            solution=solution,
            authority=authority,
            dissolution_flags=dissolution_flags,
        )
        return (
            cast("PeriodToRegimeToSimulationPolicy", policies),
            cast("PeriodToRegimeToDissolutionFlags", dissolution_flags),
        )

    def _build_external_replay_readers(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        solution: _SolutionResultBoundary,
        metadata: SolutionMetadata,
        authority: SolutionAuthority,
        flat_params: FlatParams,
    ) -> _PeriodToRegimeToReplayReader:
        """Validate each plugin replay cell once and build its immutable reader."""
        readers: dict[int, dict[RegimeName, ReplayReader]] = {}
        for regime_name, regime in self._regimes.items():
            route = regime.simulation.external_replay_route
            if route is None:
                continue
            base_state_action_space = regime.solution.state_action_space(
                regime_params=flat_params[regime_name]
            )
            for period in regime.active_periods:
                state_action_space = _state_action_space_for_period(
                    regime=regime,
                    base=base_state_action_space,
                    period=period,
                )
                replay_context = _replay_model_context_from_state_action_space(
                    regime_name=regime_name,
                    period=period,
                    state_action_space=state_action_space,
                )
                build_context = SimulationBuildContext(
                    period=replay_context.period,
                    regime_name=replay_context.regime_name,
                    state_names=replay_context.state_names,
                    action_names=replay_context.action_names,
                    state_nodes=replay_context.state_nodes,
                    action_nodes=replay_context.action_nodes,
                )
                declared = {
                    ref.key: artifact_authority
                    for ref, artifact_authority in authority.artifacts.items()
                    if ref.period == period
                    and ref.regime == regime_name
                    and artifact_authority.descriptor.channel is ArtifactChannel.REPLAY
                }

                try:
                    requirements = route.requirements(context=replay_context)
                except Exception as error:
                    raise InvalidSimulationInputError(
                        "External replay route could not declare requirements at "
                        f"({period}, {regime_name!r}): {error}"
                    ) from error
                if type(requirements) is not ReplayRouteRequirements:
                    raise InvalidSimulationInputError(
                        "External replay route returned non-exact requirements at "
                        f"({period}, {regime_name!r})."
                    )
                required_keys = requirements.required_artifacts
                authority_required_keys = frozenset(
                    key
                    for key, artifact_authority in declared.items()
                    if artifact_authority.required
                    and _same_exact_artifact_contract(
                        actual=artifact_authority.consumer_route,
                        expected=route.identity,
                    )
                )
                if not _same_exact_artifact_contract(
                    actual=required_keys,
                    expected=authority_required_keys,
                ):
                    raise InvalidSimulationInputError(
                        "External replay route requirements differ from its "
                        "model-built required authorities at "
                        f"({period}, {regime_name!r})."
                    )

                snapshot_artifacts: dict[ArtifactKey, object] = {}
                snapshot_authorities: dict[ArtifactKey, ArtifactAuthority] = {}
                defects: list[str] = []
                for key, declared_authority in declared.items():
                    ref = ArtifactRef(period=period, regime=regime_name, key=key)
                    model_authority = declared_authority
                    described = metadata.artifact_descriptors.get(ref)
                    if not _same_exact_artifact_contract(
                        actual=described,
                        expected=authority.artifact_descriptors.get(ref),
                    ):
                        defects.append(
                            f"{key.type_id!r} descriptive schema differs from model "
                            "authority"
                        )
                        continue
                    if not model_authority.applicable:
                        if ref in solution.replay_artifacts:
                            defects.append(f"{key.type_id!r} is not applicable")
                        continue
                    if ref not in solution.replay_artifacts:
                        omission = solution.omissions.get(ref)
                        reason = "unrecorded" if omission is None else omission.value
                        if model_authority.required:
                            defects.append(f"{key.type_id!r} is absent ({reason})")
                        continue
                    if key in required_keys:
                        try:
                            replay_store = solution.replay_artifacts
                            materialize_from_snapshot = (
                                replay_store._materialize_from_template_snapshot  # noqa: SLF001
                            )
                            materialized = materialize_from_snapshot(
                                ref,
                                template_snapshot=snapshot_artifact_template_declaration(
                                    model_authority
                                ),
                            )
                            payload = _canonicalize_artifact_payload(
                                payload=materialized,
                                authority=model_authority,
                            )
                        except (TypeError, ValueError) as error:
                            defects.append(
                                f"{key.type_id!r} mismatched_payload: {error}"
                            )
                            continue
                        snapshot_artifacts[key] = payload
                        snapshot_authorities[key] = model_authority

                if defects:
                    raise InvalidSimulationInputError(
                        "External replay artifacts are incompatible at "
                        f"({period}, {regime_name!r}): " + "; ".join(defects) + "."
                    )
                snapshot = ReplayRouteSnapshot(
                    artifacts=MappingProxyType(snapshot_artifacts),
                    authorities=MappingProxyType(snapshot_authorities),
                    metadata=metadata,
                )
                try:
                    route.validate(snapshot=snapshot, context=build_context)
                    reader = route.build_reader(
                        snapshot=snapshot,
                        context=build_context,
                    )
                except InvalidSimulationInputError:
                    raise
                except Exception as error:
                    raise InvalidSimulationInputError(
                        "External replay route rejected artifacts at "
                        f"({period}, {regime_name!r}): {error}"
                    ) from error
                if not isinstance(reader, ReplayReader):
                    raise InvalidSimulationInputError(
                        "External replay route returned a non-callable reader at "
                        f"({period}, {regime_name!r})."
                    )
                readers.setdefault(period, {})[regime_name] = reader
        return MappingProxyType(
            {
                period: MappingProxyType(regime_to_reader)
                for period, regime_to_reader in readers.items()
            }
        )

    def _check_solution_result_replay_policies(
        self,
        *,
        solution: _SolutionResultBoundary,
        authority: SolutionAuthority,
        policies: Mapping[int, Mapping[RegimeName, object]],
        values: PeriodToRegimeToVArr,
    ) -> None:
        """Require each solver decision that cannot be reconstructed from values."""
        policies_without_route = tuple(
            sorted(
                (period, regime_name)
                for period, regime_to_policy in policies.items()
                for regime_name in regime_to_policy
                if self._regimes[regime_name].simulation.replay_route.payload_type
                is None
            )
        )
        if policies_without_route:
            msg = (
                f"Artifact {SIMULATION_POLICY.type_id!r} has no declared replay route "
                "at (period, regime): "
                f"{policies_without_route}."
            )
            raise InvalidSimulationInputError(msg)

        missing_or_mismatched_policies: list[tuple[int, RegimeName, str]] = []
        for period, regime_to_value in values.items():
            for regime_name in regime_to_value:
                ref = ArtifactRef(
                    period=period,
                    regime=regime_name,
                    key=SIMULATION_POLICY,
                )
                descriptor = authority.replay[ref]
                policy_read = descriptor.route
                if not isinstance(policy_read, EGMPolicyRead | NNBEGMPolicyRead):
                    continue
                supplied = policies.get(period, {}).get(regime_name)
                omission = solution.omissions.get(ref)
                if supplied is None:
                    if (
                        not descriptor.required
                        and omission is OmissionReason.NOT_APPLICABLE
                    ):
                        continue
                    reason = omission.value if omission is not None else "unrecorded"
                    missing_or_mismatched_policies.append((period, regime_name, reason))
                    continue

                payload_defect = _built_in_policy_payload_defect(
                    supplied=supplied,
                    descriptor=descriptor,
                    period=period,
                )
                if payload_defect is not None:
                    missing_or_mismatched_policies.append(
                        (
                            period,
                            regime_name,
                            f"mismatched_payload: {payload_defect}",
                        )
                    )
        if missing_or_mismatched_policies:
            raise InvalidSimulationInputError(
                _missing_policy_message(
                    missing_or_mismatched_policies=tuple(missing_or_mismatched_policies)
                )
            )

    def _check_solution_result_dissolution_flags(
        self,
        *,
        solution: _SolutionResultBoundary,
        authority: SolutionAuthority,
        dissolution_flags: Mapping[int, Mapping[RegimeName, object]],
    ) -> None:
        """Validate and require only flags consumed by model-declared gates."""
        missing_dissolution_flags = self._find_malformed_dissolution_flags(
            dissolution_flags=dissolution_flags,
            authority=authority,
        )
        for ref, descriptor in authority.replay.items():
            if ref.key != DISSOLUTION_FLAG or not descriptor.required:
                continue
            supplied = dissolution_flags.get(ref.period, {}).get(ref.regime)
            if supplied is not None:
                continue
            omission = solution.omissions.get(ref)
            reason = omission.value if omission is not None else "unrecorded"
            missing_dissolution_flags.append((ref.period, ref.regime, reason))
        if missing_dissolution_flags:
            msg = (
                f"Required artifact {DISSOLUTION_FLAG.type_id!r} is absent or "
                "invalid at "
                "(period, regime, reason): "
                f"{tuple(dict.fromkeys(missing_dissolution_flags))}. Re-solve with "
                "retention=ResultRetention.VALUES_AND_REPLAY."
            )
            raise InvalidSimulationInputError(msg)

    def _find_malformed_dissolution_flags(
        self,
        *,
        dissolution_flags: Mapping[int, Mapping[RegimeName, object]],
        authority: SolutionAuthority,
    ) -> list[tuple[int, RegimeName, str]]:
        """Return structural defects among materialized required flags."""
        malformed: list[tuple[int, RegimeName, str]] = []
        for period, regime_to_flag in dissolution_flags.items():
            for regime_name, supplied in regime_to_flag.items():
                ref = ArtifactRef(
                    period=period,
                    regime=regime_name,
                    key=DISSOLUTION_FLAG,
                )
                descriptor = authority.replay[ref]
                supplied_shape = tuple(getattr(supplied, "shape", ()))
                supplied_dtype = getattr(supplied, "dtype", None)
                if (
                    not descriptor.applicable
                    or descriptor.payload_type is None
                    or not isinstance(supplied, descriptor.payload_type)
                    or supplied_shape != descriptor.shape
                    or supplied_dtype is None
                    or str(np.dtype(supplied_dtype)) != descriptor.dtype
                ):
                    malformed.append((period, regime_name, "mismatched_payload"))
        return malformed

    def _fail_if_simulation_is_unsupported(self) -> None:
        """Refuse model configurations whose solved decision cannot be replayed."""
        fixed_cost_regimes = tuple(
            regime_name
            for regime_name, regime in self._regimes.items()
            if regime.simulation.replay_route.replay_mode is ReplayMode.UNSUPPORTED
        )
        if not fixed_cost_regimes:
            return
        msg = (
            "Simulation for NNBEGM with UniformObservedFixedCost is not implemented: "
            "solution integrates the observed cost analytically, but simulation "
            "cannot yet draw it and replay the contingent keeper/adjuster policy. "
            f"Affected regimes: {fixed_cost_regimes}. Solve-only use remains supported."
        )
        raise UnsupportedOperationError(msg)

    @beartype(conf=PARAMS_CONF)
    def simulate(  # noqa: C901, PLR0912
        self,
        *,
        params: UserParams,
        initial_conditions: UserInitialConditions | pd.DataFrame,
        solution: _SolutionResultBoundary | None = None,
        log_level: LogLevel,
        seed: int | None = None,
        subject_batch_size: int = 0,
        log_path: str | Path | None = None,
        log_keep_n_latest: int = 3,
        max_compilation_workers: int | None = None,
    ) -> SimulationResult:
        """Simulate the model forward, optionally solving first.

        When ``solution`` is omitted, the model is solved before simulation. Pass
        the complete result from ``solve()`` to replay a separate solve without
        splitting values from solver-specific artifacts.

        Args:
            params: Model parameters compatible with `get_params_template()`.
                Parameters can be provided at exactly one of three levels:
                - Model level: {"arg_0": 0.0} - propagates to all functions needing
                  arg_0
                - Regime level: {"regime_0": {"arg_0": 0.0}} - propagates within
                  regime_0
                - Function level: {"regime_0": {"func": {"arg_0": 0.0}}} - direct
                  specification
                Values may be `pd.Series` with labeled indices; they are
                auto-converted to JAX arrays.
            initial_conditions: Mapping of state names (plus `"regime_id"`) to arrays.
                All arrays must have the same length (number of subjects). The
                `"regime_id"` entry must contain integer regime codes (from
                `model.regime_names_to_ids`). May also be a `pd.DataFrame`
                with a `"regime_name"` column carrying regime label strings
                (auto-converted via `initial_conditions_from_dataframe`).
                Subjects starting in a COLLECTIVE regime also need an
                `"own_stakeholder"` entry naming the role each one occupies,
                as an integer code from the model's role vocabulary
                (`model.stakeholder_names_to_ids`): which partner a row is
                decides which regime it enters when the household dissolves.
            solution: Complete labelled result returned by ``solve()``. Required
                replay artifacts are validated before forward simulation starts. Its
                canonical parameters and value schemas are checked even when
                ``log_level="off"``. In-memory results must carry this model's instance
                token; restored results instead match the durable model fingerprint.
                When omitted, ``simulate`` obtains the same complete result from an
                automatic solve.
            seed: Random seed.
            subject_batch_size: How to partition the subject axis of the forward
                simulation. Results are invariant to this knob — per-subject RNG
                keys are drawn for the full population and sliced by global index.
                - `0` (default): one pass over the whole (padded) population.
                - `> 0`: chunk the subjects into passes of this size, bounding the
                  per-period device workspace. Under distributed grids each chunk
                  is placed onto the subject mesh axis (the size is rounded up to
                  a device multiple); the value-function arrays stay sharded
                  throughout.
            log_level: Verbosity, and the runtime-validation policy it implies.
                Required — pick deliberately for the situation:
                - `"off"` — silent; initial-condition, transition-probability,
                  and NaN checks skipped.
                - `"warning"` — validation runs, failures logged as warnings,
                  the run continues.
                - `"progress"` — as `"warning"`, plus timing.
                - `"debug"` — validation runs and **raises** on the first
                  failure; adds value-function stats.
                Start every project at `"debug"`: fail early and gather maximum
                diagnostics. Ease to `"warning"` / `"off"` only once the model
                is trusted and you need the speed or the non-raising behaviour
                for an estimation loop.
            log_path: Directory for persisting diagnostic snapshots. Optional at
                every level; snapshots are written only when it is set.
            log_keep_n_latest: Maximum number of snapshots to retain on disk.
            max_compilation_workers: Maximum number of threads for parallel XLA
                compilation. Only used when ``solution`` is omitted (i.e. when
                solve runs automatically). Defaults to the number of
                physical CPU cores.
        Returns:
            SimulationResult object. Call .to_dataframe() to get a pandas DataFrame,
            optionally with additional_targets.

        """
        log = get_logger(log_level=log_level)
        self._fail_if_simulation_is_unsupported()
        # The canonical parameters bind both the supplied result preflight and an
        # automatic solve. Process them once and keep one model-authoritative seam.
        flat_params = self._process_params(params)
        if solution is not None:
            (
                period_to_regime_to_V_arr,
                period_to_regime_to_sim_policy,
                period_to_regime_to_dissolution_flags,
                period_to_regime_to_replay_reader,
            ) = self._resolve_solution_result(
                solution=solution, flat_params=flat_params
            )
        else:
            period_to_regime_to_V_arr = None
            period_to_regime_to_sim_policy = None
            period_to_regime_to_dissolution_flags = None
            period_to_regime_to_replay_reader = None
        if isinstance(initial_conditions, pd.DataFrame):
            initial_conditions = initial_conditions_from_dataframe(
                df=initial_conditions,
                user_regimes=self.user_regimes,
                regime_names_to_ids=self.regime_names_to_ids,
            )
        initial_conditions = canonicalize_initial_conditions(
            initial_conditions=initial_conditions,
            regimes=self._regimes,
        )
        # Align the subject axis to the block size the simulate path needs.
        # Every chunk must match the AOT-compiled shape, and under distributed
        # grids each chunk is additionally placed onto the subject mesh axis,
        # so the chunk itself is rounded up to a device multiple (mirroring
        # `_resolve_compile_batch_size`) before the subject axis is padded to
        # a multiple of it. Without chunking, distribution alone needs a
        # device multiple. Pad rows duplicate the last real subject and are
        # trimmed inside `simulate`; a multiple of 1 (single pass) is a no-op.
        distributes = self._distributes_subjects() and len(jax.devices()) > 1
        if subject_batch_size > 0:
            raw_n_subjects = len(next(iter(initial_conditions.values())))
            alignment = min(subject_batch_size, raw_n_subjects)
            if distributes:
                n_devices = len(jax.devices())
                alignment = -(-alignment // n_devices) * n_devices
        elif distributes:
            alignment = len(jax.devices())
        else:
            alignment = 1
        initial_conditions, original_n_subjects = pad_initial_conditions_to_multiple(
            initial_conditions=initial_conditions,
            multiple=alignment,
        )
        # The edge-fold state/source-param collision guard runs on simulation as
        # well as solve because a supplied SolutionResult skips backward induction.
        # Running it before compilation or routing covers both entry paths.
        if any(regime.gated_edges for regime in self._regimes.values()):
            _reject_edge_fold_state_param_collisions(
                regimes=self._regimes,
                base_state_action_spaces=_build_base_state_action_spaces(
                    regimes=self._regimes, flat_params=flat_params
                ),
                flat_params=flat_params,
            )
        if validation_enabled(log):
            try:
                validate_initial_conditions(
                    initial_conditions=initial_conditions,
                    regimes=self._regimes,
                    regime_names_to_ids=self.regime_names_to_ids,
                    flat_params=flat_params,
                    ages=self.ages,
                )
            except InvalidInitialConditionsError as error:
                raise_or_warn(logger=log, error=error)
        validate_transitions(
            regimes=self._regimes,
            flat_params=flat_params,
            ages=self.ages,
            logger=log,
        )
        # `actual_n_subjects` is the user's real population (matched against the
        # declared `n_subjects`); `padded_n_subjects` is the leading axis the
        # dispatch actually sees. They are equal unless distributed padding ran.
        actual_n_subjects = original_n_subjects
        padded_n_subjects = len(next(iter(initial_conditions.values())))
        compile_batch_size = self._resolve_compile_batch_size(
            subject_batch_size=subject_batch_size,
            padded_n_subjects=padded_n_subjects,
            actual_n_subjects=actual_n_subjects,
            flat_params=flat_params,
            max_compilation_workers=max_compilation_workers,
            log=log,
        )
        if solution is None:
            solution = self._solve_from_flat_params(
                flat_params=flat_params,
                params=params,
                log=log,
                retention=ResultRetention.VALUES_AND_REPLAY,
                max_compilation_workers=max_compilation_workers,
                log_path=log_path,
                log_keep_n_latest=log_keep_n_latest,
            )
            (
                period_to_regime_to_V_arr,
                period_to_regime_to_sim_policy,
                period_to_regime_to_dissolution_flags,
                period_to_regime_to_replay_reader,
            ) = self._resolve_solution_result(
                solution=solution, flat_params=flat_params
            )
        if (
            period_to_regime_to_V_arr is None
            or period_to_regime_to_sim_policy is None
            or period_to_regime_to_dissolution_flags is None
            or period_to_regime_to_replay_reader is None
        ):
            raise AssertionError("Simulation solution inputs were not resolved.")
        simulate_regimes = self._resolve_simulate_regimes(
            actual_n_subjects=actual_n_subjects,
            compile_batch_size=compile_batch_size,
            log=log,
        )
        result = simulate(
            flat_params=flat_params,
            initial_conditions=initial_conditions,
            regimes=simulate_regimes,
            regime_names_to_ids=self.regime_names_to_ids,
            logger=log,
            period_to_regime_to_V_arr=period_to_regime_to_V_arr,
            period_to_regime_to_dissolution_flags=(
                period_to_regime_to_dissolution_flags
            ),
            period_to_regime_to_sim_policy=period_to_regime_to_sim_policy,
            period_to_regime_to_replay_reader=period_to_regime_to_replay_reader,
            ages=self.ages,
            simulation_output_dtypes=self.simulation_output_dtypes,
            seed=seed,
            subject_batch_size=compile_batch_size,
            original_n_subjects=original_n_subjects,
        )
        # AOT-compiled regimes carry `jax.stages.Compiled` callables that
        # wrap an unpicklable `LoadedExecutable`. `to_dataframe` only reads
        # the lazy DAG functions / constraints / transitions on
        # `regime.simulation`, never the compiled callables — so swap in
        # the lazy regimes to keep the result cloudpickle-safe.
        if simulate_regimes is not self._regimes:
            result._regimes = self._regimes  # noqa: SLF001
        if log_path is not None and validation_raises(log):
            _save_simulate_snapshot(
                model=self,
                params=params,
                initial_conditions=initial_conditions,
                period_to_regime_to_V_arr=period_to_regime_to_V_arr,
                result=result,
                log_path=Path(log_path),
                log_keep_n_latest=log_keep_n_latest,
            )
        return result

    def _resolve_compile_batch_size(
        self,
        *,
        subject_batch_size: int,
        padded_n_subjects: int,
        actual_n_subjects: int,
        flat_params: FlatParams,
        max_compilation_workers: int | None,
        log: logging.Logger,
    ) -> int:
        """Map the `subject_batch_size` knob to a concrete chunk shape.

        - `0` ⇒ the whole padded population (single pass).
        - `> 0` ⇒ that size, clamped to the population. Under multi-device
          distribution the chunk is additionally rounded up to the next multiple
          of the device count: every chunk is placed onto the subject mesh axis
          (see `subject_array_sharding`), so its leading axis must divide evenly
          across the devices. The value-function arrays stay sharded throughout —
          chunking never gathers them.

        Also AOT-compiles (and caches) the simulate functions for the resolved
        shape when `n_subjects` matches the population.
        """
        aot_active = (
            self.n_subjects is not None and self.n_subjects == actual_n_subjects
        )
        if subject_batch_size > 0:
            compile_batch_size = min(subject_batch_size, padded_n_subjects)
            if self._distributes_subjects():
                n_devices = len(jax.devices())
                compile_batch_size = min(
                    -(-compile_batch_size // n_devices) * n_devices,
                    padded_n_subjects,
                )
        else:
            compile_batch_size = padded_n_subjects
        if aot_active:
            self._ensure_simulate_compiled(
                compile_batch_size=compile_batch_size,
                flat_params=flat_params,
                max_compilation_workers=max_compilation_workers,
                log=log,
            )
        return compile_batch_size

    def _distributes_subjects(self) -> bool:
        """Return whether any grid in any regime is distributed across devices."""
        return any(
            grid.distributed
            for regime in self._regimes.values()
            for grid in regime.solution.grids.values()
        )

    def _ensure_simulate_compiled(
        self,
        *,
        compile_batch_size: int,
        flat_params: FlatParams,
        max_compilation_workers: int | None,
        log: logging.Logger,
    ) -> None:
        """Compile and cache the simulate functions for a chunk shape."""
        with self._simulate_compile_lock:
            cached = compile_batch_size in self._simulate_compile_cache
        if cached:
            return
        compiled = compile_all_simulation_phases(
            regimes=self._regimes,
            flat_params=flat_params,
            ages=self.ages,
            n_subjects=compile_batch_size,
            max_compilation_workers=max_compilation_workers,
            logger=log,
        )
        with self._simulate_compile_lock:
            self._simulate_compile_cache[compile_batch_size] = compiled

    def _process_params(self, params: UserParams) -> FlatParams:
        """Broadcast, convert Series, dtype-cast, and validate user params.

        Step order matters: `convert_series_in_params` runs *between*
        `broadcast_to_template` and `cast_params_to_canonical_dtypes` so
        the dtype cast walks a uniform tree (no `pd.Series` to special-
        case).
        """
        flat_params = broadcast_to_template(
            params=params, template=self._params_template, required=True
        )
        if has_series(flat_params):
            flat_params = convert_series_in_params(
                flat_params=flat_params,
                ages=self.ages,
                user_regimes=self.user_regimes,
                regime_names_to_ids=self.regime_names_to_ids,
            )
        flat_params = cast_params_to_canonical_dtypes(flat_params)
        flat_params = materialize_granular_transition_params(
            flat_params=flat_params,
            expansions={
                regime_name: regime.granular_param_expansions
                for regime_name, regime in self._regimes.items()
            },
        )
        _validate_param_types(flat_params)
        fail_if_nonpositive_taste_shock_scale(flat_params)
        return flat_params


def _missing_policy_message(
    *, missing_or_mismatched_policies: tuple[tuple[int, RegimeName, str], ...]
) -> str:
    """Explain which replay policies are absent or invalid and how to obtain them."""
    msg = (
        f"Required artifact {SIMULATION_POLICY.type_id!r} is absent or "
        "invalid at (period, regime, reason): "
        f"{missing_or_mismatched_policies}. Re-solve with "
        "retention=ResultRetention.VALUES_AND_REPLAY."
    )
    if any(
        reason == OmissionReason.NOT_PERSISTED.value
        for _period, _regime_name, reason in missing_or_mismatched_policies
    ):
        msg += (
            " A policy omitted as 'not_persisted' is the NNBEGM replay policy of "
            "an AdaptiveOuterMesh search: it is read against the solve-generated "
            "mesh this model instance holds beside the result, so no persistable "
            "retention keeps it; only VALUES_AND_REPLAY retains it."
        )
    return msg


def _readable_template(value: object) -> object:
    """Replace every leaf of a params template by its name or string form."""
    if isinstance(value, Mapping):
        return {key: _readable_template(inner) for key, inner in value.items()}
    return getattr(value, "__name__", str(value))
