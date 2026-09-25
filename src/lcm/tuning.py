"""Evaluate caller-supplied execution settings against a baseline.

`evaluate_execution_settings` is an explicit, opt-in preparation step; it never
runs inside model construction or `solve`. It compares at most two candidate
`ExecutionConfig`s with a baseline and returns a frozen, JSON-serialisable
`TunedSettings` record: either the winning candidate's width fields, or the
baseline's with the reason nothing changed.

Each candidate passes through three gates, in order:

- **Resolved widths.** A candidate whose solve compiles exactly the programs,
  at exactly the widths, the baseline's does is *inactive*: it cannot run
  differently, so it is neither gated nor timed.
- **Output equivalence.** Every value array is compared, keyed by period and
  regime, against the baseline's. Exact equality is the default; a declared
  ULP allowance admits a gap up to the baseline's own A/A gap plus that
  allowance. Finite masks and signed infinities must match regardless of the
  allowance; NaNs must occupy the same positions.
- **Structure (opt-in).** With a `structural_classifier`, every program a
  solve compiles is classified from its optimized HLO. A candidate whose reduce
  fusions read more materialised gather tables than the baseline's is
  *structurally pruned*: reported with both counts, never timed. The verdict
  comes from the programs' structure, not from a threshold on any width.
- **Paired timing.** One untimed warm-up solve per configuration, then paired
  blocks of warm solves in AB/BA order. No gain below 5 % of the baseline
  time promotes. One block decides when its gain also reaches ten times the
  largest relative repeat spread in either arm; otherwise four blocks must
  agree in sign, with a median relative gain beyond twice that spread and
  a median gain beyond the baseline's between-block range.

Peak device memory is a hard constraint through the model's own admission: a
candidate the admission refuses is reported refused.

The unit timed is the whole warm solve, as `Model.solve` returns it. That is
the estimation-loop cost `WARM_REPEATED_SOLVE` minimises, and it doubles as the
whole-solve holdout.
"""

import dataclasses
import json
import logging
import os
import statistics
import time
from collections.abc import Callable, Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any

import jax
import jaxlib
import numpy as np

from _lcm.execution.hlo_fusions import (  # noqa: F401
    ReduceFusionVerdict,
    classify_reduce_fusions,
)
from _lcm.solution.fingerprint import _param_shape_signature
from _lcm.version import __version__ as pylcm_version
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig
from lcm.model import Model
from lcm.solver_api import ResultRetention
from lcm.typing import UserParams

# The fields a candidate may change; every other `ExecutionConfig` field is the
# caller's layout, allocation or safety margin and stays at the baseline's.
_TUNABLE_FIELDS = ("axis_widths", "axis_width_ceilings")

# Format string of the per-program compile line `backward_induction` logs; its
# first argument is the program label, which names the compiled widths.
_COMPILED_LINE = "  compiled  %s  %s"

# Name of the log-record attribute on the compile line that carries the
# compiled program, so the evaluator can read its optimized HLO.
_COMPILED_PROGRAM_ATTRIBUTE = "lcm_compiled"

_MAX_CANDIDATES = 2
_N_BLOCKS = 4
_ONE_BLOCK_EFFECT_RATIO = 10.0
_FOUR_BLOCK_SPREAD_RATIO = 2.0
_MIN_RELATIVE_GAIN = 0.05


class Objective(Enum):
    """What the evaluator minimises."""

    WARM_REPEATED_SOLVE = "warm_repeated_solve"
    """Ready wall time of one solve whose programs are already compiled."""


class CandidateStatus(Enum):
    """Where one candidate left the evaluation."""

    INACTIVE = "inactive"
    """Compiles the baseline's programs at the baseline's widths; not timed."""
    REFUSED_ADMISSION = "refused_admission"
    """The model's memory admission refused a width the candidate asked for."""
    REJECTED_OUTPUTS = "rejected_outputs"
    """Outputs differ from the baseline's beyond the gate; not timed."""
    OVER_BUDGET = "over_budget"
    """The remaining blocks would overrun the timing budget."""
    INCONCLUSIVE = "inconclusive"
    """The paired blocks do not separate the candidate from the baseline."""
    SLOWER = "slower"
    """The baseline is faster."""
    FASTER = "faster"
    """The candidate is faster."""
    STRUCTURALLY_PRUNED = "structurally_pruned"
    """More reduce fusions read a materialised gather table than the baseline's."""


# Maps one optimized HLO module to the verdict of each of its reduce fusions.
type StructuralClassifier = Callable[[str], Mapping[str, ReduceFusionVerdict]]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SettingsKey:
    """Identity a record is valid for; a mismatch in any field voids it."""

    model_structure: str
    """Model structure digest; the same for every parameter vector."""
    param_shape_signature: str
    """Rank, shape and dtype of every parameter leaf, never its value."""
    sharded_states: tuple[str, ...]
    """States carrying a device axis."""
    device_count: int
    """Number of devices the model runs on."""
    device_kind: str
    """Kind of those devices, as JAX reports it."""
    device_pool_limit_bytes: tuple[int | None, ...]
    """Allocator pool limit of each device; empty on an unbudgeted model."""
    device_memory_bytes: int | None
    """Requested per-device memory budget."""
    device_memory_headroom_fraction: float
    """Share of each device's pool kept out of the budget."""
    precision: str
    """`"float64"` under x64, else `"float32"`."""
    pylcm_version: str
    """pylcm version, including its commit for a development install."""
    jax_version: str
    """JAX version."""
    jaxlib_version: str
    """jaxlib version."""
    xla_flags: str
    """The `XLA_FLAGS` environment variable, which can change code generation."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class CandidateOutcome:
    """What the evaluation found for one candidate."""

    index: int
    """Position of the candidate in the caller's tuple."""
    status: CandidateStatus
    """Where the candidate left the evaluation."""
    resolved_widths: tuple[str, ...]
    """Sorted labels of every program the candidate's solve compiled."""
    ulp_gap: int | None
    """Largest value gap to the baseline in ULP; `None` when not comparable."""
    block_gains_seconds: tuple[float, ...]
    """Per block, baseline mean minus candidate mean; positive is faster."""
    materialised_gather_fusions: int | None = None
    """Reduce fusions reading a materialised gather table; `None` if unclassified."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class TunedSettings:
    """The evaluator's verdict, as the width fields to pass to `ExecutionConfig`."""

    key: SettingsKey
    """Identity the record is valid for."""
    objective: Objective
    """What was minimised."""
    changed: bool
    """Whether a candidate replaced the baseline."""
    reason: str
    """Why the winner won, or why nothing changed."""
    axis_widths: Mapping[str, Any]
    """`ExecutionConfig.axis_widths` of the verdict."""
    axis_width_ceilings: Mapping[str, int]
    """`ExecutionConfig.axis_width_ceilings` of the verdict."""
    ulp_allowance: int | None
    """Declared ULP allowance; `None` means exact equality was required."""
    baseline_ulp_gap: int
    """Largest value gap between two baseline solves, in ULP."""
    baseline_resolved_widths: tuple[str, ...]
    """Sorted labels of every program the baseline's solve compiled."""
    outcomes: tuple[CandidateOutcome, ...]
    """One outcome per candidate, in the caller's order."""
    evaluation_wall_seconds: float
    """Wall time of the whole evaluation, preparation solves included."""
    baseline_materialised_gather_fusions: int | None = None
    """The baseline's count of such reduce fusions; `None` without a classifier."""

    def apply_to(self, *, config: ExecutionConfig) -> ExecutionConfig:
        """Return `config` with this record's width fields."""
        return dataclasses.replace(
            config,
            axis_widths=self.axis_widths,
            axis_width_ceilings=self.axis_width_ceilings,
        )

    def to_json(self) -> str:
        """Serialise the record to a JSON string."""
        return json.dumps(self, default=_jsonable, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> TunedSettings:
        """Rebuild a record from `to_json` output."""
        raw: dict[str, Any] = json.loads(text)
        key: dict[str, Any] = {
            **raw["key"],
            "sharded_states": tuple(raw["key"]["sharded_states"]),
            "device_pool_limit_bytes": tuple(raw["key"]["device_pool_limit_bytes"]),
        }
        outcomes = []
        for outcome in raw["outcomes"]:
            fields: dict[str, Any] = {
                **outcome,
                "status": CandidateStatus(outcome["status"]),
                "resolved_widths": tuple(outcome["resolved_widths"]),
                "block_gains_seconds": tuple(outcome["block_gains_seconds"]),
            }
            outcomes.append(CandidateOutcome(**fields))
        record: dict[str, Any] = {
            **raw,
            "key": SettingsKey(**key),
            "objective": Objective(raw["objective"]),
            "axis_widths": _frozen(raw["axis_widths"]),
            "axis_width_ceilings": _frozen(raw["axis_width_ceilings"]),
            "baseline_resolved_widths": tuple(raw["baseline_resolved_widths"]),
            "outcomes": tuple(outcomes),
        }
        return cls(**record)


def evaluate_execution_settings(
    *,
    build_model: Callable[[ExecutionConfig], Model],
    params: UserParams,
    baseline: ExecutionConfig,
    candidates: tuple[ExecutionConfig, ...],
    budget_seconds: float,
    objective: Objective = Objective.WARM_REPEATED_SOLVE,
    ulp_allowance: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
    structural_classifier: StructuralClassifier | None = None,
) -> TunedSettings:
    """Compare at most two candidate configurations with a baseline.

    Args:
        build_model: Builds the model under one execution configuration. Widths
            are fixed when a model is built, so each configuration is its own
            model instance.
        params: Parameters every solve uses.
        baseline: The configuration a candidate must beat.
        candidates: At most two configurations, each differing from the
            baseline only in `axis_widths` and `axis_width_ceilings`.
        budget_seconds: Timed-solve seconds the evaluation may spend, per
            candidate. The first block always runs; the tuner stops before
            three more blocks whose projected cost would overrun it.
        objective: What is minimised.
        ulp_allowance: `None` requires bitwise-equal values. An integer admits a
            gap up to the baseline's own A/A gap plus that many ULP.
        clock: Read before and after every timed solve, and nowhere else.
        structural_classifier: Opt-in. Classifies each compiled program's
            optimized HLO, for example `classify_reduce_fusions`. A candidate
            with more materialised gather tables than the baseline is pruned
            before timing; any `UNKNOWN` fusion leaves the count unset and
            prunes nothing. `UnrecognisedHloError` from the classifier
            propagates. `None` classifies nothing.

    Returns:
        The winning candidate's width fields, or the baseline's with the reason
        nothing changed.

    Raises:
        ValueError: More than two candidates, a candidate that changes a field
            other than the widths, or a non-positive budget or negative
            allowance.

    """
    _fail_if_request_invalid(
        baseline=baseline,
        candidates=candidates,
        budget_seconds=budget_seconds,
        ulp_allowance=ulp_allowance,
    )
    started = time.monotonic()
    baseline_model = build_model(baseline)
    baseline_widths, baseline_values, baseline_programs = _prepare(
        model=baseline_model,
        params=params,
        collect_programs=structural_classifier is not None,
    )
    baseline_materialised = _count_materialised(
        programs=baseline_programs, classifier=structural_classifier
    )
    baseline_ulp_gap = _ulp_gap(
        expected=baseline_values,
        got=_solve_values(model=baseline_model, params=params),
    )
    if baseline_ulp_gap is None:
        msg = "Two baseline solves disagree in keys, shapes, dtypes or NaN pattern."
        raise RuntimeError(msg)
    outcomes = tuple(
        _evaluate_candidate(
            index=index,
            model=build_model(candidate),
            baseline_model=baseline_model,
            params=params,
            baseline_widths=baseline_widths,
            baseline_values=baseline_values,
            allowed_gap=None
            if ulp_allowance is None
            else baseline_ulp_gap + ulp_allowance,
            budget_seconds=budget_seconds,
            clock=clock,
            classifier=structural_classifier,
            baseline_materialised=baseline_materialised,
        )
        for index, candidate in enumerate(candidates)
    )
    winner = _winner(outcomes=outcomes)
    verdict = baseline if winner is None else candidates[winner.index]
    return TunedSettings(
        key=_settings_key(model=baseline_model, config=baseline, params=params),
        objective=objective,
        changed=winner is not None,
        reason=_reason(outcomes=outcomes, winner=winner),
        axis_widths=verdict.axis_widths,
        axis_width_ceilings=verdict.axis_width_ceilings,
        ulp_allowance=ulp_allowance,
        baseline_ulp_gap=baseline_ulp_gap,
        baseline_resolved_widths=baseline_widths,
        outcomes=outcomes,
        evaluation_wall_seconds=time.monotonic() - started,
        baseline_materialised_gather_fusions=baseline_materialised,
    )


def _fail_if_request_invalid(
    *,
    baseline: ExecutionConfig,
    candidates: tuple[ExecutionConfig, ...],
    budget_seconds: float,
    ulp_allowance: int | None,
) -> None:
    """Refuse requests the evaluator cannot answer faithfully."""
    if len(candidates) > _MAX_CANDIDATES:
        msg = f"Evaluate at most two candidates; got {len(candidates)}."
        raise ValueError(msg)
    if not budget_seconds > 0:
        msg = f"budget_seconds must be positive; got {budget_seconds!r}."
        raise ValueError(msg)
    if ulp_allowance is not None and ulp_allowance < 0:
        msg = f"ulp_allowance must not be negative; got {ulp_allowance!r}."
        raise ValueError(msg)
    fixed = [
        field.name
        for field in dataclasses.fields(ExecutionConfig)
        if field.name not in _TUNABLE_FIELDS
    ]
    for index, candidate in enumerate(candidates):
        changed = [
            name
            for name in fixed
            if getattr(candidate, name) != getattr(baseline, name)
        ]
        if changed:
            msg = (
                f"Candidate {index} changes {changed}; only {list(_TUNABLE_FIELDS)} "
                "may differ from the baseline."
            )
            raise ValueError(msg)


def _evaluate_candidate(
    *,
    index: int,
    model: Model,
    baseline_model: Model,
    params: UserParams,
    baseline_widths: tuple[str, ...],
    baseline_values: Mapping[tuple[int, str], np.ndarray],
    allowed_gap: int | None,
    budget_seconds: float,
    clock: Callable[[], float],
    classifier: StructuralClassifier | None,
    baseline_materialised: int | None,
) -> CandidateOutcome:
    """Run one candidate through the width, output, structure and timing gates."""
    try:
        widths, values, programs = _prepare(
            model=model, params=params, collect_programs=classifier is not None
        )
    except ExecutionPlanningError:
        return CandidateOutcome(
            index=index,
            status=CandidateStatus.REFUSED_ADMISSION,
            resolved_widths=(),
            ulp_gap=None,
            block_gains_seconds=(),
        )
    if widths == baseline_widths:
        return CandidateOutcome(
            index=index,
            status=CandidateStatus.INACTIVE,
            resolved_widths=widths,
            ulp_gap=None,
            block_gains_seconds=(),
        )
    gap = _ulp_gap(expected=baseline_values, got=values)
    if gap is None or gap > (0 if allowed_gap is None else allowed_gap):
        return CandidateOutcome(
            index=index,
            status=CandidateStatus.REJECTED_OUTPUTS,
            resolved_widths=widths,
            ulp_gap=gap,
            block_gains_seconds=(),
        )
    materialised = _count_materialised(programs=programs, classifier=classifier)
    if (
        materialised is not None
        and baseline_materialised is not None
        and materialised > baseline_materialised
    ):
        return CandidateOutcome(
            index=index,
            status=CandidateStatus.STRUCTURALLY_PRUNED,
            resolved_widths=widths,
            ulp_gap=gap,
            block_gains_seconds=(),
            materialised_gather_fusions=materialised,
        )
    status, gains = _paired_timing(
        baseline_model=baseline_model,
        model=model,
        params=params,
        budget_seconds=budget_seconds,
        clock=clock,
    )
    return CandidateOutcome(
        index=index,
        status=status,
        resolved_widths=widths,
        ulp_gap=gap,
        block_gains_seconds=gains,
        materialised_gather_fusions=materialised,
    )


def _count_materialised(
    *, programs: tuple[str, ...], classifier: StructuralClassifier | None
) -> int | None:
    """Count reduce fusions reading a materialised gather table, over all programs.

    Returns `None` when there is no classifier, or when any fusion is `UNKNOWN`:
    an unread fusion may hold a materialised gather, so the count would be a
    lower bound, and the candidate is timed rather than pruned.
    """
    if classifier is None:
        return None
    verdicts = [v for text in programs for v in classifier(text).values()]
    if ReduceFusionVerdict.UNKNOWN in verdicts:
        return None
    return sum(v is ReduceFusionVerdict.MATERIALISED_GATHER for v in verdicts)


def _paired_timing(
    *,
    baseline_model: Model,
    model: Model,
    params: UserParams,
    budget_seconds: float,
    clock: Callable[[], float],
) -> tuple[CandidateStatus, tuple[float, ...]]:
    """Time ready solves in AB/BA order, without materialising their values.

    The untimed output gate owns host snapshots. Here `Model.solve` already
    returns ready arrays, so neither conversion nor result destruction belongs
    inside the measured interval. The first block is always run; otherwise the
    existing projected-cost check decides whether three more blocks fit.
    """

    def timed(solved_model: Model) -> float:
        start = clock()
        result = solved_model.solve(
            params=params, log_level="off", retention=ResultRetention.VALUES
        )
        elapsed = clock() - start
        del result
        return elapsed

    def block() -> tuple[float, float, float, float]:
        return (
            timed(baseline_model),
            timed(model),
            timed(model),
            timed(baseline_model),
        )

    first = block()
    status, gains = _classify_timing_blocks(blocks=(first,))
    if status is not CandidateStatus.INCONCLUSIVE:
        return status, gains
    first_cost = sum(first)
    if first_cost * (_N_BLOCKS - 1) > budget_seconds - first_cost:
        return CandidateStatus.OVER_BUDGET, gains
    blocks = (first, *(block() for _ in range(_N_BLOCKS - 1)))
    return _classify_timing_blocks(blocks=blocks)


def _classify_timing_blocks(
    *, blocks: tuple[tuple[float, float, float, float], ...]
) -> tuple[CandidateStatus, tuple[float, ...]]:
    """Apply the effect floor using all repeats of both arms in an allocation.

    Durations are nonnegative ready wall seconds. A zero baseline mean supplies
    no relative-effect denominator and is inconclusive. For positive samples,
    the largest pairwise relative difference, normalised by the pair's mean,
    occurs between the minimum and maximum sample. Keep both arms' samples:
    candidate-only variation or drift between blocks is still an A/A spread.
    """
    baseline_means = tuple((a1 + a2) / 2 for a1, _, _, a2 in blocks)
    gains = tuple(
        mean - (b1 + b2) / 2
        for mean, (_, b1, b2, _) in zip(baseline_means, blocks, strict=True)
    )
    if any(mean <= 0 for mean in baseline_means):
        return CandidateStatus.INCONCLUSIVE, gains
    baseline_samples = tuple(value for a1, _, _, a2 in blocks for value in (a1, a2))
    candidate_samples = tuple(value for _, b1, b2, _ in blocks for value in (b1, b2))
    spread = max(
        _relative_repeat_spread(samples=baseline_samples),
        _relative_repeat_spread(samples=candidate_samples),
    )
    if len(blocks) == 1:
        relative_gain = abs(gains[0]) / baseline_means[0]
        floor = max(_MIN_RELATIVE_GAIN, _ONE_BLOCK_EFFECT_RATIO * spread)
        if gains[0] != 0 and relative_gain >= floor:
            return _signed_status(gain=gains[0]), gains
        return CandidateStatus.INCONCLUSIVE, gains
    median = statistics.median(gains)
    median_relative = abs(
        statistics.median(
            gain / mean for gain, mean in zip(gains, baseline_means, strict=True)
        )
    )
    same_sign = all(gain > 0 for gain in gains) or all(gain < 0 for gain in gains)
    if (
        len(blocks) == _N_BLOCKS
        and same_sign
        and abs(median) > max(baseline_means) - min(baseline_means)
        and median_relative > max(_MIN_RELATIVE_GAIN, _FOUR_BLOCK_SPREAD_RATIO * spread)
    ):
        return _signed_status(gain=median), gains
    return CandidateStatus.INCONCLUSIVE, gains


def _relative_repeat_spread(*, samples: tuple[float, ...]) -> float:
    """Largest pairwise relative difference, using the pair mean as scale."""
    low, high = min(samples), max(samples)
    return 0.0 if high == 0 else 2 * (high - low) / (high + low)


def _signed_status(*, gain: float) -> CandidateStatus:
    return CandidateStatus.FASTER if gain > 0 else CandidateStatus.SLOWER


def _winner(*, outcomes: tuple[CandidateOutcome, ...]) -> CandidateOutcome | None:
    """Return the faster candidate with the largest median gain, if any."""
    faster = [o for o in outcomes if o.status is CandidateStatus.FASTER]
    if not faster:
        return None
    return max(faster, key=lambda o: statistics.median(o.block_gains_seconds))


def _reason(
    *, outcomes: tuple[CandidateOutcome, ...], winner: CandidateOutcome | None
) -> str:
    if winner is not None:
        gain = statistics.median(winner.block_gains_seconds)
        return f"Candidate {winner.index} is faster by {gain:.6g} s per warm solve."
    if not outcomes:
        return "No candidate was supplied; the baseline stands."
    statuses = ", ".join(f"{o.index}: {o.status.value}" for o in outcomes)
    return f"No candidate beat the baseline ({statuses}); the baseline stands."


class _CompiledLabels(logging.Handler):
    """Collect the label, and optionally the program, of every compile a solve logs."""

    def __init__(self, *, collect_programs: bool) -> None:
        super().__init__(level=logging.INFO)
        self.collect_programs = collect_programs
        self.labels: set[str] = set()
        self.programs: list[jax.stages.Compiled] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg != _COMPILED_LINE or not isinstance(record.args, tuple):
            return
        self.labels.add(str(record.args[0]))
        program = getattr(record, _COMPILED_PROGRAM_ATTRIBUTE, None)
        if self.collect_programs and program is not None:
            self.programs.append(program)


def _prepare(
    *, model: Model, params: UserParams, collect_programs: bool
) -> tuple[tuple[str, ...], dict[tuple[int, str], np.ndarray], tuple[str, ...]]:
    """Solve once, untimed, returning program labels, values and optimized HLO.

    A label names its regime, core, representative age and tile widths. Two
    configurations that compile the same label set lower the same programs at
    the same widths; under a memory budget they also walk the same frontier, so
    the admission selects the same widths for both. The optimized HLO of each
    compiled program is returned only when `collect_programs` is set.
    """
    handler = _CompiledLabels(collect_programs=collect_programs)
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        values = _solve_values(model=model, params=params, log_level="progress")
    finally:
        logger.removeHandler(handler)
    programs = tuple(_program_text(program) for program in handler.programs)
    return tuple(sorted(handler.labels)), values, programs


def _program_text(program: jax.stages.Compiled) -> str:
    """Return a compiled program's optimized HLO; a program without one is refused.

    Skipping it would undercount the materialised gathers the candidate carries.
    """
    text = program.as_text()
    if text is None:
        msg = "A compiled program has no HLO text to classify."
        raise RuntimeError(msg)
    return text


def _solve_values(
    *, model: Model, params: UserParams, log_level: str = "off"
) -> dict[tuple[int, str], np.ndarray]:
    """Solve, retaining values only, and return them keyed by period and regime."""
    result = model.solve(
        params=params,
        log_level=log_level,  # ty: ignore[invalid-argument-type]
        retention=ResultRetention.VALUES,
    )
    values = result.values
    return {
        (period, regime): np.asarray(array)
        for period in values.keys()  # noqa: SIM118
        for regime, array in values[period].items()
    }


def _ulp_gap(
    *,
    expected: Mapping[tuple[int, str], np.ndarray],
    got: Mapping[tuple[int, str], np.ndarray],
) -> int | None:
    """Return the largest element gap in ULP; `None` when not comparable.

    Two value sets are comparable when they share keys, shapes, dtypes and NaN
    positions. Non-float arrays compare exactly.
    """
    if expected.keys() != got.keys():
        return None
    gaps = [_array_ulp_gap(expected=expected[key], got=got[key]) for key in expected]
    if any(gap is None for gap in gaps):
        return None
    return max((gap for gap in gaps if gap is not None), default=0)


def _array_ulp_gap(*, expected: np.ndarray, got: np.ndarray) -> int | None:
    if expected.shape != got.shape or expected.dtype != got.dtype:
        return None
    if not np.issubdtype(expected.dtype, np.floating):
        return 0 if np.array_equal(expected, got) else None
    finite = np.isfinite(expected)
    if not np.array_equal(finite, np.isfinite(got)):
        return None
    if not np.array_equal(expected[~finite], got[~finite], equal_nan=True):
        return None
    ordered_expected = _ordered_bits(np.where(finite, expected, 0))
    ordered_got = _ordered_bits(np.where(finite, got, 0))
    # The true difference lies in [0, 2**64), so unsigned wrap-around is exact.
    unsigned_expected = ordered_expected.view(np.uint64)
    unsigned_got = ordered_got.view(np.uint64)
    diff = np.where(
        ordered_expected >= ordered_got,
        unsigned_expected - unsigned_got,
        unsigned_got - unsigned_expected,
    )
    return int(diff.max(initial=0))


def _ordered_bits(values: np.ndarray) -> np.ndarray:
    """Map floats to int64 so adjacent representable floats differ by one."""
    n_bits = values.dtype.itemsize * 8
    bits = values.view(np.dtype(f"int{n_bits}")).astype(np.int64)
    return np.where(bits < 0, np.int64(-(2 ** (n_bits - 1))) - bits, bits)


def _settings_key(
    *, model: Model, config: ExecutionConfig, params: UserParams
) -> SettingsKey:
    """Read every cheaply available identity field of the baseline model."""
    execution = model._execution  # noqa: SLF001
    devices = {device.id: device for device in jax.devices()}
    flat_params = model._process_params(params)  # noqa: SLF001
    shapes = {
        regime: {name: _param_shape_signature(value) for name, value in leaves.items()}
        for regime, leaves in flat_params.items()
    }
    return SettingsKey(
        model_structure=model._model_structure_fingerprint,  # noqa: SLF001
        param_shape_signature=repr(shapes),
        sharded_states=tuple(config.sharded_states),
        device_count=len(model.execution_devices),
        device_kind=",".join(
            sorted({devices[i].device_kind for i in model.execution_devices})
        ),
        device_pool_limit_bytes=tuple(execution.device_pool_limit_bytes.values()),
        device_memory_bytes=config.device_memory_bytes,
        device_memory_headroom_fraction=config.device_memory_headroom_fraction,
        precision="float64" if jax.config.read("jax_enable_x64") else "float32",
        pylcm_version=pylcm_version,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
    )


def _jsonable(value: object) -> object:
    """Encode the record's dataclasses, enums and read-only mappings."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return dict(value)
    msg = f"Cannot serialise {type(value).__name__} to JSON."
    raise TypeError(msg)


def _frozen(value: object) -> object:
    """Freeze decoded JSON mappings the way `ExecutionConfig` stores them."""
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(child) for key, child in value.items()})
    return value
