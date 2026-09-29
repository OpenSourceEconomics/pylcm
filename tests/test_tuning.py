"""The execution-settings evaluator times only candidates that change widths.

Every timed solve reads an injected clock, so the paired-block classification is
deterministic here: a fake clock hands out the solve durations a test declares,
in the evaluator's AB/BA order (baseline, candidate, candidate, baseline).
"""

import ast
import inspect
from collections.abc import Callable, Sequence
from types import MappingProxyType

import jax
import numpy as np
import pytest

from _lcm.execution import hlo_fusions
from _lcm.version import __version__ as pylcm_version
from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.solvers import GridSearch
from lcm.tuning import (
    CandidateStatus,
    ReduceFusionVerdict,
    StructuralClassifier,
    TunedSettings,
    _array_ulp_gap,
    _classify_timing_blocks,
    _count_materialised,
    _paired_timing,
    _relative_repeat_spread,
    classify_reduce_fusions,
    evaluate_execution_settings,
)
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)

_N_PERIODS = 2
_N_WEALTH = 8
_BASELINE = ExecutionConfig()
_CEILING_ABOVE_EVERY_CELL_COUNT = ExecutionConfig(axis_width_ceilings={"cell": 1000})
_NARROWER_CELLS = ExecutionConfig(axis_width_ceilings={"cell": 2})
_PERTURBED_CONSUMPTION_STOP = 3.001
_WIDE_ALLOWANCE = 10**18

type Block = tuple[float, float, float, float]


def _build_model(*, config: ExecutionConfig, consumption_stop: float = 3.0) -> Model:
    """Build a two-period GridSearch model whose cell axis has extent `_N_WEALTH`."""
    final_age_alive = START_AGE + _N_PERIODS - 2
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=_N_WEALTH)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(
                        start=1, stop=consumption_stop, n_points=3
                    ),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=config,
    )


def _unperturbed(config: ExecutionConfig) -> Model:
    return _build_model(config=config)


def _perturbed_candidate(config: ExecutionConfig) -> Model:
    """Seed an output perturbation into every configuration but the baseline's."""
    if config == _BASELINE:
        return _build_model(config=config)
    return _build_model(config=config, consumption_stop=_PERTURBED_CONSUMPTION_STOP)


class _PairedClock:
    """Hand out declared solve durations as a monotone clock, and count reads."""

    def __init__(self, *, blocks: Sequence[Block]) -> None:
        self.durations = [duration for block in blocks for duration in block]
        self.n_reads = 0
        self.now = 0.0

    def __call__(self) -> float:
        solve, is_end = divmod(self.n_reads, 2)
        self.n_reads += 1
        if is_end:
            self.now += self.durations[solve]
        return self.now


def _evaluate(
    *,
    candidates: tuple[ExecutionConfig, ...],
    blocks: Sequence[Block] = ((10.0, 5.0, 5.0, 10.0),),
    build_model: Callable[[ExecutionConfig], Model] = _unperturbed,
    ulp_allowance: int | None = None,
    budget_seconds: float = 1_000.0,
    clock: _PairedClock | None = None,
    structural_classifier: StructuralClassifier | None = None,
) -> TunedSettings:
    return evaluate_execution_settings(
        build_model=build_model,
        params=get_params(n_periods=_N_PERIODS),
        baseline=_BASELINE,
        candidates=candidates,
        budget_seconds=budget_seconds,
        ulp_allowance=ulp_allowance,
        clock=_PairedClock(blocks=blocks) if clock is None else clock,
        structural_classifier=structural_classifier,
    )


@pytest.fixture(scope="module")
def faster_record() -> TunedSettings:
    return _evaluate(candidates=(_NARROWER_CELLS,))


def test_evaluate_execution_settings_reports_an_unbinding_ceiling_inactive() -> None:
    """A ceiling above every core's cell count resolves to the baseline's widths."""
    record = _evaluate(candidates=(_CEILING_ABOVE_EVERY_CELL_COUNT,))

    assert record.outcomes[0].status is CandidateStatus.INACTIVE


def test_evaluate_execution_settings_times_nothing_for_an_inactive_candidate() -> None:
    """No clock read happens when every candidate is inactive."""
    clock = _PairedClock(blocks=())
    _evaluate(candidates=(_CEILING_ABOVE_EVERY_CELL_COUNT,), clock=clock)

    assert clock.n_reads == 0


def test_evaluate_execution_settings_keeps_the_baseline_when_nothing_is_active() -> (
    None
):
    record = _evaluate(candidates=(_CEILING_ABOVE_EVERY_CELL_COUNT,))

    assert record.changed is False


def test_evaluate_execution_settings_resolves_the_narrower_cell_width() -> None:
    """The active candidate's compiled widths carry the ceiling's cell width."""
    record = _evaluate(candidates=(_NARROWER_CELLS,))

    assert any("'cell': 2" in label for label in record.outcomes[0].resolved_widths)


def test_evaluate_execution_settings_reports_a_faster_candidate_faster(
    faster_record: TunedSettings,
) -> None:
    assert faster_record.outcomes[0].status is CandidateStatus.FASTER


def test_evaluate_execution_settings_adopts_the_faster_candidates_ceiling(
    faster_record: TunedSettings,
) -> None:
    assert faster_record.axis_width_ceilings == {"cell": 2}


def test_evaluate_execution_settings_decides_a_tenfold_gain_in_one_block() -> None:
    """A gain at least ten times either arm's repeat spread needs one block only."""
    clock = _PairedClock(blocks=((10.0, 5.0, 5.0, 10.0),))
    _evaluate(candidates=(_NARROWER_CELLS,), clock=clock)

    assert clock.n_reads == 8


def test_evaluate_execution_settings_reports_a_slower_candidate_slower() -> None:
    record = _evaluate(candidates=(_NARROWER_CELLS,), blocks=((5.0, 10.0, 10.0, 5.0),))

    assert record.outcomes[0].status is CandidateStatus.SLOWER


def test_evaluate_execution_settings_keeps_the_baseline_over_a_slower_candidate() -> (
    None
):
    record = _evaluate(candidates=(_NARROWER_CELLS,), blocks=((5.0, 10.0, 10.0, 5.0),))

    assert record.changed is False


def test_evaluate_execution_settings_accepts_a_consistent_four_block_gain() -> None:
    """Same sign in four blocks, median beyond the baseline's between-block range."""
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        blocks=(
            (10.0, 8.0, 8.0, 11.0),
            (10.0, 8.0, 8.0, 10.5),
            (10.2, 8.0, 8.0, 10.0),
            (10.0, 8.0, 8.0, 10.0),
        ),
    )

    assert record.outcomes[0].status is CandidateStatus.FASTER


def test_evaluate_execution_settings_calls_a_sign_flipping_gain_inconclusive() -> None:
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        blocks=(
            (10.0, 8.0, 8.0, 11.0),
            (10.0, 11.0, 11.0, 10.0),
            (10.0, 8.0, 8.0, 10.0),
            (10.0, 8.0, 8.0, 10.0),
        ),
    )

    assert record.outcomes[0].status is CandidateStatus.INCONCLUSIVE


def test_evaluate_execution_settings_calls_a_two_percent_gain_inconclusive() -> None:
    """A same-sign gain beyond the baseline range still needs 5 % to promote."""
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        blocks=(
            (10.0, 9.8, 9.8, 10.01),
            (10.0, 9.8, 9.8, 10.0),
            (10.0, 9.8, 9.8, 10.0),
            (10.0, 9.8, 9.8, 10.0),
        ),
    )

    assert record.outcomes[0].status is CandidateStatus.INCONCLUSIVE


def test_evaluate_execution_settings_runs_four_blocks_for_a_three_percent_gain() -> (
    None
):
    """A zero-spread block does not decide a gain below the 5 % floor."""
    clock = _PairedClock(blocks=((10.0, 9.7, 9.7, 10.0),) * 4)
    _evaluate(candidates=(_NARROWER_CELLS,), clock=clock)

    assert clock.n_reads == 32


def test_evaluate_execution_settings_decides_a_thirty_percent_gain_in_one_block() -> (
    None
):
    """A gain above both 5 % and ten times a small spread needs one block only."""
    clock = _PairedClock(blocks=((10.0, 7.0, 7.0, 10.1),))
    _evaluate(candidates=(_NARROWER_CELLS,), clock=clock)

    assert clock.n_reads == 8


def test_evaluate_execution_settings_stops_when_four_blocks_overrun_the_budget() -> (
    None
):
    """The first block costs 37 s, so three more cannot fit a 60 s budget."""
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        blocks=((10.0, 8.0, 8.0, 11.0),),
        budget_seconds=60.0,
    )

    assert record.outcomes[0].status is CandidateStatus.OVER_BUDGET


def test_evaluate_execution_settings_rejects_outputs_beyond_the_allowance() -> None:
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        build_model=_perturbed_candidate,
        ulp_allowance=4,
    )

    assert record.outcomes[0].status is CandidateStatus.REJECTED_OUTPUTS


def test_evaluate_execution_settings_rejects_perturbed_outputs_by_default() -> None:
    """Exact equality is the default gate."""
    record = _evaluate(candidates=(_NARROWER_CELLS,), build_model=_perturbed_candidate)

    assert record.outcomes[0].status is CandidateStatus.REJECTED_OUTPUTS


def test_evaluate_execution_settings_times_no_rejected_candidate() -> None:
    clock = _PairedClock(blocks=())
    _evaluate(
        candidates=(_NARROWER_CELLS,), build_model=_perturbed_candidate, clock=clock
    )

    assert clock.n_reads == 0


def test_evaluate_execution_settings_accepts_outputs_within_the_allowance() -> None:
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        build_model=_perturbed_candidate,
        ulp_allowance=_WIDE_ALLOWANCE,
    )

    assert record.outcomes[0].status is CandidateStatus.FASTER


def test_evaluate_execution_settings_measures_the_seeded_perturbation() -> None:
    """The perturbed candidate's values sit more than four ULP from the baseline's."""
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        build_model=_perturbed_candidate,
        ulp_allowance=_WIDE_ALLOWANCE,
    )

    assert (record.outcomes[0].ulp_gap or 0) > 4


def test_evaluate_execution_settings_records_the_declared_allowance() -> None:
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        build_model=_perturbed_candidate,
        ulp_allowance=_WIDE_ALLOWANCE,
    )

    assert record.ulp_allowance == _WIDE_ALLOWANCE


def test_evaluate_execution_settings_refuses_a_candidate_changing_the_budget() -> None:
    """Only width fields may vary; a safety margin never does."""
    with pytest.raises(ValueError, match="device_memory_bytes"):
        _evaluate(candidates=(ExecutionConfig(device_memory_bytes=10**9),))


def test_evaluate_execution_settings_refuses_more_than_two_candidates() -> None:
    with pytest.raises(ValueError, match="at most two"):
        _evaluate(candidates=(_NARROWER_CELLS,) * 3)


def test_tuned_settings_round_trips_through_json(
    faster_record: TunedSettings,
) -> None:
    assert TunedSettings.from_json(faster_record.to_json()) == faster_record


def _x64_precision() -> str:
    return "float64" if jax.config.read("jax_enable_x64") else "float32"


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("device_count", lambda: len(jax.devices())),
        ("device_kind", lambda: jax.devices()[0].device_kind),
        ("precision", _x64_precision),
        ("pylcm_version", lambda: pylcm_version),
        ("jax_version", lambda: jax.__version__),
        ("sharded_states", lambda: ()),
        ("device_memory_bytes", lambda: None),
    ],
)
def test_tuned_settings_key_carries_the_identity_field(
    *, faster_record: TunedSettings, field: str, expected: Callable[[], object]
) -> None:
    assert getattr(faster_record.key, field) == expected()


def test_tuned_settings_key_carries_the_parameter_shape_signature(
    faster_record: TunedSettings,
) -> None:
    """The shape signature names the working-life regime's parameters, not values."""
    assert "working_life" in faster_record.key.param_shape_signature


def test_tuned_settings_key_is_parameter_value_free() -> None:
    """Two parameter draws of the same shape share one key."""
    record = _evaluate(candidates=(_CEILING_ABOVE_EVERY_CELL_COUNT,))
    params = get_params(n_periods=_N_PERIODS, disutility_of_work=0.123)
    redrawn = evaluate_execution_settings(
        build_model=_unperturbed,
        params=params,
        baseline=_BASELINE,
        candidates=(_CEILING_ABOVE_EVERY_CELL_COUNT,),
        budget_seconds=1_000.0,
        clock=_PairedClock(blocks=()),
    )

    assert redrawn.key == record.key


def test_tuned_settings_carries_the_baseline_ceilings_on_no_change() -> None:
    record = _evaluate(candidates=(_CEILING_ABOVE_EVERY_CELL_COUNT,))

    assert record.axis_width_ceilings == MappingProxyType({})


def _steps(*, start: float, n_steps: int, toward: float, dtype: type) -> np.ndarray:
    """Walk `n_steps` representable neighbours from `start` with `np.nextafter`."""
    value = dtype(start)
    for _ in range(n_steps):
        value = np.nextafter(value, dtype(toward))
    return np.array([value], dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("start", "n_steps", "toward"),
    [
        (1.0, 1, 2.0),
        (1.0, 7, 0.0),
        (-3.5, 5, -np.inf),
        (0.0, 3, -1.0),
    ],
)
def test_array_ulp_gap_counts_representable_neighbours(
    *, dtype: type, start: float, n_steps: int, toward: float
) -> None:
    got = _array_ulp_gap(
        expected=np.array([start], dtype=dtype),
        got=_steps(start=start, n_steps=n_steps, toward=toward, dtype=dtype),
    )

    assert got == n_steps


def test_array_ulp_gap_counts_signed_zeros_equal() -> None:
    assert _array_ulp_gap(expected=np.array([0.0]), got=np.array([-0.0])) == 0


def test_array_ulp_gap_refuses_a_moved_nan() -> None:
    assert (
        _array_ulp_gap(expected=np.array([np.nan, 1.0]), got=np.array([1.0, np.nan]))
        is None
    )


def test_evaluate_execution_settings_retains_candidate_timing_spread() -> None:
    """The public injected-clock path must not promote candidate-only noise."""
    clock = _PairedClock(blocks=((10.0, 1.0, 17.0, 10.0),) * 4)
    record = _evaluate(candidates=(_NARROWER_CELLS,), clock=clock)

    assert record.outcomes[0].status is CandidateStatus.INCONCLUSIVE
    assert record.changed is False
    assert clock.n_reads == 32


@pytest.mark.parametrize("scale", [0.125, 1.0, 1024.0])
@pytest.mark.parametrize("reverse_candidate", [False, True])
def test_timing_noise_class_is_scale_and_order_invariant(
    *, scale: float, reverse_candidate: bool
) -> None:
    """Candidate-only spread survives scaling and reversal of its two repeats."""
    pair = (17.0, 1.0) if reverse_candidate else (1.0, 17.0)
    block = (10.0 * scale, pair[0] * scale, pair[1] * scale, 10.0 * scale)
    for n_blocks in (1, 4):
        status, _ = _classify_timing_blocks(blocks=(block,) * n_blocks)
        assert status is CandidateStatus.INCONCLUSIVE


def test_timing_spread_includes_between_block_candidate_drift() -> None:
    """Low within-block noise cannot hide changes between paired blocks."""
    blocks = (
        (10.0, 9.7, 9.7, 10.0),
        (10.0, 5.0, 5.0, 10.0),
        (10.0, 5.0, 5.0, 10.0),
        (10.0, 5.0, 5.0, 10.0),
    )
    status, _ = _classify_timing_blocks(blocks=blocks)

    assert status is CandidateStatus.INCONCLUSIVE


@pytest.mark.parametrize(
    ("samples", "expected"),
    [((0.0, 0.0), 0.0), ((10.0, 10.0), 0.0), ((1.0, 17.0), 16.0 / 9.0)],
)
def test_relative_repeat_spread_uses_both_extremes(
    *, samples: tuple[float, ...], expected: float
) -> None:
    assert _relative_repeat_spread(samples=samples) == expected


def test_timed_solve_boundary_does_not_materialise_values() -> None:
    """Keep the literal ready solve between the two clock reads, with no snapshot.

    This is a source-boundary regression, not a GPU performance measurement.
    The public numerical and admission tests still exercise the actual models.
    """
    tree = ast.parse(inspect.getsource(_paired_timing))
    timed = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "timed"
    )
    calls = [node for node in ast.walk(timed) if isinstance(node, ast.Call)]
    assert (
        sum(isinstance(c.func, ast.Name) and c.func.id == "clock" for c in calls) == 2
    )
    assert any(
        isinstance(c.func, ast.Attribute) and c.func.attr == "solve" for c in calls
    )
    assert not any(
        isinstance(c.func, ast.Name) and c.func.id == "_solve_values" for c in calls
    )
    assert not any(
        isinstance(c.func, ast.Attribute)
        and c.func.attr in {"asarray", "items", "keys"}
        for c in calls
    )
    assert [type(node) for node in timed.body] == [
        ast.Assign,
        ast.Assign,
        ast.Assign,
        ast.Delete,
        ast.Return,
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [-1, 1])
def test_array_ulp_gap_rejects_finite_infinite_boundary(
    *, dtype: type[np.floating], sign: int
) -> None:
    """An adjacent encoding is not an allowed ULP step across the finite mask."""
    finite = np.array([sign * np.finfo(dtype).max], dtype=dtype)
    infinite = np.array([sign * np.inf], dtype=dtype)
    assert _array_ulp_gap(expected=finite, got=infinite) is None
    assert _array_ulp_gap(expected=infinite, got=finite) is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_array_ulp_gap_requires_matching_nonfinite_values(
    dtype: type[np.floating],
) -> None:
    values = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=dtype)
    assert _array_ulp_gap(expected=values, got=values.copy()) == 0
    for index, replacement in ((0, np.inf), (1, -np.inf), (2, np.nan)):
        changed = values.copy()
        changed[index] = replacement
        assert _array_ulp_gap(expected=values, got=changed) is None


def test_tuning_exposes_the_execution_reduce_fusion_classifier() -> None:
    """`lcm.tuning.classify_reduce_fusions` is the execution layer's classifier."""
    assert classify_reduce_fusions is hlo_fusions.classify_reduce_fusions


class _FirstProgramClassifier:
    """Mark the first program of each armed model as holding one materialised gather.

    `arm` runs when the evaluator builds a model, before that model's programs
    are compiled and classified, so each configuration's count is set exactly.
    """

    def __init__(self, *, materialised_configs: tuple[ExecutionConfig, ...]) -> None:
        self.materialised_configs = materialised_configs
        self.remaining = 0
        self.texts: list[str] = []

    def arm(self, config: ExecutionConfig) -> Model:
        self.remaining = 1 if config in self.materialised_configs else 0
        return _build_model(config=config)

    def __call__(self, hlo_text: str) -> dict[str, ReduceFusionVerdict]:
        self.texts.append(hlo_text)
        if self.remaining == 0:
            return {"input_reduce_fusion": ReduceFusionVerdict.FUSED_GATHER}
        self.remaining -= 1
        return {"input_reduce_fusion": ReduceFusionVerdict.MATERIALISED_GATHER}


def _evaluate_classified(
    *,
    materialised_configs: tuple[ExecutionConfig, ...],
    clock: _PairedClock | None = None,
) -> tuple[TunedSettings, _FirstProgramClassifier]:
    classifier = _FirstProgramClassifier(materialised_configs=materialised_configs)
    record = _evaluate(
        candidates=(_NARROWER_CELLS,),
        build_model=classifier.arm,
        clock=clock,
        structural_classifier=classifier,
    )
    return record, classifier


def test_evaluate_execution_settings_prunes_an_added_materialised_gather() -> None:
    """More materialised gathers than the baseline prunes the candidate untimed."""
    record, _ = _evaluate_classified(materialised_configs=(_NARROWER_CELLS,))

    assert record.outcomes[0].status is CandidateStatus.STRUCTURALLY_PRUNED


def test_evaluate_execution_settings_times_no_structurally_pruned_candidate() -> None:
    clock = _PairedClock(blocks=())
    _evaluate_classified(materialised_configs=(_NARROWER_CELLS,), clock=clock)

    assert clock.n_reads == 0


def test_evaluate_execution_settings_records_both_materialised_gather_counts() -> None:
    """The pruned outcome carries its count next to the baseline's."""
    record, _ = _evaluate_classified(materialised_configs=(_NARROWER_CELLS,))

    assert (
        record.outcomes[0].materialised_gather_fusions,
        record.baseline_materialised_gather_fusions,
    ) == (1, 0)


def test_evaluate_execution_settings_times_a_candidate_matching_the_baseline() -> None:
    """A candidate materialising no more than the baseline is timed as usual."""
    record, _ = _evaluate_classified(materialised_configs=(_BASELINE, _NARROWER_CELLS))

    assert record.outcomes[0].status is CandidateStatus.FASTER


def test_evaluate_execution_settings_hands_the_classifier_optimized_hlo() -> None:
    """Every classified text is a compiled module, as `as_text()` prints it."""
    _, classifier = _evaluate_classified(materialised_configs=())

    assert {text.split(" ", 1)[0] for text in classifier.texts} == {"HloModule"}


def test_evaluate_execution_settings_classifies_nothing_without_a_classifier(
    faster_record: TunedSettings,
) -> None:
    assert (
        faster_record.outcomes[0].materialised_gather_fusions,
        faster_record.baseline_materialised_gather_fusions,
    ) == (None, None)


def test_count_materialised_is_unclassified_when_any_fusion_is_unknown() -> None:
    """An `UNKNOWN` verdict leaves the count unset, so it cannot prune a candidate."""
    verdicts = {
        "a": ReduceFusionVerdict.MATERIALISED_GATHER,
        "b": ReduceFusionVerdict.UNKNOWN,
    }

    assert _count_materialised(programs=("x",), classifier=lambda _: verdicts) is None


def test_tuned_settings_round_trips_a_pruned_record_through_json() -> None:
    record, _ = _evaluate_classified(materialised_configs=(_NARROWER_CELLS,))

    assert TunedSettings.from_json(record.to_json()) == record
