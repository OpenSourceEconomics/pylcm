"""Time each host phase of a cold and a warm public solve on one process.

The harness attaches a handler to the `lcm` logger, runs the requested number
of public solves, and reads each call's phase records back into a table: every
phase's seconds, the residual `public_solve` keeps beyond its children, and the
JAX request counters the call issued. A second, handler-free pass over the same
model reports what the stamping handler itself costs.

The calibration the table is read for:

- a cold call issues backend compile requests;
- a warm call's compile requests equal the lowering requests it issued, which
  is zero when every executable is already resident in this process;
- the warm call's trace requests are an observation, not an assumption.

Usage:
    pixi run python benchmarks/warm_solve_phases.py --model precautionary_savings \\
        --calls 2 --release-previous yes
"""

import argparse
import contextlib
import dataclasses
import gc
import logging
import re
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

if not __package__:
    # Run as a path (`python benchmarks/warm_solve_phases.py`), the repository
    # root is not on the path, so the sibling counter module is unreachable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _lcm.solution.solve_phase_records import PHASE_NAMES, PUBLIC_PHASE
from benchmarks.asv._compile_counters import (
    CompileRequestCounts,
    count_compile_requests,
)

MODEL_NAMES = ("precautionary_savings", "iskhakov", "aca_reduced")


# Verbosity every measured call runs at. The planner's `candidate evaluation`
# record is written at DEBUG, so anything below it observes none.
LOG_LEVEL = "progress"

# `timestamped_lcm_log` writes `<ISO stamp>.<ms> <LEVEL> <message>`, so the same
# record reaches a reader either bare (from a handler that renders the message
# alone) or behind that prefix. Both are accepted, and the pattern stays anchored
# at the start of the line, so a record that quotes the grammar mid-sentence is
# not read as a phase of its own.
_STAMP_PREFIX = r"(?:[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]+ [A-Z]+ )?"

_RECORD = re.compile(
    rf"^{_STAMP_PREFIX}solve call (?P<call>[0-9a-f]+) phase (?P<name>[a-z_]+) "
    r"(?P<edge>begin|end)(?: status=(?P<status>\w+) seconds=(?P<seconds>[0-9.]+))?$"
)

_LOWERING_RECORD = re.compile(r"^\s*lowering ")
_COMPILING_RECORD = re.compile(r"^\s*compiling\b")
_CANDIDATE_RECORD = re.compile(r"^candidate evaluation ")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PhaseOutcome:
    """One phase of one public call, as the log reports it."""

    name: str
    """The phase's name in the record vocabulary."""
    status: str
    """`ok`, `error`, or `incomplete` for a phase whose end never arrived."""
    seconds: float | None
    """Monotonic duration, or `None` for an incomplete phase."""
    depth: int = 0
    """Phases of the same call open around this one: `0` for the public bracket,
    `1` for its direct children, deeper for a phase nested inside a child."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class CallPhases:
    """Every phase one public call passed through, in emission order."""

    call_id: str
    """Identifier the call stamped on its records."""
    phases: tuple[PhaseOutcome, ...]
    """The call's phases, the public bracket included."""

    def seconds(self, *, name: str) -> float | None:
        """Return the named phase's duration, or `None` when it has none."""
        for phase in self.phases:
            if phase.name == name:
                return phase.seconds
        return None

    def residual_seconds(self) -> float | None:
        """Return what the public bracket holds beyond its timed direct children.

        A nested phase's time is already inside its parent's, so only the
        public bracket's direct children are subtracted.
        """
        public = next(
            (phase.seconds for phase in self.phases if phase.depth == 0), None
        )
        if public is None:
            return None
        children = [
            phase.seconds
            for phase in self.phases
            if phase.depth == 1 and phase.seconds is not None
        ]
        return public - sum(children)


def parse_phase_records(*, lines: Sequence[str]) -> tuple[CallPhases, ...]:
    """Read every public call's phases out of a stamped log.

    A phase whose `begin` has no matching `end` is reported as `incomplete`
    with no duration, so a run that died inside a phase still names where it
    was. Each phase carries its nesting depth within the call. Same-named phases
    open at the same time — compiles running on a worker pool — are siblings at
    one depth, and each `end` closes the earliest of them still open. Calls come
    back in the order their first record appeared.

    Args:
        lines: Log lines, in emission order.

    Returns:
        One entry per call id seen, each carrying that call's phases.

    """
    open_phases: dict[tuple[str, str], list[int]] = {}
    by_call: dict[str, list[PhaseOutcome]] = {}
    for line in lines:
        match = _RECORD.match(line.strip())
        if match is None:
            continue
        call_id = match["call"]
        name = match["name"]
        phases = by_call.setdefault(call_id, [])
        depth = sum(
            open_call == call_id and open_name != name
            for open_call, open_name in open_phases
        )
        if match["edge"] == "begin":
            open_phases.setdefault((call_id, name), []).append(len(phases))
            phases.append(
                PhaseOutcome(name=name, status="incomplete", seconds=None, depth=depth)
            )
            continue
        siblings = open_phases.get((call_id, name), [])
        position = siblings.pop(0) if siblings else None
        if not siblings:
            open_phases.pop((call_id, name), None)
        if position is not None:
            depth = phases[position].depth
        outcome = PhaseOutcome(
            name=name,
            status=match["status"],
            seconds=float(match["seconds"]),
            depth=depth,
        )
        if position is None:
            phases.append(outcome)
        else:
            phases[position] = outcome
    return tuple(
        CallPhases(call_id=call_id, phases=tuple(phases))
        for call_id, phases in by_call.items()
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CallReport:
    """One call's phases beside the counters it issued."""

    phases: CallPhases
    """The call's host phases."""
    counts: CompileRequestCounts
    """Trace, lowering, and backend compile requests of the call."""
    lower_or_compile_records: int
    """`lowering` / `compiling` records the engine wrote during the call."""
    candidate_evaluations: int | None
    """`candidate evaluation` records the planner wrote during the call.

    The planner writes that record at DEBUG, so a call made at any lower
    verbosity observes none whatever the planner did, and the count is `None`
    rather than zero.
    """


def _build_precautionary_savings() -> tuple[object, object]:
    """Build the precautionary-savings model and its parameters."""
    from lcm_examples import precautionary_savings

    model = precautionary_savings.get_model(
        n_periods=5,
        shock_type="rouwenhorst",
        wealth_grid_type="lin",
        wealth_n_points=100,
        consumption_n_points=100,
    )
    params = precautionary_savings.get_params(
        shock_type="rouwenhorst", sigma=0.2, rho=0.9
    )
    return model, params


def _build_iskhakov() -> tuple[object, object]:
    """Build the taste-shock retirement model and its parameters."""
    from benchmarks.asv.bench_iskhakov_et_al_2017 import _make_model_and_params

    return _make_model_and_params(wealth_n_points=100, consumption_n_points=100)


def _build_aca_reduced() -> tuple[object, object]:
    """Build the benchmark-sized ACA baseline model and its parameters."""
    try:
        from aca_model.agent.preferences import BenchmarkPrefType
        from aca_model.benchmark import create_benchmark_model, get_benchmark_params
    except ImportError as error:
        msg = (
            "--model aca_reduced needs the `aca_model` package, which this "
            f"environment does not provide: {error}"
        )
        raise SystemExit(msg) from error

    from lcm import DiscreteGrid

    model = create_benchmark_model(
        pref_type_grid=DiscreteGrid(category_class=BenchmarkPrefType)
    )
    return model, get_benchmark_params(model=model)[2]


_BUILDERS = {
    "precautionary_savings": _build_precautionary_savings,
    "iskhakov": _build_iskhakov,
    "aca_reduced": _build_aca_reduced,
}


class _LineCollector(logging.Handler):
    """Keep every `lcm` record of one run as a rendered line."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Append one rendered record."""
        self.lines.append(record.getMessage())


@contextlib.contextmanager
def _collecting_lcm_records() -> Iterator[list[str]]:
    """Collect the `lcm` logger's records for the duration of the block."""
    handler = _LineCollector()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        yield handler.lines
    finally:
        logger.removeHandler(handler)


def _run_calls(
    *,
    model: object,
    params: object,
    n_calls: int,
    release_previous: bool,
    log_level: str = LOG_LEVEL,
) -> tuple[CallReport, ...]:
    """Solve `n_calls` times, reading each call's phases and counters back."""
    reports: list[CallReport] = []
    retained: list[object] = []
    for index in range(n_calls):
        if release_previous:
            retained.clear()
            gc.collect()
        with _collecting_lcm_records() as lines, count_compile_requests() as counts:
            result = model.solve(params=params, log_level=log_level)  # ty: ignore[unresolved-attribute]
        retained.append(result)
        calls = parse_phase_records(lines=lines)
        if len(calls) != 1:
            msg = (
                f"Call {index} wrote phase records for {len(calls)} public calls; "
                "exactly one was expected. A count of zero means the solve emitted "
                'no records at all — check that it ran above `log_level="off"`.'
            )
            raise RuntimeError(msg)
        reports.append(
            CallReport(
                phases=calls[0],
                counts=dataclasses.replace(counts),
                lower_or_compile_records=sum(
                    bool(_LOWERING_RECORD.match(line) or _COMPILING_RECORD.match(line))
                    for line in lines
                ),
                candidate_evaluations=(
                    sum(bool(_CANDIDATE_RECORD.match(line)) for line in lines)
                    if log_level == "debug"
                    else None
                ),
            )
        )
    retained.clear()
    return tuple(reports)


def _time_without_handler(*, model: object, params: object, n_calls: int) -> float:
    """Return the total seconds of `n_calls` solves with nothing attached."""
    start = time.monotonic()
    for _ in range(n_calls):
        model.solve(params=params, log_level=LOG_LEVEL)  # ty: ignore[unresolved-attribute]
    return time.monotonic() - start


def _print_report(*, reports: Sequence[CallReport]) -> None:
    """Print one block per call: its phases, residual, and counters."""
    for index, report in enumerate(reports):
        label = "cold" if index == 0 else f"warm {index}"
        public = report.phases.seconds(name=PUBLIC_PHASE)
        print(f"\ncall {index} ({label})  id={report.phases.call_id}")
        print(f"  {PUBLIC_PHASE:<24} {public:>10.6f}s")
        for name in PHASE_NAMES:
            seconds = report.phases.seconds(name=name)
            share = (
                ""
                if public in (None, 0.0) or seconds is None
                else f"  ({100 * seconds / public:5.1f}%)"
            )
            rendered = "incomplete" if seconds is None else f"{seconds:>10.6f}s"
            print(f"    {name:<22} {rendered}{share}")
        residual = report.phases.residual_seconds()
        print(f"    {'residual':<22} {residual:>10.6f}s")
        candidates = (
            "not observed (record is DEBUG)"
            if report.candidate_evaluations is None
            else report.candidate_evaluations
        )
        print(f"  candidate evaluations        {candidates}")
        print(f"  lower/compile records        {report.lower_or_compile_records}")
        print(f"  trace requests               {report.counts.trace_requests}")
        print(f"  lowering requests            {report.counts.lowering_requests}")
        print(f"  compile requests             {report.counts.compile_requests}")
        print("  confirmed code generation    not observed")


def _assert_calibration(*, reports: Sequence[CallReport]) -> None:
    """Fail loudly when a run does not reproduce the calibration facts."""
    cold = reports[0]
    if cold.counts.compile_requests <= 0:
        msg = (
            "The cold call issued no backend compile request; the counter is "
            "not observing this process's compilations."
        )
        raise AssertionError(msg)
    for index, report in enumerate(reports[1:], start=1):
        if report.counts.compile_requests != report.counts.lowering_requests:
            msg = (
                f"Warm call {index} issued {report.counts.compile_requests} compile "
                f"requests against {report.counts.lowering_requests} lowering "
                "requests; a compile request without a new lowering key is a "
                "cache miss this harness does not explain."
            )
            raise AssertionError(msg)
    for report in reports:
        public = next(
            phase for phase in report.phases.phases if phase.name == PUBLIC_PHASE
        )
        if public.status != "ok":
            msg = f"Call {report.phases.call_id} ended with status {public.status}."
            raise AssertionError(msg)
        residual = report.phases.residual_seconds()
        if residual is None or residual < 0.0:
            msg = f"Call {report.phases.call_id} has a negative residual: {residual}."
            raise AssertionError(msg)


def main(*, argv: Sequence[str] | None = None) -> None:
    """Run the harness over one model and print its phase table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--release-previous", choices=("yes", "no"), default="yes")
    args = parser.parse_args(argv)

    model, params = _BUILDERS[args.model]()
    reports = _run_calls(
        model=model,
        params=params,
        n_calls=args.calls,
        release_previous=args.release_previous == "yes",
    )
    _print_report(reports=reports)
    _assert_calibration(reports=reports)

    # Overhead is read off the warm calls alone: a cold call's wall is
    # dominated by compilation, which no second pass reproduces.
    n_warm = args.calls - 1
    if n_warm < 1:
        print("\ninstrument overhead: not measured (needs at least two calls)")
        return
    stamped = sum(
        seconds
        for report in reports[1:]
        if (seconds := report.phases.seconds(name=PUBLIC_PHASE)) is not None
    )
    unstamped = _time_without_handler(model=model, params=params, n_calls=n_warm)
    print(
        f"\ninstrument overhead over {n_warm} warm call(s): {stamped:.6f}s "
        f"stamped vs {unstamped:.6f}s handler-free ({stamped - unstamped:+.6f}s)"
    )


if __name__ == "__main__":
    main()
