"""Immutable results and supported linear costs for regime-structure analysis."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from _lcm.typing import JSONValue
from lcm.typing import ActionName, RegimeName, StateName

_COST_FEATURES = ("work", "storage", "clarity")


@dataclass(frozen=True, kw_only=True)
class LinearRegimeCostModel:
    """Declare a positive linear objective before candidate generation.

    Costs use the existing work, allocated-value-entry and structural-complexity
    features. Coefficients can express different priorities or externally fitted
    linear predictors. No measured-runtime accuracy is implied. Arbitrary callback
    and nonlinear cost models are outside this supported class.
    """

    work_weight: float = 1.0
    """Positive coefficient of declared work."""
    storage_weight: float = 1.0
    """Positive coefficient of allocated value entries."""
    complexity_weight: float = 1.0
    """Positive coefficient of structural complexity."""
    reference_scales: tuple[int, int, int] | None = None
    """Fixed feature scales, or the unchanged original costs when omitted."""
    minimum_improvement: float = 0.0
    """Required relative objective improvement over the original."""
    allow_final_only_split: bool = False
    """Permit extra final-decision-only blocks in recommendations."""

    def __post_init__(self) -> None:
        for name in ("work_weight", "storage_weight", "complexity_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                msg = f"{name} must be finite and strictly positive; got {value!r}."
                raise ValueError(msg)
        if self.reference_scales is not None and (
            len(self.reference_scales) != len(_COST_FEATURES)
            or any(
                type(value) is not int or value <= 0 for value in self.reference_scales
            )
        ):
            msg = "reference_scales must contain three positive integers."
            raise ValueError(msg)
        if (
            not math.isfinite(self.minimum_improvement)
            or not 0 <= self.minimum_improvement < 1
        ):
            msg = "minimum_improvement must be finite and in [0, 1)."
            raise ValueError(msg)

    def as_recommendation(self) -> dict[str, JSONValue]:
        """Return the search objective, including its recommendation preferences."""
        result: dict[str, JSONValue] = {
            "weights": {
                "work": self.work_weight,
                "storage": self.storage_weight,
                "clarity": self.complexity_weight,
            },
            "minimum_improvement": self.minimum_improvement,
            "allow_final_only_split": self.allow_final_only_split,
        }
        if self.reference_scales is not None:
            result["reference"] = dict(
                zip(_COST_FEATURES, self.reference_scales, strict=True)
            )
        return result


@dataclass(frozen=True)
class RegimeCost:
    """Structural work, allocation and complexity estimates."""

    work: int
    """Declared state-action evaluations and boundary charges."""
    storage: int
    """Allocated value entries, not peak memory bytes."""
    complexity: int
    """Weighted blocks, characteristics and dispatch clauses."""


@dataclass(frozen=True)
class RegimeCheck:
    """A named check with its recorded scope and outcome."""

    name: str
    """The check or layer being reported."""
    status: str
    """The recorded outcome; unresolved is distinct from passed."""


@dataclass(frozen=True)
class RegimeBlock:
    """A proposed computational block retaining economic regime identities."""

    regimes: tuple[RegimeName, ...]
    """Original economic regimes dispatched within this block."""
    periods: tuple[int, ...]
    """Original computational periods covered by the block."""
    dates: tuple[str, ...]
    """Exact age labels, or period labels in a period-based model."""
    states: tuple[StateName, ...]
    """Allocated state axes."""
    actions: tuple[ActionName, ...]
    """Allocated action axes."""
    cells: tuple[str, ...]
    """Original dated-cell identifiers."""


@dataclass(frozen=True)
class RegimeCandidate:
    """A representation and its independently recorded checks."""

    identifier: str
    """Candidate identity, or retain-original."""
    scenario: int | None
    """Allocation scenario index, or None for the original."""
    cost: RegimeCost | None
    """Estimated cost, or None if the original allocation is incomparable."""
    blocks: tuple[RegimeBlock, ...]
    """Directly inspectable computational blocks."""
    checks: tuple[RegimeCheck, ...]
    """Local checks and disposition, separate from global search coverage."""
    reasons: tuple[str, ...]
    """Reasons supporting or qualifying the proposal."""


@dataclass(frozen=True)
class RegimeAnalysisResult:
    """Inspect recommendations without joining candidate IDs in wire records.

    The data view is recursively immutable. Use to_dict for an independent,
    serializable copy. Recommendations reference the same candidate objects as
    candidates, or original when the original is retained.
    """

    status: str
    """Analysis outcome, with detailed scope in checks and data."""
    original: RegimeCandidate
    """The original representation and comparable structural costs."""
    candidates: tuple[RegimeCandidate, ...]
    """Examined complete representations."""
    recommendations: tuple[RegimeCandidate, ...]
    """Direct candidate references, including the original fallback."""
    checks: tuple[RegimeCheck, ...]
    """Global check layers; these do not replace local proposal checks."""
    cost_model: LinearRegimeCostModel
    """The objective supplied to candidate generation and pruning."""
    data: Mapping[str, JSONValue] = field(repr=False)
    """Recursively immutable detailed report and source records."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _mapping(_freeze(self.data)))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a detached JSON-serializable report."""
        return {key: _thaw(value) for key, value in self.data.items()}

    def write_report(self, directory: str | Path) -> None:
        """Write HTML, graphs and JSON to a new directory.

        Raises:
            FileExistsError: If directory already exists.
        """
        from _lcm_regime_finder.production_render import write_production_report  # noqa: PLC0415

        output = Path(directory)
        output.mkdir(parents=True)
        write_production_report(self.to_dict(), output)


def _freeze(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_thaw(item) for item in value]
    return value


def _mapping(value: JSONValue) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        msg = "Analysis record must be a mapping."
        raise TypeError(msg)
    return value


def _rows(value: JSONValue) -> tuple[Mapping[str, JSONValue], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = "Analysis rows must be a sequence."
        raise TypeError(msg)
    return tuple(_mapping(item) for item in value)


def _cost(value: JSONValue) -> RegimeCost | None:
    if value is None:
        return None
    record = _mapping(value)
    return RegimeCost(
        work=_integer(record["work"]),
        storage=_integer(record["storage"]),
        complexity=_integer(record["clarity"]),
    )


def _integer(value: JSONValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = "Structural costs must be nonnegative integers."
        raise ValueError(msg)
    return value


def _strings(value: JSONValue) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = "Analysis labels must be a sequence."
        raise TypeError(msg)
    return tuple(str(item) for item in value)


# keyword-only-exempt: primary-argument=report
def result_from_report(
    report: Mapping[str, JSONValue], *, cost_model: LinearRegimeCostModel
) -> RegimeAnalysisResult:
    """Build a stable typed view without changing the independently checked report."""
    analysis = _mapping(report.get("analysis", {}))
    declarations = _mapping(analysis.get("declarations", {}))
    cells = {str(cell["id"]): cell for cell in _rows(declarations.get("cells", []))}

    def blocks(records: Sequence[Mapping[str, JSONValue]]) -> tuple[RegimeBlock, ...]:
        result = []
        for record in records:
            identifiers = _strings(record.get("cells", []))
            members = [cells[identifier] for identifier in identifiers]
            result.append(
                RegimeBlock(
                    regimes=_strings(record.get("original_regimes", [])),
                    periods=tuple(
                        sorted({_integer(cell["period"]) for cell in members})
                    ),
                    dates=tuple(
                        dict.fromkeys(
                            str(
                                cell.get("age")
                                if cell.get("age") is not None
                                else cell["period"]
                            )
                            for cell in members
                        )
                    ),
                    states=tuple(_mapping(record.get("state_axes", {}))),
                    actions=tuple(_mapping(record.get("action_axes", {}))),
                    cells=identifiers,
                )
            )
        return tuple(result)

    original_groups: dict[str, list[str]] = {}
    for identifier, cell in cells.items():
        original_groups.setdefault(str(cell["regime"]), []).append(identifier)
    original_blocks = blocks(
        tuple(
            {
                "original_regimes": [regime],
                "cells": identifiers,
                "state_axes": {
                    name: domain
                    for identifier in identifiers
                    for name, domain in _mapping(
                        cells[identifier].get("states", {})
                    ).items()
                },
                "action_axes": {
                    name: domain
                    for identifier in identifiers
                    for name, domain in _mapping(
                        cells[identifier].get("actions", {})
                    ).items()
                },
            }
            for regime, identifiers in original_groups.items()
        )
    )
    scenarios = _rows(report.get("scenarios", []))
    original_cost = (
        _cost(_mapping(scenarios[0].get("original", {})).get("cost"))
        if scenarios
        else None
    )
    original = RegimeCandidate(
        "retain-original",
        None,
        original_cost,
        original_blocks,
        (RegimeCheck("preservation", "IDENTITY"),),
        ("Retain the supplied model unchanged.",),
    )
    decisions = {
        (row.get("scenario_index"), row.get("graph_id")): row
        for row in _rows(report.get("proposal_decisions", []))
    }
    candidates = []
    by_id = {}
    for index, scenario in enumerate(scenarios):
        for point in _rows(scenario.get("candidates", scenario.get("frontier", []))):
            identifier = str(point["id"])
            decision = decisions.get((index, identifier), {})
            candidate = RegimeCandidate(
                identifier,
                index,
                _cost(point.get("cost")),
                blocks(_rows(point.get("blocks", []))),
                (
                    RegimeCheck(
                        "disposition", str(decision.get("disposition", "investigate"))
                    ),
                    RegimeCheck(
                        "source_semantics",
                        str(
                            _mapping(decision.get("semantic_checks", {})).get(
                                "status", "UNRESOLVED"
                            )
                        ),
                    ),
                    RegimeCheck(
                        "candidate_costs",
                        str(
                            _mapping(scenario.get("independent_check", {})).get(
                                "status", "UNRESOLVED"
                            )
                        ),
                    ),
                ),
                _strings(decision.get("reasons", [])),
            )
            candidates.append(candidate)
            by_id[index, identifier] = candidate
    recommendations = tuple(
        by_id.get((row.get("scenario_index"), row.get("graph_id")), original)
        for row in _rows(report.get("recommendations", []))
    ) or (original,)
    return RegimeAnalysisResult(
        status=str(report.get("status", "UNRESOLVED")),
        original=original,
        candidates=tuple(candidates),
        recommendations=recommendations,
        checks=tuple(
            RegimeCheck(key, str(value))
            for key, value in _mapping(report.get("layers", {})).items()
        ),
        cost_model=cost_model,
        data=report,
    )
