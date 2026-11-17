"""Immutable results and supported linear costs for regime-structure analysis."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from _lcm.typing import JSONValue
from lcm.typing import ActionName, RegimeName, StateName

# Where an analysis takes its dependency facts from.
type FactsBackend = Literal["metadata", "dense", "implicit"]

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
    """State-action evaluations: periods times state grid times action grid, summed."""
    storage: int
    """Stored values: periods times state grid, summed; not peak memory bytes."""
    complexity: int
    """The number of regimes (blocks)."""


@dataclass(frozen=True)
class RegimeCheck:
    """A named check with its recorded scope and outcome."""

    name: str
    """The check or layer being reported."""
    status: str
    """The recorded outcome; unresolved is distinct from passed."""


@dataclass(frozen=True)
class StatusLane:
    """A lane: one original regime at one value of its uncontracted statuses."""

    regime: RegimeName
    """The original regime."""
    statuses: tuple[tuple[StateName, int], ...]
    """The (status, code) pairs of the lane; contracted statuses are absent."""


@dataclass(frozen=True)
class RegimeBlock:
    """A regime of a layout: a group of lanes over an interval of its periods."""

    regimes: tuple[RegimeName, ...]
    """Original economic regimes dispatched within this block."""
    periods: tuple[int, ...]
    """Original computational periods covered by the block."""
    dates: tuple[str, ...]
    """Exact age labels, or period labels in a period-based model."""
    states: tuple[StateName, ...]
    """State coordinates the block keeps."""
    actions: tuple[ActionName, ...]
    """Action coordinates the block keeps."""
    cells: tuple[str, ...]
    """Original dated-cell identifiers."""
    lanes: tuple[StatusLane, ...]
    """The lanes the block holds."""
    cost: RegimeCost
    """The block's evaluations, stored values and one regime."""


@dataclass(frozen=True)
class RegimeCandidate:
    """A layout of the curve, or the original model, with its checks."""

    identifier: str
    """`max_blocks=<k>` for a curve point, or retain-original."""
    max_blocks: int | None
    """The block budget k of the curve point, or None for the original."""
    cost: RegimeCost | None
    """Its evaluations, stored values and number of regimes."""
    blocks: tuple[RegimeBlock, ...]
    """Directly inspectable computational blocks."""
    checks: tuple[RegimeCheck, ...]
    """The checks the layout passed through."""
    reasons: tuple[str, ...]
    """Reasons supporting or qualifying the layout."""


@dataclass(frozen=True)
class RegimeAnalysisResult:
    """The layout curve of a model and the layout the cost model recommends.

    The data view is recursively immutable. Use to_dict for an independent,
    serializable copy. The recommendation is one of the candidates, or the
    original when no layout improves on it.
    """

    status: str
    """Analysis outcome, with detailed scope in checks and data."""
    original: RegimeCandidate
    """The original representation and its costs."""
    candidates: tuple[RegimeCandidate, ...]
    """One layout per block budget k: the at-most-k minimum of the curve."""
    recommendation: RegimeCandidate
    """The candidate the cost model selects, or the original."""
    checks: tuple[RegimeCheck, ...]
    """Global check layers."""
    cost_model: LinearRegimeCostModel
    """The objective the recommendation minimizes."""
    facts_backend: FactsBackend
    """The backend that supplied the dependency facts."""
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


def _integer(value: JSONValue) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = "Structural costs must be nonnegative integers."
        raise ValueError(msg)
    return value


def _sequence(value: JSONValue) -> Sequence[JSONValue]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = "Analysis rows must be a sequence."
        raise TypeError(msg)
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
    by_place = {
        (_integer(cell["period"]), str(cell["regime"])): identifier
        for identifier, cell in cells.items()
    }

    def date(identifier: str) -> str:
        cell = cells[identifier]
        return str(cell.get("age") if cell.get("age") is not None else cell["period"])

    def layout_blocks(
        layout: Sequence[Mapping[str, JSONValue]],
    ) -> tuple[RegimeBlock, ...]:
        result = []
        for group in layout:
            lanes = tuple(
                StatusLane(
                    regime=str(lane["regime"]),
                    statuses=tuple(
                        (str(n), _integer(c))
                        for n, c in _mapping(lane["statuses"]).items()
                    ),
                )
                for lane in _rows(group["lanes"])
            )
            for block in _rows(group["blocks"]):
                fibres = _rows(block["fibres"])
                identifiers = tuple(
                    dict.fromkeys(
                        by_place[_integer(f["period"]), str(f["regime"])]
                        for f in fibres
                    )
                )
                regimes = tuple(sorted({str(f["regime"]) for f in fibres}))
                result.append(
                    RegimeBlock(
                        regimes=regimes,
                        periods=tuple(_integer(t) for t in _sequence(block["periods"])),
                        dates=tuple(dict.fromkeys(date(i) for i in identifiers)),
                        states=tuple(
                            n for n in _mapping(block["states"]) if n != "regime"
                        ),
                        actions=tuple(_mapping(block["actions"])),
                        cells=identifiers,
                        lanes=tuple(lane for lane in lanes if lane.regime in regimes),
                        cost=RegimeCost(
                            work=_integer(block["evaluations"]),
                            storage=_integer(block["storage"]),
                            complexity=1,
                        ),
                    )
                )
        return tuple(result)

    original_cost = report.get("original_cost")
    groups: dict[str, list[str]] = {}
    for identifier, cell in cells.items():
        groups.setdefault(str(cell["regime"]), []).append(identifier)
    original = RegimeCandidate(
        "retain-original",
        None,
        None
        if original_cost is None
        else RegimeCost(
            work=_integer(_mapping(original_cost)["work"]),
            storage=_integer(_mapping(original_cost)["storage"]),
            complexity=_integer(_mapping(original_cost)["clarity"]),
        ),
        tuple(
            RegimeBlock(
                regimes=(regime,),
                periods=tuple(
                    sorted({_integer(cells[i]["period"]) for i in identifiers})
                ),
                dates=tuple(dict.fromkeys(date(i) for i in identifiers)),
                states=tuple(
                    dict.fromkeys(
                        n for i in identifiers for n in _mapping(cells[i]["states"])
                    )
                ),
                actions=tuple(
                    dict.fromkeys(
                        n for i in identifiers for n in _mapping(cells[i]["actions"])
                    )
                ),
                cells=tuple(identifiers),
                lanes=(),
                cost=RegimeCost(work=0, storage=0, complexity=1),
            )
            for regime, identifiers in groups.items()
        ),
        (RegimeCheck("preservation", "IDENTITY"),),
        ("Retain the supplied model unchanged.",),
    )
    layouts = report.get("layouts")
    points = _rows(_mapping(layouts).get("curve", [])) if layouts is not None else ()
    checks = (
        RegimeCheck(
            "source_semantics",
            str(_mapping(report.get("semantic_check", {})).get("status", "NOT_RUN")),
        ),
        RegimeCheck(
            "layout_check",
            str(_mapping(report.get("layout_check", {})).get("status", "NOT_RUN")),
        ),
    )
    candidates = tuple(
        RegimeCandidate(
            f"max_blocks={_integer(point['max_blocks'])}",
            _integer(point["max_blocks"]),
            RegimeCost(
                work=_integer(point["evaluations"]),
                storage=_integer(point["storage"]),
                complexity=_integer(point["blocks"]),
            ),
            layout_blocks(_rows(point["layout"])),
            checks,
            (f"Fewest evaluations with at most {point['max_blocks']} regimes.",),
        )
        for point in points
    )
    chosen = _mapping(report.get("recommendation", {}))
    recommendation = (
        candidates[_integer(chosen["curve_index"])]
        if chosen.get("operation") == "layout"
        else original
    )
    return RegimeAnalysisResult(
        status=str(report.get("status", "UNRESOLVED")),
        original=original,
        candidates=candidates,
        recommendation=recommendation,
        checks=tuple(
            RegimeCheck(key, str(value))
            for key, value in _mapping(report.get("layers", {})).items()
        ),
        cost_model=cost_model,
        facts_backend=cast("FactsBackend", report["facts_backend"]),
        data=report,
    )
