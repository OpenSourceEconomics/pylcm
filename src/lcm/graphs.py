"""Inspect a constructed model's temporal regime graph and check its structure."""

from collections.abc import Iterable, Mapping
from pathlib import Path

from networkx import DiGraph

from _lcm.time import landing_step
from _lcm.typing import JSONValue
from _lcm_regime_finder.layout_search import analyze_layouts as _analyze_layouts
from lcm.model import Model
from lcm.regime_analysis import (
    FactsBackend,
    LinearRegimeCostModel,
    RegimeAnalysisResult,
    RegimeBlock,
    RegimeCandidate,
    RegimeCheck,
    RegimeCost,
    StatusLane,
)
from lcm.regime_analysis import result_from_report as _result_from_report
from lcm.typing import Phase

__all__ = [
    "LinearRegimeCostModel",
    "RegimeAnalysisResult",
    "RegimeBlock",
    "RegimeCandidate",
    "RegimeCheck",
    "RegimeCost",
    "StatusLane",
    "analyze_regime_structure",
    "digraph",
    "find_regimes",
    "layout_curve",
]


# keyword-only-exempt: primary-argument=model
def analyze_regime_structure(
    model: Model,
    *,
    facts_backend: FactsBackend,
    cost_model: LinearRegimeCostModel | None = None,
) -> RegimeAnalysisResult:
    """Search the model's layout curve and recommend one point by a linear cost.

    For every block budget k the curve holds a layout of the least evaluations
    with at most k regimes. The recommendation is the curve point that minimizes
    the cost model, or the original model. No report is written; call
    result.write_report(directory) to save one.

    Args:
        model: The constructed model to analyze.
        facts_backend: Where the dependency facts come from. `"metadata"` reads
            the declared scalar expressions and is conservative; `"dense"` and
            `"implicit"` interpret the captured source exactly and raise
            `FactsBudgetError` when they run out of budget. `"implicit"` stores
            per-status facts as decision diagrams.
        cost_model: The linear objective over evaluations, stored values and
            regimes; the default weighs them equally.

    Returns:
        The typed analysis result.

    """
    from _lcm_regime_finder import find_regimes as analyze  # noqa: PLC0415

    selected_cost = LinearRegimeCostModel() if cost_model is None else cost_model
    report = analyze(
        model,
        facts_backend=facts_backend,
        recommendation=selected_cost.as_recommendation(),
    )
    report["cost_model"] = selected_cost.as_recommendation()
    return _result_from_report(report, cost_model=selected_cost)


# keyword-only-exempt: primary-argument=model
def digraph(model: Model, *, phase: Phase | None = None) -> DiGraph:
    """Return an independent view of effective next-period regime transitions.

    Nodes are `(age, regime_name)` pairs using exact age labels, or
    `(period, regime_name)` in a period-based model. The selected construction
    graph owns node admission, including isolated and terminal nodes.

    `phase=None` requires matching solve/simulate admission, effective support,
    declared edges and pruning reasons. Economic functions may differ: both
    phases remain in `graph.graph["declarations"]`, and no function is evaluated.
    Edge `law` metadata retains the complete original bound source law, including
    any phase variants. Same-period exits and value reads remain declaration
    metadata, not temporal edges.

    Graph and attribute containers are independent and editable. Declaration,
    callable and grid leaves are shared originals and obey the model's stability
    contract. Conversion neither prepares execution nor rebuilds factories.

    Args:
        model: The constructed model to inspect.
        phase: Select `"solve"` or `"simulate"`, or require common topology.

    Returns:
        A NetworkX directed graph with the complete clock and declaration view.

    Raises:
        ValueError: If the phase is ambiguous or a temporal endpoint is absent
            from the constructed graph's admitted nodes.

    """
    declarations = model.get_declarations()
    graph = declarations.graph
    if phase is None and (
        graph.solution != graph.simulation
        or graph.edges.solve != graph.edges.simulate
        or graph.pruned_edges["solve"] != graph.pruned_edges["simulate"]
    ):
        msg = (
            "The solve and simulate regime graphs differ. "
            'Pass phase="solve" or phase="simulate" explicitly.'
        )
        raise ValueError(msg)

    selected_phase = "solve" if phase is None else phase
    temporal = graph.solution if selected_phase == "solve" else graph.simulation
    clock = model._time.exact_values  # noqa: SLF001
    result = DiGraph(
        phase=phase,
        coordinate_kind=graph.coordinate_kind,
        clock=clock,
        declarations=declarations,
        declared_edges=graph.edges,
        pruned_edges=graph.pruned_edges[selected_phase],
    )
    for period, regime_names in enumerate(temporal.active_regimes_by_period):
        for regime_name in sorted(regime_names):
            result.add_node(
                (clock[period], regime_name),
                period=period,
                declaration=declarations.regimes_by_period[period][regime_name],
            )
    for period, targets_by_source in enumerate(temporal.targets_by_period):
        for source in sorted(targets_by_source):
            for target in sorted(targets_by_source[source]):
                source_node = (clock[period], source)
                target_node = (
                    clock[landing_step(ages=model._time, step=period, target=target)],  # noqa: SLF001
                    target,
                )
                # NetworkX otherwise silently admits a missing endpoint.
                if source_node not in result or target_node not in result:
                    msg = "A temporal edge has an endpoint outside the admitted nodes."
                    raise ValueError(msg)
                result.add_edge(
                    source_node,
                    target_node,
                    law=graph.laws[source],
                    status=temporal.edge_status(
                        period=period, source=source, target=target
                    ),
                )
    return result


# keyword-only-exempt: primary-argument=model
def find_regimes(
    model: Model,
    *,
    facts_backend: FactsBackend,
    output: Path | None = None,
) -> dict[str, JSONValue]:
    """Search and check the layout curve of a constructed model.

    The search uses the original solve and simulate declarations and grid sizes.
    It does not solve or simulate the model.

    Args:
        model: The constructed model to analyze.
        facts_backend: Where the dependency facts come from. `"metadata"` reads
            the declared scalar expressions and is conservative; `"dense"` and
            `"implicit"` interpret the captured source exactly and raise
            `FactsBudgetError` when they run out of budget. `"implicit"` stores
            per-status facts as decision diagrams.
        output: Optional new directory for the HTML and JSON reports.

    Returns:
        The analysis report: the layout curve, its check and the recommendation.

    Raises:
        FileExistsError: If the requested report directory already exists.

    """
    # Import the analysis engine only when explicitly requested.
    from _lcm_regime_finder import find_regimes as analyze  # noqa: PLC0415

    report = analyze(model, facts_backend=facts_backend)
    if output is not None:
        from _lcm_regime_finder.production_render import (  # noqa: PLC0415
            write_production_report,
        )

        output.mkdir(parents=True)
        write_production_report(report, output)
    return report


# keyword-only-exempt: primary-argument=model
def layout_curve(
    model: Model,
    *,
    facts_backend: FactsBackend,
    limits: Mapping[str, JSONValue] | None = None,
    price: Iterable[dict[str, JSONValue]] = (),
) -> dict[str, JSONValue]:
    """Run the Supplement's layout search alone and return its layout curve.

    There is no source check, no layout check and no recommendation; use
    `analyze_regime_structure` or `find_regimes` for those. The model is not
    solved or simulated.

    Args:
        model: The constructed model to analyze.
        facts_backend: Where the dependency facts come from. `"metadata"` reads
            the declared scalar expressions and is conservative; `"dense"` and
            `"implicit"` interpret the captured source exactly and raise
            `FactsBudgetError` when they run out of budget. `"implicit"` stores
            per-status facts as decision diagrams.
        limits: Optional search limits, such as `seconds` or `max_blocks`.
        price: Given blocks to price, each as `regimes`, `periods` and optional
            `statuses`.

    Returns:
        The `supplement-layout-curve/v1` record with its phase timings.

    """
    return _analyze_layouts(
        model, facts_backend=facts_backend, limits=limits, price=price
    )
