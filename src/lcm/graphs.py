"""Inspect a constructed model's temporal regime graph and check its structure."""

from pathlib import Path

from networkx import DiGraph

from _lcm.typing import JSONValue
from lcm.model import Model
from lcm.typing import Phase


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
    clock = (
        tuple(range(model.n_periods)) if model.ages is None else model.ages.exact_values
    )
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
                target_node = (clock[period + 1], target)
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
def find_regimes(model: Model, *, output: Path | None = None) -> dict[str, JSONValue]:
    """Find and check candidate simplifications of a constructed model.

    The search uses the original solve and simulate declarations and grid sizes.
    It checks candidate changes against the captured model. It does not solve or
    simulate the model or establish a global optimum.

    Args:
        model: The constructed model to analyze.
        output: Optional new directory for the HTML and JSON reports.

    Returns:
        The analysis report, including recommendations and any unresolved checks.

    Raises:
        FileExistsError: If the requested report directory already exists.

    """
    # Import the analysis engine only when explicitly requested.
    from _lcm_regime_finder import find_regimes as analyze  # noqa: PLC0415

    report = analyze(model)
    if output is not None:
        from _lcm_regime_finder.production_render import (  # noqa: PLC0415
            write_production_report,
        )

        output.mkdir(parents=True)
        write_production_report(report, output)
    return report
