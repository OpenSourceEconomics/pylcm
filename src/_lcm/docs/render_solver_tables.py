"""Render solver capability tables without constructing or solving a model."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from lcm import LinSpacedGrid
from lcm.solvers import (
    DCEGM,
    EGM,
    NBEGM,
    NEGM,
    NNBEGM,
    FiniteOuterGrid,
    GridSearch,
    Solver,
)

_BEGIN = "<!-- capability tables: rendered, do not edit by hand -->"
_END = "<!-- end capability tables -->"


def render_solver_tables(*, solvers: Mapping[str, Solver]) -> str:
    """Return marked capability and execution-axis tables for these configurations."""
    descriptions = [
        [
            "Solver",
            "Required declaration",
            "Problem shape",
            "Hard prerequisites and supported constraints",
            "Main tradeoff",
        ]
    ]
    axes = [
        [
            "Solver",
            "Reduced axes",
            "Tiled axes",
            "Host axes",
            "Host-repeated programs",
            "Donation candidates",
            "EV1 taste shocks",
            "Nonlinear CE",
        ]
    ]
    for name, solver in solvers.items():
        cap = solver.capabilities
        descriptions.append(
            [
                f"`{name}`",
                cap.required_declaration,
                cap.problem_shape,
                cap.prerequisites,
                cap.main_tradeoff,
            ]
        )
        axes.append(
            [
                f"`{name}`",
                _names(cap.reduced_axes),
                _names(cap.tiled_axes),
                _names(cap.host_axes),
                _names(cap.host_driven_programs),
                _names(cap.donation_candidates),
                "Yes" if cap.supports_ev1_taste_shocks else "No",
                "Yes" if cap.supports_nonlinear_certainty_equivalent else "No",
            ]
        )
    return "\n\n".join(
        [_BEGIN, _table(descriptions), "## Execution axes", _table(axes), _END]
    )


def table_region(*, text: str) -> str:
    """Return the unique complete marked table region, refusing missing markers."""
    if text.count(_BEGIN) != 1 or text.count(_END) != 1:
        raise ValueError("Solver tables require exactly one begin and end marker.")
    start = text.index(_BEGIN)
    stop = text.index(_END)
    if stop < start:
        raise ValueError("Solver table end marker precedes its beginning.")
    return text[start : stop + len(_END)]


def default_solvers() -> Mapping[str, Solver]:
    """Construct representative public configurations without numerical kernels."""
    grid = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
    return {
        "GridSearch": GridSearch(),
        "EGM": EGM(savings_grid=grid),
        "DCEGM": DCEGM(savings_grid=grid),
        "NBEGM": NBEGM(savings_grid=grid),
        "NEGM": NEGM(inner=DCEGM(savings_grid=grid), outer_grid=grid),
        "NNBEGM": NNBEGM(
            inner=NBEGM(savings_grid=grid), outer_search=FiniteOuterGrid(grid=grid)
        ),
    }


def _names(names: tuple[str, ...]) -> str:
    return ", ".join(f"`{name}`" for name in names) or "—"


def _table(rows: Sequence[Sequence[str]]) -> str:
    escaped = [
        [cell.replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows
    ]
    widths = [
        max(3, *(len(cell) for cell in column)) for column in zip(*escaped, strict=True)
    ]
    lines = [
        "| "
        + " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        + " |"
        for row in escaped
    ]
    lines.insert(1, "| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(lines)


def main() -> None:
    """Print the rendered tables, or rewrite their checked-in region with --write."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    rendered = render_solver_tables(solvers=default_solvers())
    if arguments.write:
        path = Path(__file__).resolve().parents[3] / "docs/reference/solvers.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(table_region(text=text), rendered), encoding="utf-8"
        )
    else:
        print(rendered)  # noqa: T201


if __name__ == "__main__":
    main()
