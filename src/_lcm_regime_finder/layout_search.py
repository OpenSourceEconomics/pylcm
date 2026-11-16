"""The layout search of the Supplement (lcm-paper, paper/sections/supplement-choice.tex).

Line numbers refer to that file at lcm-paper d28955bd.
"""

from collections import defaultdict
from collections.abc import Mapping

from .frontier import canonical
from .facts import Record


def regime_kinds(declarations: Record) -> dict[str, Record]:
    """Return each original regime's pooling attributes.

    A regime is terminal if none of its cells has a successor. Its kind is its
    stage, solver and operators (aggregator, certainty equivalent, Pareto
    objective); its law is its taste-shock declaration. It has a discrete choice
    if it has a categorical action coordinate.
    """
    domains = declarations["domains"]
    kinds: dict[str, Record] = {}
    for cell in declarations["cells"]:
        operators = cell.get("operators", {})
        key = canonical([cell.get("stage"), cell.get("solver_identity"),
                         {k: v for k, v in operators.items() if k != "taste_shocks"}])
        record = kinds.setdefault(cell["regime"], {
            "terminal": True, "key": key, "taste_shocks": operators.get("taste_shocks"),
            "discrete_choice": False})
        if record["key"] != key:
            raise ValueError(f"Regime {cell['regime']} changes its kind between periods")
        record["terminal"] = record["terminal"] and cell["terminal"]
        record["discrete_choice"] = record["discrete_choice"] or any(
            domains[did].get("kind") == "categorical" for did in cell["actions"].values())
    return kinds


def dense_regimes(kinds: Mapping[str, Record]) -> list[tuple[str, ...]]:
    """Pool the nonterminal regimes of each kind by Def. "Dense model" (:126-146).

    (i) Regimes with equal laws pool. Only the degenerate law (no taste shocks)
    is shown equal across regimes: a shocked regime's scale is a parameter of its
    own. (ii) A shocked regime always has a discrete action, so no regime changes
    its law. (iii) If the degenerate regimes are without discrete choice and
    exactly one shocked regime of the kind exists, the two pool. Terminal regimes
    stay regimes of their own (:128, :131).
    """
    groups: list[tuple[str, ...]] = []
    by_kind: dict[str, list[str]] = defaultdict(list)
    for name, kind in kinds.items():
        if kind["terminal"]:
            groups.append((name,))
        else:
            by_kind[kind["key"]].append(name)
    for names in by_kind.values():
        degenerate = sorted(n for n in names if kinds[n]["taste_shocks"] is None)
        shocked = sorted(n for n in names if kinds[n]["taste_shocks"] is not None)
        if degenerate and len(shocked) == 1 and not any(kinds[n]["discrete_choice"] for n in degenerate):
            groups.append(tuple(sorted([*degenerate, *shocked])))
            continue
        if degenerate:
            groups.append(tuple(degenerate))
        groups.extend((n,) for n in shocked)
    return sorted(groups)
