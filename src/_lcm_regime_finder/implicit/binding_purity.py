"""Sufficient source-only proof of reference binding-cache order independence.

This inspects data, never a callback. It does not translate primitive facts.
An unavailable proof selects original-order streaming, not weaker read facts.
"""

from collections.abc import Callable, Mapping
from ..records import JSONValue


def certify_binding_cache(
    *,
    callbacks: Mapping[str, JSONValue],
    records: Mapping[str, JSONValue],
    resolve: Callable[[JSONValue], JSONValue],
) -> dict[str, JSONValue]:
    roots = []
    pending: list[JSONValue] = [callbacks, records]
    seen = set()
    while pending:
        value = pending.pop()
        if isinstance(value, (Mapping, tuple, list)):
            if id(value) in seen:
                continue
            seen.add(id(value))
        if isinstance(value, Mapping):
            for slot in (
                "bindings",
                "partial_bindings",
                "partial_keywords",
                "defaults",
            ):
                container = value.get(slot)
                if isinstance(container, Mapping):
                    roots.extend(container.values())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)

    memo = {}

    def measure(item: JSONValue, active: frozenset[str]) -> tuple[int, int]:
        if isinstance(item, Mapping) and item.get("kind") == "callable-reference":
            ref = item.get("capture_ref")
            if not isinstance(ref, str):
                raise ValueError("Invalid binding reference")
            target = resolve(item)
            if not isinstance(target, Mapping):
                raise ValueError("unresolved/cyclic binding")
            if target.get("kind") != "binding-record":
                # A valid reference to a callable is not scalar binding data.
                # The reference rejects it before traversing its payload.
                return 0, 1
            if target.get("capture_status", "COMPLETE") != "COMPLETE" or ref in active:
                raise ValueError("unresolved/cyclic binding")
            if ref not in memo:
                depth, size = measure(target.get("value"), active | {ref})
                memo[ref] = (1 + depth, size)
            return memo[ref]
        if isinstance(item, Mapping):
            if set(item) == {"fraction"}:
                return 0, 1
            if item.get("kind") == "array" and "values" in item:
                depth, size = measure(item["values"], active)
                return 1 + depth, size
            if item.get("kind") == "categorical-class":
                depth, size = measure(item.get("members", {}), active)
                return 1 + depth, size
            if item.get("kind") in {"callable", "primitive", "module", "unresolved"}:
                # The reference decoder raises here before looking at children.
                # This is an invariant failure, never a decoded scalar claim.
                # Treating it as a depth-zero terminal proves only cache-order
                # independence; the unchanged analyzer retains proved=False.
                return 0, 1
            children = [measure(child, active) for child in item.values()]
        elif isinstance(item, (list, tuple)):
            children = [measure(child, active) for child in item]
        else:
            if isinstance(item, str) and len(item.encode("utf-8")) > 65536:
                raise ValueError("string budget")
            return 0, 1
        depth = 1 + max((depth for depth, _ in children), default=-1)
        size = 1 + sum(size for _, size in children)
        if size > 8192:
            raise ValueError("logical scalar size budget")
        return depth, size

    depths = []
    try:
        for root in roots:
            depth, size = measure(root, frozenset())
            if depth > 64 or size > 8192:
                raise ValueError("expanded binding depth or logical size budget")
            depths.append(depth)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        return {
            "proved": False,
            "method": "all-potential-binding-roots-expanded-depth/v1",
            "root_count": len(roots),
            "reason": str(error),
        }
    return {
        "proved": True,
        "method": "all-potential-binding-roots-expanded-depth/v1",
        "root_count": len(roots),
        "max_expanded_depth": max(depths, default=0),
    }
