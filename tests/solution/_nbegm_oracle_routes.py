"""Route identities for the direct-oracle census.

The direct scalar oracle checks the ride-along NB-EGM kernel one route at a time.
A route is identified by what makes the kernel take a different production path:
the test-model module, the semantic model-builder keywords, the semantic
parameter-builder keywords, the regime, the period, and the context the kernel is
reached in. Grid sizes only bound the test and are not part of the identity.

`SUPPORTED_ROUTES` is the explicit manifest of routes the oracle must cover;
`declared_route_identities` derives the identities the oracle test actually
declares from its source, and `census_discrepancies` compares the two. The
manifest and the derivation are deliberately separate, so deleting, duplicating,
or silently re-flagging a route in either place is a census failure.
"""

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

# A plain ride-along NB-EGM kernel reached through `ride_along_kernel`.
RIDE_ALONG = "ride_along"

# The keeper and adjuster kernels of a nested NNBEGM solve.
NNBEGM_INNER = "nnbegm_keeper_and_adjuster"

# Builder keywords that bound the test's grids and carry no route identity.
GRID_SIZE_KEYWORDS = frozenset(
    {
        "n_liquid",
        "n_savings",
        "n_consumption",
        "n_wage",
        "liquid_max",
        "savings_max",
        "wage_max",
    }
)


@dataclass(frozen=True, kw_only=True)
class RouteIdentity:
    """What distinguishes one direct-oracle route from every other."""

    module: str
    """The test-model module the route builds its model from."""
    model_kwargs: tuple[tuple[str, object], ...]
    """Sorted semantic keywords of the model builder call."""
    params_kwargs: tuple[tuple[str, object], ...]
    """Sorted semantic keywords of the parameter builder call, or its literal."""
    regime_name: str = "alive"
    """The regime whose kernel is checked."""
    period: int | None = None
    """The period whose kernel is checked, `None` for the route's default."""
    context: str = RIDE_ALONG
    """How the kernel is reached."""


def _kw(**kwargs: object) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(kwargs.items()))


def _ride(
    *,
    module: str,
    params: tuple[tuple[str, object], ...] = (),
    regime_name: str = "alive",
    period: int | None = None,
    **model_kwargs: object,
) -> RouteIdentity:
    return RouteIdentity(
        module=module,
        model_kwargs=_kw(variant="nbegm", n_periods=3, **model_kwargs),
        params_kwargs=params,
        regime_name=regime_name,
        period=period,
    )


_RIDE_DISCRETE = "nbegm_ride_discrete_toy"

# The routes the direct oracle must cover, by route name.
SUPPORTED_ROUTES: Mapping[str, RouteIdentity] = MappingProxyType(
    {
        "ride_along": _ride(module="nbegm_ride_along_toy"),
        "ride_along_per_kind_crra": _ride(
            module="nbegm_ride_along_toy",
            per_kind_crra=True,
            params=_kw(per_kind_crra=True),
        ),
        "ride_along_per_kind_discount": _ride(
            module="nbegm_ride_along_toy",
            per_kind_discount=True,
            params=_kw(per_kind_discount=True),
        ),
        "ride_along_distributed_kind": _ride(
            module="nbegm_ride_along_toy", distributed_kind=True
        ),
        "derived_var": _ride(module="nbegm_derived_var_toy"),
        "multi_source": _ride(module="nbegm_multi_source_toy"),
        "multi_target": RouteIdentity(
            module="nbegm_multi_target_toy",
            model_kwargs=_kw(variant="nbegm", n_periods=4),
            params_kwargs=(),
            regime_name="alive_a",
            period=1,
        ),
        "stochastic_node_kink": _ride(
            module="nbegm_stochastic_node_toy", tax_kind="kink", period=0
        ),
        "stochastic_node_kink_with_kind": _ride(
            module="nbegm_stochastic_node_toy",
            tax_kind="kink",
            with_kind=True,
            params=_kw(with_kind=True),
            period=0,
        ),
        "brute_child": RouteIdentity(
            module="nbegm_brute_child_toy",
            model_kwargs=_kw(young_variant="nbegm"),
            params_kwargs=(),
            regime_name="young",
            period=0,
        ),
        "ces_utility_kink": _ride(
            module="nbegm_ces_utility_toy",
            breakpoint_kind="continuous_kink",
            params=_kw(breakpoint_kind="continuous_kink"),
        ),
        "ces_utility_jump": _ride(
            module="nbegm_ces_utility_toy",
            breakpoint_kind="jump",
            params=_kw(breakpoint_kind="jump"),
        ),
        "jump_ride_along": _ride(module="nbegm_jump_ride_along_toy"),
        "jump_ride_along_bridged": _ride(
            module="nbegm_jump_ride_along_toy", jump_read="bridged"
        ),
        "continuous_ride_along": _ride(module="nbegm_continuous_ride_along_toy"),
        "indexed_threshold": _ride(module="nbegm_indexed_threshold_toy"),
        "mappingleaf_threshold": _ride(module="nbegm_mappingleaf_threshold_toy"),
        "multi_source_jump": _ride(module="nbegm_multi_source_jump_toy"),
        "stochastic_node_jump": _ride(
            module="nbegm_stochastic_node_toy",
            tax_kind="jump",
            params=_kw(tax_lump=1.0),
            period=0,
        ),
        "ride_discrete": _ride(module=_RIDE_DISCRETE),
        "ride_discrete_action_in_costate": _ride(
            module=_RIDE_DISCRETE, action_in_costate=True
        ),
        "ride_discrete_action_in_utility": _ride(
            module=_RIDE_DISCRETE, action_in_utility=True
        ),
        "ride_discrete_action_in_regime_transition": _ride(
            module=_RIDE_DISCRETE, action_in_regime_transition=True
        ),
        "ride_discrete_jump_schedule": _ride(
            module=_RIDE_DISCRETE, jump_schedule=True, params=_kw(jump_schedule=True)
        ),
        "ride_discrete_action_in_liquid_law": _ride(
            module=_RIDE_DISCRETE,
            action_in_liquid_law=True,
            params=_kw(action_in_liquid_law=True),
        ),
        "ride_discrete_action_in_schedule_variable": _ride(
            module=_RIDE_DISCRETE,
            action_in_schedule_variable=True,
            params=_kw(action_in_schedule_variable=True),
        ),
        "ride_discrete_costate_reads_liquid_piecewise": _ride(
            module=_RIDE_DISCRETE, costate_reads_liquid=True, costate_smooth=False
        ),
        "ride_discrete_transition_reads_liquid": _ride(
            module=_RIDE_DISCRETE, transition_reads_liquid=True
        ),
        "ride_discrete_schedule_variable_with_interval_continuation": _ride(
            module=_RIDE_DISCRETE,
            action_in_schedule_variable=True,
            costate_reads_liquid=True,
            params=_kw(action_in_schedule_variable=True),
        ),
        "ride_discrete_action_in_costate_with_jump_schedule": _ride(
            module=_RIDE_DISCRETE,
            action_in_costate=True,
            jump_schedule=True,
            params=_kw(jump_schedule=True),
        ),
        "ride_discrete_action_in_health_transition": _ride(
            module=_RIDE_DISCRETE, action_in_health_transition=True, period=0
        ),
        "ride_discrete_action_in_discount": _ride(
            module=_RIDE_DISCRETE,
            action_in_discount=True,
            params=_kw(action_in_discount=True),
            period=0,
        ),
        "ride_discrete_action_in_all_channels": _ride(
            module=_RIDE_DISCRETE,
            action_in_costate=True,
            action_in_liquid_law=True,
            action_in_utility=True,
            params=_kw(action_in_liquid_law=True),
        ),
        "multi_discrete": _ride(
            module="nbegm_multi_discrete_toy", n_actions=2, params=_kw(n_actions=2)
        ),
        "next_asset_cliff": _ride(module="nbegm_next_asset_cliff_toy"),
        "epstein_zin": RouteIdentity(
            module="epstein_zin_model",
            model_kwargs=_kw(
                solver=(
                    "NBEGM(savings_grid=epstein_zin_model._SAVINGS_GRID, "
                    "envelope_arithmetic='ordinary')"
                )
            ),
            params_kwargs=(("value", "epstein_zin_model._PARAMS"),),
            period=0,
        ),
        "n_nbegm": RouteIdentity(
            module="n_nbegm_toy",
            model_kwargs=_kw(variant="n_nbegm"),
            params_kwargs=(("value", "{'discount_factor': 0.95}"),),
            context=NNBEGM_INNER,
        ),
        "n_nbegm_discrete": RouteIdentity(
            module="n_nbegm_discrete_toy",
            model_kwargs=_kw(variant="n_nbegm"),
            params_kwargs=(
                ("value", "{'discount_factor': 0.95, 'alive': {'premium': 1.0}}"),
            ),
            context=NNBEGM_INNER,
        ),
    }
)


# Production-path tests of the ride-discrete toy and the flags each one builds:
# the test module, the function in it that builds the model, and the semantic
# flags that function passes. The census requires a supported ride-discrete route
# with exactly those flags.
POSITIVE_WITNESSES: tuple[tuple[str, str, Mapping[str, object]], ...] = (
    (
        "tests/solution/test_nbegm_action_in_liquid_law_agreement.py",
        "_solve",
        MappingProxyType({"action_in_liquid_law": True}),
    ),
    (
        "tests/solution/test_nbegm_action_in_schedule_variable_agreement.py",
        "_solve",
        MappingProxyType({"action_in_schedule_variable": True}),
    ),
    (
        "tests/solution/test_nbegm_interval_constant_continuation.py",
        "test_costate_law_piecewise_constant_in_liquid_builds",
        MappingProxyType({"costate_reads_liquid": True, "costate_smooth": False}),
    ),
    (
        "tests/solution/test_nbegm_transition_reads_liquid_agreement.py",
        "_solve",
        MappingProxyType({"transition_reads_liquid": True}),
    ),
    (
        "tests/solution/test_nbegm_schedule_variable_interval_agreement.py",
        "_solve",
        MappingProxyType(
            {"action_in_schedule_variable": True, "costate_reads_liquid": True}
        ),
    ),
    (
        "tests/solution/test_nbegm_action_costate_jump_agreement.py",
        "_solve",
        MappingProxyType({"action_in_costate": True, "jump_schedule": True}),
    ),
    (
        "tests/solution/test_nbegm_action_all_channels_agreement.py",
        "_solve",
        MappingProxyType(
            {
                "action_in_costate": True,
                "action_in_liquid_law": True,
                "action_in_utility": True,
            }
        ),
    ),
    (
        "tests/solution/test_nbegm_action_in_costate_agreement.py",
        "_solve",
        MappingProxyType({"action_in_costate": True}),
    ),
    (
        "tests/solution/test_nbegm_action_in_utility_agreement.py",
        "_solve",
        MappingProxyType({"action_in_utility": True}),
    ),
    (
        "tests/solution/test_nbegm_action_in_regime_transition_agreement.py",
        "_solve",
        MappingProxyType({"action_in_regime_transition": True}),
    ),
)


def declared_route_identities(
    *, source: str, table_name: str, context: str
) -> dict[str, RouteIdentity]:
    """Derive the route identities a route table in `source` declares.

    The table is a module-level tuple of `_Route(...)` calls whose `build_model`
    is a lambda calling `<module>.build_model(...)` or `<module>._build_model(...)`
    with literal keywords, and whose `build_params` is either a bare attribute,
    a lambda calling `<module>.build_params(...)`, or a lambda returning a literal.

    Raises `TypeError` on a route whose builder is not statically visible and
    `ValueError` on a duplicate route name, so a table the census cannot read is a
    failure rather than an empty census.
    """
    tree = ast.parse(source)
    table = _module_level_tuple(tree=tree, name=table_name)
    identities: dict[str, RouteIdentity] = {}
    for element in table.elts:
        if not isinstance(element, ast.Call):
            msg = f"{table_name} holds a non-call element at line {element.lineno}."
            raise TypeError(msg)
        keywords = {kw.arg: kw.value for kw in element.keywords if kw.arg}
        name = _literal(keywords["name"])
        if not isinstance(name, str):
            msg = f"{table_name} route at line {element.lineno} has no literal name."
            raise TypeError(msg)
        if name in identities:
            msg = f"{table_name} declares route {name!r} twice."
            raise ValueError(msg)
        module, model_kwargs = _builder_call(
            expression=keywords["build_model"],
            attributes={"build_model", "_build_model"},
        )
        params_module, params_kwargs = _params_declaration(keywords["build_params"])
        if params_module not in {None, module}:
            msg = (
                f"route {name!r} builds its params from {params_module!r}, "
                f"not from {module!r}."
            )
            raise ValueError(msg)
        identities[name] = RouteIdentity(
            module=module,
            model_kwargs=model_kwargs,
            params_kwargs=params_kwargs,
            regime_name=cast(
                "str",
                _literal(keywords["regime_name"])
                if "regime_name" in keywords
                else "alive",
            ),
            period=cast(
                "int | None",
                _literal(keywords["period"]) if "period" in keywords else None,
            ),
            context=context,
        )
    return identities


def census_discrepancies(
    *, declared: Mapping[str, RouteIdentity], supported: Mapping[str, RouteIdentity]
) -> tuple[str, ...]:
    """Return every way `declared` fails to be exactly the supported manifest.

    A route is missing when a supported name has no declaration, unsupported when
    a declared name is not in the manifest, drifted when the two identities under
    one name differ, and duplicated when two declared names share one identity.
    """
    discrepancies = [
        f"missing route {name!r}: {supported[name]}"
        for name in sorted(set(supported) - set(declared))
    ]
    discrepancies.extend(
        f"unsupported route {name!r}: {declared[name]}"
        for name in sorted(set(declared) - set(supported))
    )
    discrepancies.extend(
        f"drifted route {name!r}: declared {declared[name]}, "
        f"supported {supported[name]}"
        for name in sorted(set(declared) & set(supported))
        if declared[name] != supported[name]
    )
    by_identity: dict[RouteIdentity, list[str]] = {}
    for name, identity in declared.items():
        by_identity.setdefault(identity, []).append(name)
    for identity, names in by_identity.items():
        if len(names) > 1:
            discrepancies.append(f"duplicate identity {identity} under {sorted(names)}")
    return tuple(discrepancies)


def witness_flags(*, source: str, function: str) -> dict[str, object]:
    """Return the semantic flags the model-builder call in `function` passes.

    The function must contain exactly one `<module>.build_model(...)` call.
    """
    tree = ast.parse(source)
    calls = [
        node
        for definition in ast.walk(tree)
        if isinstance(definition, ast.FunctionDef) and definition.name == function
        for node in ast.walk(definition)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_model"
    ]
    if len(calls) != 1:
        msg = f"{function} holds {len(calls)} build_model calls, not one."
        raise ValueError(msg)
    return dict(_semantic_keywords(calls[0]))


def _module_level_tuple(*, tree: ast.Module, name: str) -> ast.Tuple:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            if not isinstance(node.value, ast.Tuple):
                msg = f"{name} is not a literal tuple."
                raise TypeError(msg)
            return node.value
    msg = f"no module-level assignment to {name}."
    raise ValueError(msg)


def _builder_call(
    *, expression: ast.expr, attributes: set[str]
) -> tuple[str, tuple[tuple[str, object], ...]]:
    body = expression.body if isinstance(expression, ast.Lambda) else expression
    if not (
        isinstance(body, ast.Call)
        and isinstance(body.func, ast.Attribute)
        and body.func.attr in attributes
        and isinstance(body.func.value, ast.Name)
    ):
        msg = f"no statically visible builder call at line {expression.lineno}."
        raise TypeError(msg)
    return body.func.value.id, _semantic_keywords(body)


def _params_declaration(
    expression: ast.expr,
) -> tuple[str | None, tuple[tuple[str, object], ...]]:
    body = expression.body if isinstance(expression, ast.Lambda) else expression
    if isinstance(body, ast.Attribute) and body.attr == "build_params":
        owner = body.value
        return (owner.id if isinstance(owner, ast.Name) else None), ()
    if (
        isinstance(body, ast.Call)
        and isinstance(body.func, ast.Attribute)
        and body.func.attr == "build_params"
        and isinstance(body.func.value, ast.Name)
    ):
        return body.func.value.id, _semantic_keywords(body)
    return None, (("value", ast.unparse(body)),)


def _semantic_keywords(call: ast.Call) -> tuple[tuple[str, object], ...]:
    keywords = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in GRID_SIZE_KEYWORDS:
            continue
        keywords[keyword.arg] = _literal(keyword.value)
    return tuple(sorted(keywords.items()))


def _literal(node: ast.expr) -> object:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ast.unparse(node)
