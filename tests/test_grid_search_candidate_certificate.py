"""Certify that `GridSearch` evaluates every candidate the action grids declare.

The profile contract claims zero *search* error for both required profiles: the
default solver is exhaustive grid search, so no action combination
representable on the declared grids is left out of the maximization. That claim
is a property of the code, not of a run, and it is only worth what re-checks it
after the code changes — so it is stated here twice, in two instruments that
fail for different reasons.

**Structural.** The enumeration route is read off the syntax tree of
`_lcm.solution.grid_search` and `_lcm.regime_building.max_Q_over_a`: the whole
action-name tuple reaches `get_max_Q_over_a`, the whole action mapping reaches
the compiled core, the core product-maps over that tuple unbatched, and each
reduction covers every action axis. A line-number match would rot on the first
edit, so every obligation resolves a name or an attribute chain instead.

**Executable.** A tiny model is solved once per shape and then re-solved from
params alone, with each declared candidate made in turn the *only* feasible one
and, separately, the *unique* maximizer. A candidate the search never visits
publishes $-\\infty$ in the first sweep and a runner-up value in the second, so
the sweep is sensitive to a dropped grid point wherever it is dropped — in the
solver, in the product map, or upstream in the state-action space.

The singleton, collective, and taste-shock reductions are separate code paths and
all are swept. The taste-shock sweep uses candidate-distinct deterministic noise
at the production draw seam, so it checks the continuous winner, the noisy
discrete winner, the row-major flat index, and the returned unshocked value.
`test_dropping_one_candidate_is_visible_to_the_sweep` masks one cell inside the
action product map and shows the ordinary sweep going red on it, so a green sweep
is evidence rather than an assertion nothing ran.

**Universal transport, with an explicit semantic boundary.** Every coordinate
from an already-constructed, finalized concrete built-in action-grid object or
public runtime-points input is carried
to the pointwise `Q_and_F` call. Separately, the exact candidate arrays returned
by `Q_and_F` are followed route by route into the full reducers and publication.
The standard-library AST
proof rejects any extra assignment, branch, slice, index, mask, `where`, rank,
gap, support-size, shape, or axis transformation. The collective path permits
only the independently checked split of the trailing stakeholder axis and pins
the shared scalarization, argmax, and gather implementations exactly. The two
taste-shock routes additionally pin masking, continuous reduction, logsum,
per-discrete-cell mean-zero Gumbel noise, and flat-index reconstruction, including
the deep shared helper bodies. The construction and correctness of Q/F values
and feasibility—including user DAGs, constraints, transitions, interpolation,
continuation arithmetic, and folded-process weights—is outside this certificate
and remains owned by its dedicated semantic tests.

The executable sweep remains deliberately finite: two action grids, six
candidates, six strict winner orderings, and all 63 nonempty feasibility masks on
the ordinary and collective routes; the taste-shock routes use singleton,
intermediate, spread, and full supports for every winning candidate. It is an
executable sanity check, not the universal proof. The modeller's grid
specification and off-grid discretization error remain outside the claim.
Preserving every candidate produced by the sealed built-in materializers or
supplied through the runtime-points API, and searching all of them, is inside it.
"""

import ast
import functools
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.regime_building import max_Q_over_a as max_Q_over_a_module
from _lcm.simulation import compile as simulation_compile_module
from _lcm.simulation import simulate as simulation_module
from _lcm.solution import grid_search as grid_search_module
from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.collective import CollectiveUtility
from lcm.regime import Regime
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.candidate_certificate.direct_flow import (
    direct_flow_mutation_specs,
    verify_direct_candidate_flow,
)
from tests.candidate_certificate.generate_sources import (
    derive_source_paths,
    sha256_file,
)
from tests.candidate_certificate.verify import (
    nonempty_feasibility_masks,
    rank_vectors,
    reference_masked_argmax,
)
from tests.conftest import DECIMAL_PRECISION

_SRC_ROOT = Path(__file__).parent.parent / "src"

_SOURCE_INVENTORY_PATH = (
    Path(__file__).parent / "candidate_certificate" / "sources.json"
)


def _load_certified_sources() -> tuple[str, ...]:
    """Load the generated source inventory consumed by every certificate layer."""
    payload = json.loads(_SOURCE_INVENTORY_PATH.read_text(encoding="utf-8"))
    return tuple(item["path"] for item in payload["sources"])


#: Generated from the certificate's literal ``_parse`` obligations.
CERTIFIED_SOURCES: tuple[str, ...] = _load_certified_sources()

_WORK_VALUES: tuple[float, ...] = (0.0, 1.0)
_CONSUMPTION_VALUES: tuple[float, ...] = (1.0, 2.0, 3.0)
_WEALTH_VALUES = jnp.array([1.0, 2.0])
_CANDIDATES: tuple[tuple[float, float], ...] = tuple(
    (work, consumption) for work in _WORK_VALUES for consumption in _CONSUMPTION_VALUES
)
_NONEMPTY_FEASIBILITY_MASKS = nonempty_feasibility_masks(len(_CANDIDATES))
_RANK_VECTORS = rank_vectors(len(_CANDIDATES))
_MASK_PARAMETER_NAMES = tuple(f"feasible_{index}" for index in range(len(_CANDIDATES)))
_RANK_PARAMETER_NAMES = tuple(f"rank_{index}" for index in range(len(_CANDIDATES)))


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    working: ScalarInt


@categorical(ordered=False)
class RegimeId:
    acting: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class DedupRegimeId:
    left: ScalarInt
    right: ScalarInt
    done: ScalarInt


@categorical(ordered=False)
class FoldRegimeId:
    src: ScalarInt
    folded: ScalarInt
    dead: ScalarInt


def _parse(relative: str) -> ast.Module:
    """Parse one certified source into its syntax tree.

    Args:
        relative: Repo-relative path of the source to parse.

    Returns:
        The parsed module.
    """
    path = _SRC_ROOT.parent / relative
    # Named, not defaulted: house style puts literal UTF-8 in sources, and a
    # platform whose default codec is not UTF-8 cannot decode them.
    return ast.parse(path.read_text(encoding="utf-8"))


def _definition(*, tree: ast.Module, qualname: str) -> ast.AST:
    """Return the definition reached by a dotted `Class.method` or `function` path.

    Args:
        tree: Parsed module to search.
        qualname: Dotted path of the definition.

    Returns:
        The matching function or class definition node.
    """
    node: ast.AST = tree
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.ClassDef) and child.name == part:
                node = child
                break
        else:  # pragma: no cover - a missing definition is the failure itself
            msg = f"{qualname!r} has no definition named {part!r}"
            raise AssertionError(msg)
    return node


def _calls_named(*, node: ast.AST, name: str) -> list[ast.Call]:
    """Return every call to `name`, plain or as an attribute.

    Args:
        node: Subtree to search.
        name: Callee name to match.

    Returns:
        List of matching call nodes, in source order.
    """
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and (getattr(child.func, "id", None) or getattr(child.func, "attr", None))
        == name
    ]


def _keyword(*, call: ast.Call, name: str) -> ast.expr | None:
    """Return the value passed as `name=`, or `None` when it is not passed.

    Args:
        call: Call node to read.
        name: Keyword argument name.

    Returns:
        The keyword's value expression, or `None`.
    """
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _chain(node: ast.expr | None) -> str | None:
    """Render a bare name or attribute chain as dotted text, else `None`.

    Args:
        node: Expression to render.

    Returns:
        The dotted text, or `None` for any other expression shape.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _splatted_chains(*, node: ast.AST) -> set[str]:
    """Return the dotted sources of every `**` splat below `node`.

    A splat of `dict(x)` is reported as `x`, so `**dict(space.actions)` and
    `**space.actions` read the same. Every other call is left wrapped and so
    reports nothing: `dict` copies a mapping whole, and anything else is exactly
    the transformation this certificate must not let past.

    Args:
        node: Subtree to search.

    Returns:
        Set of dotted chains that are splatted.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        values: list[ast.expr | None] = []
        if isinstance(child, ast.Call):
            values = [word.value for word in child.keywords if word.arg is None]
        elif isinstance(child, ast.Dict):
            values = [
                value
                for key, value in zip(child.keys, child.values, strict=True)
                if key is None
            ]
        for value in values:
            unwrapped = (
                value.args[0]
                if isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "dict"
                and len(value.args) == 1
                and not value.keywords
                else value
            )
            chain = _chain(unwrapped)
            if chain is not None:
                found.add(chain)
    return found


def test_build_period_kernels_passes_the_whole_action_name_tuple():
    """The kernel builder hands `get_max_Q_over_a` the space's own action names.

    Anything that filtered, sliced or rebuilt the tuple here would shrink the
    search without shrinking the declared grids, which is exactly the omission
    the zero bound denies.
    """
    tree = _parse("src/_lcm/solution/grid_search.py")
    builder = _definition(tree=tree, qualname="GridSearch.build_period_kernels")
    call = _calls_named(node=builder, name="get_max_Q_over_a")[0]

    assert (
        _chain(_keyword(call=call, name="action_names"))
        == "context.state_action_space.action_names"
    )


@pytest.mark.parametrize("method", ["build_lower_args", "__call__"])
def test_period_kernel_splats_the_whole_action_mapping(method: str):
    """Lowering and the call both splat the state-action space's whole actions.

    The compiled core is lowered against the same mapping it is called with, so
    a subset in either place would silently compile a smaller search than the
    one the other announces.
    """
    tree = _parse("src/_lcm/solution/grid_search.py")
    kernel = _definition(tree=tree, qualname=f"_GridSearchPeriodKernel.{method}")

    assert "state_action_space.actions" in _splatted_chains(node=kernel)


def test_actions_are_product_mapped_over_the_whole_action_tuple():
    """`Q_and_F` is product-mapped over the action-name tuple itself."""
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    builder = _definition(tree=tree, qualname="get_max_Q_over_a")
    call = _calls_named(node=builder, name="productmap")[0]

    assert _chain(_keyword(call=call, name="variables")) == "action_names"


def test_the_action_product_map_is_unbatched():
    """Every action axis carries batch size 0, so no chunk boundary can drop one.

    Batching splits an axis into blocks; a block-shaped reduction that missed a
    remainder would omit candidates the grids declare. Actions are the inner
    optimization axis and are never batched, which removes that failure mode by
    construction rather than by test.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    builder = _definition(tree=tree, qualname="get_max_Q_over_a")
    call = _calls_named(node=builder, name="productmap")[0]
    batch_sizes = _keyword(call=call, name="batch_sizes")
    from_keys = _calls_named(node=batch_sizes, name="fromkeys") if batch_sizes else []

    assert [
        (_chain(node.args[0]), ast.literal_eval(node.args[1])) for node in from_keys
    ] == [("action_names", 0)]


@pytest.mark.parametrize(
    ("builder", "reducer"),
    [
        ("get_max_Q_over_a", "max"),
        ("get_argmax_and_max_Q_over_a", "argmax_and_max"),
    ],
)
def test_the_singleton_reduction_covers_every_action_axis(builder: str, reducer: str):
    """The singleton value is a full reduction: masked, and over no named axis.

    A reduction with no `axis=` covers the whole action product. An `axis=` here
    would leave some action axis unsearched and change the published rank, and
    `where=F_arr` is what makes the mask remove infeasible candidates only —
    never representable ones. Solve and simulate reduce with different callables
    and each owes the obligation separately.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    node = _definition(tree=tree, qualname=builder)
    reductions = [
        call
        for call in _calls_named(node=node, name=reducer)
        if _chain(_keyword(call=call, name="where")) == "F_arr"
    ]

    assert [_keyword(call=call, name="axis") for call in reductions] == [None]


@pytest.mark.parametrize("builder", ["get_max_Q_over_a", "get_argmax_and_max_Q_over_a"])
def test_the_collective_reduction_treats_every_feasibility_axis_as_an_action(
    builder: str,
):
    """Both collective reductions scalarize over `tuple(range(F_arr.ndim))`.

    The feasibility array carries one axis per action and no stakeholder axis,
    so its full rank *is* the action product. Reducing over a prefix of it would
    hand the household an argmax taken over part of its choice set.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    node = _definition(tree=tree, qualname=builder)
    assigned = [
        ast.unparse(child.value)
        for child in ast.walk(node)
        if isinstance(child, ast.Assign)
        and [target.id for target in child.targets if isinstance(target, ast.Name)]
        == ["action_axes"]
    ]

    assert assigned == ["tuple(range(F_arr.ndim))"]


def test_the_taste_shock_reduction_covers_every_action_axis():
    """Continuous axes are maximized out and every remaining axis is smoothed.

    With taste shocks the discrete axes are aggregated by `logsum` rather than
    by `max`, but the candidate set is unchanged: the continuous axes run from
    the first non-discrete axis to the last, and the smoothing runs over all of
    what is left.
    """
    tree = _parse("src/_lcm/regime_building/max_Q_over_a.py")
    builder = _definition(tree=tree, qualname="get_max_Q_over_a")
    assigned = {
        target.id: ast.unparse(child.value)
        for child in ast.walk(builder)
        if isinstance(child, ast.Assign)
        for target in child.targets
        if isinstance(target, ast.Name) and target.id == "continuous_axes"
    }
    smoothing = _calls_named(node=builder, name="logsum_and_softmax")[0]
    axes = _keyword(call=smoothing, name="axes")

    assert (
        assigned["continuous_axes"],
        ast.unparse(axes) if axes is not None else None,
    ) == ("tuple(range(n_discrete_action_axes, Q_arr.ndim))", "tuple(range(Qc.ndim))")


def test_no_obligation_rests_on_an_undeclared_source():
    """The AST-derived obligations equal the generated inventory exactly."""
    parsed = set(derive_source_paths(Path(__file__)))

    assert parsed == set(CERTIFIED_SOURCES)


def test_q_and_f_arrays_reach_full_reducers_without_candidate_transformation():
    """The universal bound is a route-local data-flow proof, not a value sample.

    Literal parses add the shared singleton and collective reducer definitions to
    the same generated source inventory as the route builders. The verifier then
    requires exact source arrays at each reducer and rejects any non-allowlisted
    statement in the certified corridor.
    """
    assert isinstance(_parse("src/_lcm/regime_building/argmax.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/regime_building/collective.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/logsum.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/regime_building/processing.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/utils/dispatchers.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/utils/functools.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/utils/containers.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/zero_safe.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/probability.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/engine.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/state_action_space.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/simulate.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/transitions.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/compile.py"), ast.Module)
    assert isinstance(_parse("src/lcm/model.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/solution/backward_induction.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/initial_conditions.py"), ast.Module)
    assert isinstance(_parse("src/lcm/result.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/result_dataframe.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/result_metadata.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/additional_targets.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/simulation/random.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/regime_building/zero_safe.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/solution/contract.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/grids/__init__.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/grids/base.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/grids/coordinates.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/grids/discrete.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/grids/continuous.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/grids/piecewise.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/processes/__init__.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/processes/base.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/processes/iid.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/processes/ar1.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/variables.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/params/regime_template.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/params/processing.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/dtypes.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/utils/namespace.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/pandas_utils.py"), ast.Module)
    assert isinstance(_parse("src/_lcm/model_processing.py"), ast.Module)

    result = verify_direct_candidate_flow(repo_root=_SRC_ROOT.parent)

    assert result["ok"], "\n".join(result["errors"])
    assert tuple(sorted(result["certified_corridor_sources"])) == CERTIFIED_SOURCES


def test_direct_flow_certificate_names_every_supported_route():
    """The universal proof covers both taste-shock routes as first-class routes."""
    result = verify_direct_candidate_flow(repo_root=_SRC_ROOT.parent)

    assert set(result["routes"]) == {
        "singleton_solve",
        "singleton_simulate",
        "collective_solve",
        "collective_simulate",
        "taste_shock_solve",
        "taste_shock_simulate",
    }


def test_direct_flow_mutations_cover_taste_routes_helpers_and_every_candidate():
    """Synchronized controls attack both taste corridors and their deep helpers."""
    names = set(direct_flow_mutation_specs(repo_root=_SRC_ROOT.parent))
    required = {
        "taste_shock_solve:q_order",
        "taste_shock_solve:q_gap",
        "taste_shock_solve:support_size",
        "taste_shock_solve:shape_axis",
        "taste_shock_solve:inline_q_transform",
        "taste_shock_solve:inline_f_transform",
        "taste_shock_solve:continuous_axis_prefix",
        "taste_shock_solve:logsum_axis_prefix",
        "taste_shock_simulate:q_order",
        "taste_shock_simulate:mt10_rank_permutation",
        "taste_shock_simulate:q_gap",
        "taste_shock_simulate:support_size",
        "taste_shock_simulate:shape_axis",
        "taste_shock_simulate:inline_q_transform",
        "taste_shock_simulate:inline_f_transform",
        "taste_shock_simulate:reshape_drop",
        "taste_shock_simulate:wrong_stride",
        "shared_logsum:q_gap_filter",
        "shared_logsum:support_filter",
        "shared_logsum:axis_prefix",
        "shared_taste_noise:shared_draw",
        "shared_taste_noise:permuted_draw",
        "solve:action_names_rebinding",
        "simulate:action_names_rebinding",
        "solve:dormant_certified_reducer",
        "simulate:return_bypasses_certified_reducer",
        "shared_max:productmap_module_shadow",
        "caller_solve:action_names_slice",
        "caller_solve:wrong_discrete_axis_count",
        "caller_solve:taste_flag_disabled",
        "caller_solve:published_empty_mapping",
        "caller_simulate:action_names_slice",
        "caller_simulate:wrong_discrete_axis_count",
        "caller_simulate:taste_flag_disabled",
        "caller_simulate:live_taste_flag_rebinding",
        "caller_simulate:published_empty_mapping",
        "shared_logsum:wrong_euler_gamma",
        "shared_taste_noise:cast_import_replaced",
        "solve:builder_decorator_wrapper",
        "simulate:builder_decorator_wrapper",
        "solve:builder_default_changed",
        "simulate:builder_default_changed",
        "taste_shock_solve:attribute_with_signature",
        "taste_shock_simulate:attribute_with_signature",
        "taste_shock_solve:attribute_q_and_f",
        "taste_shock_simulate:attribute_q_and_f",
        "singleton_simulate:attribute_argmax_and_max",
        "collective_solve:attribute_collective_readout",
        "collective_simulate:attribute_collective_argmax",
        "solve:attribute_productmap",
        "simulate:attribute_productmap",
        "caller_simulate:attribute_simulation_phase",
        "shared_argmax:range_module_shadow",
        "shared_collective:argmax_module_shadow",
        "shared_productmap:drop_last_axis",
        "shared_functools:drop_last_argument",
        "shared_containers:duplicate_threshold",
        "shared_zero_safe:ordered_sum_slice",
        "shared_probability:unbalanced_product",
        "shared_engine:action_order_reversed",
        "shared_engine:actions_drop_last_candidate",
        "shared_engine:replace_drops_inherited_candidates",
        "shared_state_action_space:continuous_order_reversed",
        "shared_state_action_space:drop_last_continuous_candidate",
        "candidate_materialization:grid_base_intercepts_to_jax",
        "simulation_state_action_space:drops_inherited_candidates",
        "simulation_state_action_space:caller_drops_inherited_candidates",
        "simulation_index_consumer:next_candidate",
        "aot_compile:argmax_index_shift",
        "aot_model:compiled_regime_filter",
        "shared_dedup_key:collapse_plain_callables",
        "simulation_publication:shift_padded_actions",
        "simulation_result:shift_raw_actions",
        "simulation_dataframe:shift_action_column",
        "simulation_metadata:drop_regime_actions",
        "additional_targets:overwrite_actions_single_pass",
        "additional_targets:overwrite_actions_chunked",
        "simulation_random:reassign_taste_keys",
        "shared_fold_average:negated_value",
        "solution_contract:negate_kernel_result",
        "candidate_materialization:rebind_continuous_grid",
        "candidate_materialization:rebind_process_class",
        "candidate_materialization:drop_last_discrete_code",
        "candidate_materialization:drop_last_linear_point",
        "candidate_materialization:drop_last_coordinate_point",
        "candidate_materialization:drop_last_piecewise_point",
        "candidate_materialization:drop_last_process_node",
        "candidate_materialization:drop_last_iid_node",
        "candidate_materialization:drop_last_ar1_node",
        "candidate_materialization:drop_last_action_name",
        "candidate_materialization:skip_first_runtime_action_template",
        "candidate_materialization:negate_broadcast_runtime_points",
        "candidate_materialization:negate_flattened_runtime_points",
        "candidate_materialization:negate_cast_runtime_points",
        "candidate_materialization:negate_series_runtime_points",
        "candidate_materialization:negate_fixed_runtime_points",
    }
    required.update(
        f"{route}:candidate_index_{index}"
        for route in ("taste_shock_solve", "taste_shock_simulate")
        for index in range(len(_CANDIDATES))
    )

    assert required <= names
    # Independent literals make both cardinality and family identity part of this
    # certificate, rather than trusting constants supplied by the mutation generator.
    assert len(names) == 176
    assert (
        hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()
        == "fea9bb27d93337e16d2505c643660b3816cb37d5e87d3cb41a0a2e4baad4739f"
    )


def test_certified_sources_are_read_as_utf_8_whatever_the_platform_default_is():
    """The structural reads name their encoding rather than taking the locale's.

    House style puts literal `—`, `→` and `μ` in source files, and a platform
    whose default codec is not UTF-8 cannot decode them: the read raises before
    anything can be parsed.
    """
    recorded: list[str | None] = []
    original = Path.read_text

    def recording_read_text(self: Path, *args: object, **kwargs: object) -> str:
        recorded.append(kwargs.get("encoding"))  # ty: ignore[invalid-argument-type]
        return original(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "read_text", recording_read_text)
        for relative in CERTIFIED_SOURCES:
            _parse(relative)

    assert set(recorded) == {"utf-8"}


def test_certified_source_hashes_ignore_checkout_newline_convention(tmp_path: Path):
    """LF and CRLF worktrees represent the same sealed Python source."""
    source = tmp_path / "source.py"
    source.write_bytes(b"def f():\n    return 1\n")
    lf_digest = sha256_file(source)

    source.write_bytes(b"def f():\r\n    return 1\r\n")
    assert sha256_file(source) == lf_digest

    source.write_bytes(b"def f():\r\n    return 2\r\n")
    assert sha256_file(source) != lf_digest


def test_direct_flow_source_seals_ignore_checkout_newline_convention(
    monkeypatch: pytest.MonkeyPatch,
):
    """A Windows CRLF checkout satisfies the same direct-flow source seals."""
    original = Path.read_bytes
    certified = set(CERTIFIED_SOURCES)

    def crlf_read_bytes(self: Path) -> bytes:
        payload = original(self)
        try:
            relative = self.relative_to(_SRC_ROOT.parent).as_posix()
        except ValueError:
            return payload
        if relative not in certified:
            return payload
        normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return normalized.replace(b"\n", b"\r\n")

    monkeypatch.setattr(Path, "read_bytes", crlf_read_bytes)

    result = verify_direct_candidate_flow(repo_root=_SRC_ROOT.parent)

    assert result["ok"], "\n".join(result["errors"])


def _only_target(
    work: DiscreteAction,
    consumption: ContinuousAction,
    target_work: float,
    target_consumption: float,
) -> BoolND:
    """Admit exactly the one candidate the params name."""
    return jnp.isclose(work, target_work) & jnp.isclose(consumption, target_consumption)


def _candidate_mask(
    work: DiscreteAction,
    consumption: ContinuousAction,
    feasible_0: float,
    feasible_1: float,
    feasible_2: float,
    feasible_3: float,
    feasible_4: float,
    feasible_5: float,
) -> BoolND:
    """Admit the candidates named by a six-bit parameterized mask."""
    flat_index = jnp.asarray(work, dtype=jnp.int32) * len(
        _CONSUMPTION_VALUES
    ) + jnp.asarray(consumption - _CONSUMPTION_VALUES[0], dtype=jnp.int32)
    flags = jnp.asarray(
        [feasible_0, feasible_1, feasible_2, feasible_3, feasible_4, feasible_5]
    )
    return flags[flat_index] > 0.5


def _ranked_utility(
    wealth: ContinuousState,
    work: DiscreteAction,
    consumption: ContinuousAction,
    rank_0: float,
    rank_1: float,
    rank_2: float,
    rank_3: float,
    rank_4: float,
    rank_5: float,
) -> FloatND:
    """Publish the strict flattened ordering named by the six rank parameters.

    Taking the ordering from parameters lets every candidate be placed in the
    winning position. Under any single fixed ordering only one candidate is ever
    the maximizer of a fully feasible neighborhood.
    """
    flat_index = jnp.asarray(work, dtype=jnp.int32) * len(
        _CONSUMPTION_VALUES
    ) + jnp.asarray(consumption - _CONSUMPTION_VALUES[0], dtype=jnp.int32)
    ranks = jnp.asarray([rank_0, rank_1, rank_2, rank_3, rank_4, rank_5])
    return wealth + ranks[flat_index]


def _labelled_utility(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """Give every candidate its own value, so the published one identifies it."""
    return wealth + consumption + 10.0 * work


def _peaked_utility(
    wealth: ContinuousState,
    work: DiscreteAction,
    consumption: ContinuousAction,
    target_work: float,
    target_consumption: float,
) -> FloatND:
    """Peak at the state's own value on the named candidate, below it elsewhere."""
    return wealth - (work - target_work) ** 2 - (consumption - target_consumption) ** 2


def _utility_f(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """First stakeholder's own value of a candidate."""
    return wealth + consumption + 10.0 * work


def _utility_m(
    wealth: ContinuousState, work: DiscreteAction, consumption: ContinuousAction
) -> FloatND:
    """Second stakeholder's own value of the same candidate."""
    return wealth + 2.0 * consumption + 5.0 * work


def _zero_f() -> FloatND:
    """First stakeholder's terminal payoff."""
    return jnp.array(0.0)


def _zero_m() -> FloatND:
    """Second stakeholder's terminal payoff."""
    return jnp.array(0.0)


def _next_regime() -> ScalarInt:
    """Leave the acting regime after its single period."""
    return RegimeId.done


def _actions() -> dict[str, DiscreteGrid | LinSpacedGrid]:
    """Return the two action grids every model in this module declares."""
    return {
        "work": DiscreteGrid(Work),
        "consumption": LinSpacedGrid(
            start=_CONSUMPTION_VALUES[0],
            stop=_CONSUMPTION_VALUES[-1],
            n_points=len(_CONSUMPTION_VALUES),
        ),
    }


def _build_model(
    *,
    utility: Callable[..., FloatND] | CollectiveUtility,
    constraints: Mapping[str, Callable[..., BoolND]],
    terminal_utility: Callable[..., FloatND] | CollectiveUtility,
    taste_shocks: bool = False,
    n_subjects: int | None = None,
) -> Model:
    """Build the one-decision model the sweeps solve.

    Args:
        utility: The acting regime's utility declaration.
        constraints: The acting regime's constraints.
        terminal_utility: The terminal regime's utility declaration.
        taste_shocks: Whether the acting singleton declares EV1 taste shocks.
        n_subjects: Subject count to precompile, or ``None`` for the lazy path.

    Returns:
        The built model.
    """
    acting = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "wealth": LinSpacedGrid(
                start=float(_WEALTH_VALUES[0]),
                stop=float(_WEALTH_VALUES[-1]),
                n_points=len(_WEALTH_VALUES),
            )
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        actions=_actions(),
        functions={"utility": utility},
        constraints=dict(constraints),
        taste_shocks=ExtremeValueTasteShocks() if taste_shocks else None,
    )
    done = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": terminal_utility},
    )
    return Model(
        regimes={"acting": acting, "done": done},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
        n_subjects=n_subjects,
    )


def _dedup_utility_left(
    wealth: ContinuousState,
    work: DiscreteAction,
    consumption: ContinuousAction,
) -> FloatND:
    """Select a wealth-specific candidate on the left regime's shared grid."""
    target_work = wealth - 1
    target_consumption = 2 * wealth - 1
    return (
        wealth
        - jnp.square(work - target_work)
        - jnp.square(consumption - target_consumption)
    )


def _dedup_utility_right(
    wealth: ContinuousState,
    work: DiscreteAction,
    consumption: ContinuousAction,
) -> FloatND:
    """Select the last candidate on the right regime's shared-shaped grid."""
    return wealth - jnp.square(work - 1) - jnp.square(consumption - 3)


def _next_dedup_done() -> ScalarInt:
    """Send either collision-witness regime to the common terminal regime."""
    return DedupRegimeId.done


def _dedup_terminal_utility(wealth: ContinuousState) -> FloatND:
    """Keep the collision witness's terminal continuation action-neutral."""
    return wealth


def _build_dedup_collision_model(*, n_subjects: int | None = 2) -> Model:
    """Build same-shaped regimes whose AOT argmax partials must remain distinct."""
    wealth_grid = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)

    def decision_regime(utility: Callable[..., FloatND]) -> Regime:
        return Regime(
            transition=_next_dedup_done,
            active=lambda age: age < 1,
            states={"wealth": wealth_grid},
            state_transitions={"wealth": fixed_transition("wealth")},
            actions=_actions(),
            functions={"utility": utility},
        )

    return Model(
        regimes={
            "left": decision_regime(_dedup_utility_left),
            "right": decision_regime(_dedup_utility_right),
            "done": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={"wealth": wealth_grid},
                functions={"utility": _dedup_terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=DedupRegimeId,
        n_subjects=n_subjects,
    )


def _dedup_params(model: Model) -> dict[str, Any]:
    """Populate continuation parameters for both same-shaped decision regimes."""
    params = cast("dict[str, Any]", model.get_params_template())
    for name in ("left", "right"):
        params[name]["koopmans_aggregator"]["discount_factor"] = 0.5
    return params


def _simulate_dedup_model(model: Model, *, params: dict[str, Any]):
    """Run both same-shaped decision regimes through the public AOT path."""
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(2),
            "wealth": jnp.ones(2),
            "regime_id": jnp.asarray(
                [DedupRegimeId.left, DedupRegimeId.right], dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False)
    return frame[frame["period"] == 0].sort_values("subject_id")


def _solve_acting(
    *, model: Model, work: float, consumption: float, function_name: str
) -> FloatND:
    """Solve for one named target candidate and return the acting regime's value.

    Args:
        model: The model to solve.
        work: Target value of the discrete action.
        consumption: Target value of the continuous action.
        function_name: Params key of the function reading the two targets.

    Returns:
        The acting regime's period-0 value array.
    """
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"][function_name]["target_work"] = work
    params["acting"][function_name]["target_consumption"] = consumption
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    return model.solve(params=params, log_level="debug")[0]["acting"]


@pytest.fixture(scope="module")
def unique_feasible_model() -> Model:
    """A singleton model whose params name the one admissible candidate."""
    return _build_model(
        utility=_labelled_utility,
        constraints={"only_target": _only_target},
        terminal_utility=lambda: jnp.array(0.0),
    )


@pytest.fixture(scope="module")
def unique_maximizer_model() -> Model:
    """A singleton model with no constraints whose params name the argmax."""
    return _build_model(
        utility=_peaked_utility,
        constraints={},
        terminal_utility=lambda: jnp.array(0.0),
    )


@pytest.fixture(scope="module")
def collective_model() -> Model:
    """A two-stakeholder model whose params name the one admissible candidate."""
    return _build_model(
        utility=CollectiveUtility(utilities={"f": _utility_f, "m": _utility_m}),
        constraints={"only_target": _only_target},
        terminal_utility=CollectiveUtility(utilities={"f": _zero_f, "m": _zero_m}),
    )


@pytest.fixture(scope="module")
def masked_singleton_model() -> Model:
    """Singleton model whose feasibility neighborhood is supplied by parameters."""
    return _build_model(
        utility=_ranked_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
    )


@pytest.fixture(scope="module")
def masked_taste_shock_model() -> Model:
    """Taste-shock singleton with parameterized ranks and feasibility."""
    return _build_model(
        utility=_ranked_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
        taste_shocks=True,
    )


@pytest.fixture(scope="module")
def masked_taste_shock_aot_model() -> Model:
    """Taste-shock singleton routed through public subject-count AOT compilation."""
    return _build_model(
        utility=_ranked_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
        taste_shocks=True,
        n_subjects=len(_WEALTH_VALUES),
    )


@pytest.fixture(scope="module")
def masked_collective_model() -> Model:
    """Collective model with the same strict ordering for both stakeholders."""
    utility = CollectiveUtility(utilities={"f": _ranked_utility, "m": _ranked_utility})
    return _build_model(
        utility=utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=CollectiveUtility(utilities={"f": _zero_f, "m": _zero_m}),
    )


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_every_declared_candidate_can_be_the_only_feasible_one(
    unique_feasible_model: Model, work: float, consumption: float
):
    """Constraining the search to one candidate publishes that candidate's value.

    A candidate the search never evaluates leaves the feasibility mask empty,
    and the published value is `-inf` rather than the candidate's own.
    """
    V_arr = _solve_acting(
        model=unique_feasible_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    aaae(V_arr, _WEALTH_VALUES + consumption + 10.0 * work, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_every_declared_candidate_can_win_the_maximization(
    unique_maximizer_model: Model, work: float, consumption: float
):
    """With everything feasible, the search finds whichever candidate peaks.

    The utility peaks at exactly zero on the named candidate and is strictly
    negative on every other, so a published zero says that candidate entered the
    argmax and a published negative says the search never reached it.
    """
    V_arr = _solve_acting(
        model=unique_maximizer_model,
        work=work,
        consumption=consumption,
        function_name="utility",
    )

    aaae(V_arr, _WEALTH_VALUES, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_every_declared_candidate_can_be_the_household_choice(
    collective_model: Model, work: float, consumption: float
):
    """The collective reduction reads each stakeholder's own value at the candidate.

    The collective path scalarizes and argmaxes on its own code, so exhausting
    the action product there is a separate obligation from the singleton case.
    """
    V_arr = _solve_acting(
        model=collective_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    aaae(
        V_arr,
        _WEALTH_VALUES[:, None]
        + jnp.array([consumption + 10.0 * work, 2.0 * consumption + 5.0 * work]),
        decimal=DECIMAL_PRECISION,
    )


def _simulate_acting(
    *, model: Model, work: float, consumption: float, function_name: str
) -> dict[str, list[float]]:
    """Simulate one target candidate and return the actions it published at period 0.

    Args:
        model: The model to solve and simulate.
        work: Target value of the discrete action.
        consumption: Target value of the continuous action.
        function_name: Params key of the function reading the two targets.

    Returns:
        Mapping of each action name to its published value, one entry per subject.
    """
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"][function_name]["target_work"] = work
    params["acting"][function_name]["target_consumption"] = consumption
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(len(_WEALTH_VALUES)),
            "wealth": _WEALTH_VALUES,
            "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False)
    period_0 = frame[frame["period"] == 0]
    return {
        name: period_0[name].to_numpy().tolist() for name in ("work", "consumption")
    }


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_simulation_routes_to_every_declared_candidate(
    unique_feasible_model: Model, work: float, consumption: float
):
    """Simulation publishes the one candidate the constraint admits.

    Simulation maximizes with `get_argmax_and_max_Q_over_a`, a different callable from
    the one backward induction reduces with, so a candidate the solve reaches is not
    thereby a candidate simulation reaches. The bound covers every workload, and
    simulation is one.
    """
    rows = _simulate_acting(
        model=unique_feasible_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    assert rows == {
        "work": [work] * len(_WEALTH_VALUES),
        "consumption": [consumption] * len(_WEALTH_VALUES),
    }


@pytest.mark.parametrize(("work", "consumption"), _CANDIDATES)
def test_simulation_routes_a_household_to_every_declared_candidate(
    collective_model: Model, work: float, consumption: float
):
    """The collective simulate reduction reaches every candidate too.

    `collective_argmax_and_readout` is a third reduction, distinct from both the
    singleton simulate path and the collective solve path.
    """
    rows = _simulate_acting(
        model=collective_model,
        work=work,
        consumption=consumption,
        function_name="only_target",
    )

    assert rows == {
        "work": [work] * len(_WEALTH_VALUES),
        "consumption": [consumption] * len(_WEALTH_VALUES),
    }


def _params_for_mask(
    *, model: Model, mask: tuple[bool, ...], ranks: tuple[float, ...]
) -> dict[str, Any]:
    """Populate the parameter template with one mask and one candidate ordering."""
    params = cast("dict[str, Any]", model.get_params_template())
    for name, feasible in zip(_MASK_PARAMETER_NAMES, mask, strict=True):
        params["acting"]["candidate_mask"][name] = float(feasible)
    for stakeholder in _utility_keys(model):
        for name, rank in zip(_RANK_PARAMETER_NAMES, ranks, strict=True):
            params["acting"][stakeholder][name] = float(rank)
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    if "taste_shocks" in params["acting"]:
        params["acting"]["taste_shocks"]["scale"] = _TASTE_SCALE
    return params


def _utility_keys(model: Model) -> tuple[str, ...]:
    """Name the params keys carrying the ranked utility, one per stakeholder."""
    params = cast("dict[str, Any]", model.get_params_template())
    return tuple(
        key
        for key, entry in params["acting"].items()
        if isinstance(entry, dict) and _RANK_PARAMETER_NAMES[0] in entry
    )


def _solve_mask_case(
    *, model: Model, mask: tuple[bool, ...], ranks: tuple[float, ...]
) -> FloatND:
    """Solve one mask neighborhood and return the acting regime's value."""
    params = _params_for_mask(model=model, mask=mask, ranks=ranks)
    return model.solve(params=params, log_level="debug")[0]["acting"]


def _simulate_mask_case(
    *, model: Model, mask: tuple[bool, ...], ranks: tuple[float, ...]
) -> dict[str, list[float]]:
    """Simulate one mask neighborhood and return actions plus published values."""
    params = _params_for_mask(model=model, mask=mask, ranks=ranks)
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(len(_WEALTH_VALUES)),
            "wealth": _WEALTH_VALUES,
            "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False)
    period_0 = frame[frame["period"] == 0]
    return {name: period_0[name].to_numpy().tolist() for name in period_0.columns}


def _period_zero_action_views(
    *, result: Any, regime_name: str
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Read selected actions from raw, ordinary, and enriched public results."""
    raw_period = result.raw_results[regime_name][0]
    action_names = tuple(raw_period.actions)
    raw = {name: raw_period.actions[name].tolist() for name in action_names}

    frame = result.to_dataframe(use_labels=False)
    frame = frame[frame["period"] == 0].sort_values("subject_id")
    ordinary = {name: frame[name].to_numpy().tolist() for name in action_names}

    enriched_frame = result.to_dataframe(
        additional_targets=["utility"], use_labels=False
    )
    enriched_frame = enriched_frame[enriched_frame["period"] == 0].sort_values(
        "subject_id"
    )
    enriched = {name: enriched_frame[name].to_numpy().tolist() for name in action_names}
    return raw, ordinary, enriched


def _simulate_materialization_case(
    *, model: Model
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Select the last two-grid candidate and expose all publication views."""
    result = model.simulate(
        params=_params_for_mask(
            model=model,
            mask=tuple(True for _ in _CANDIDATES),
            ranks=tuple(float(index) for index in range(len(_CANDIDATES))),
        ),
        initial_conditions={
            "age": jnp.zeros(len(_WEALTH_VALUES)),
            "wealth": _WEALTH_VALUES,
            "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    return _period_zero_action_views(result=result, regime_name="acting")


def _ranked_materialization_model(*, n_subjects: int | None) -> Model:
    """Build the ordinary two-grid model for materialization mutations."""
    return _build_model(
        utility=_ranked_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
        n_subjects=n_subjects,
    )


@pytest.mark.parametrize(
    "compiled_n_subjects", [None, len(_WEALTH_VALUES)], ids=["lazy", "aot"]
)
def test_dropping_a_discrete_grid_code_changes_every_public_candidate_view(
    compiled_n_subjects: int | None,
    monkeypatch: pytest.MonkeyPatch,
):
    """The discrete-grid materializer is behaviorally live before Q_and_F."""
    baseline = _simulate_materialization_case(
        model=_ranked_materialization_model(n_subjects=compiled_n_subjects)
    )
    original = DiscreteGrid.to_jax

    def drop_last_code(grid: DiscreteGrid) -> FloatND:
        return original(grid)[:-1]

    monkeypatch.setattr(DiscreteGrid, "to_jax", drop_last_code)
    shifted = _simulate_materialization_case(
        model=_ranked_materialization_model(n_subjects=compiled_n_subjects)
    )

    expected = {
        "work": [1, 1],
        "consumption": [3.0, 3.0],
    }
    shifted_expected = {
        "work": [0, 0],
        "consumption": [3.0, 3.0],
    }
    assert baseline == (expected, expected, expected)
    assert shifted == (shifted_expected, shifted_expected, shifted_expected)


@pytest.mark.parametrize(
    "compiled_n_subjects", [None, len(_WEALTH_VALUES)], ids=["lazy", "aot"]
)
def test_dropping_a_linear_grid_point_changes_every_public_candidate_view(
    compiled_n_subjects: int | None,
    monkeypatch: pytest.MonkeyPatch,
):
    """The continuous-grid materializer is behaviorally live before Q_and_F."""
    baseline = _simulate_materialization_case(
        model=_ranked_materialization_model(n_subjects=compiled_n_subjects)
    )
    original = LinSpacedGrid.to_jax

    def drop_last_action_point(grid: LinSpacedGrid) -> FloatND:
        points = original(grid)
        is_action_grid = (
            int(grid.n_points) == len(_CONSUMPTION_VALUES)
            and math.isclose(float(grid.start), _CONSUMPTION_VALUES[0])
            and math.isclose(float(grid.stop), _CONSUMPTION_VALUES[-1])
        )
        return points[:-1] if is_action_grid else points

    monkeypatch.setattr(LinSpacedGrid, "to_jax", drop_last_action_point)
    shifted = _simulate_materialization_case(
        model=_ranked_materialization_model(n_subjects=compiled_n_subjects)
    )

    expected = {
        "work": [1, 1],
        "consumption": [3.0, 3.0],
    }
    shifted_expected = {
        "work": [1, 1],
        "consumption": [2.0, 2.0],
    }
    assert baseline == (expected, expected, expected)
    assert shifted == (shifted_expected, shifted_expected, shifted_expected)


def _runtime_action_utility(
    wealth: ContinuousState, choice: ContinuousAction, target_choice: float
) -> FloatND:
    """Make any selected supplied irregular action point the unique winner."""
    return wealth - (jnp.asarray(choice, dtype=float) - target_choice) ** 2


def _build_runtime_action_model(
    *,
    n_subjects: int | None,
    fixed_points: FloatND | None = None,
) -> Model:
    """Build a model whose candidate menu arrives through public params."""
    fixed_params: dict[str, Any] = {}
    if fixed_points is not None:
        fixed_params = {"acting": {"choice": {"points": fixed_points}}}
    return Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                states={
                    "wealth": LinSpacedGrid(
                        start=float(_WEALTH_VALUES[0]),
                        stop=float(_WEALTH_VALUES[-1]),
                        n_points=len(_WEALTH_VALUES),
                    )
                },
                state_transitions={"wealth": fixed_transition("wealth")},
                actions={"choice": IrregSpacedGrid(n_points=3)},
                functions={"utility": _runtime_action_utility},
            ),
            "done": Regime(
                transition=None,
                active=lambda age: age >= 1,
                states={
                    "wealth": LinSpacedGrid(
                        start=float(_WEALTH_VALUES[0]),
                        stop=float(_WEALTH_VALUES[-1]),
                        n_points=len(_WEALTH_VALUES),
                    )
                },
                functions={"utility": lambda wealth: wealth},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=RegimeId,
        fixed_params=fixed_params,
        n_subjects=n_subjects,
    )


def _simulate_runtime_action_model(
    *, point_source: str, n_subjects: int | None, target_choice: float
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Run array, Series, or constructor-fixed runtime action points."""
    points = jnp.asarray([1.0, 2.0, 3.0])
    model = _build_runtime_action_model(
        n_subjects=n_subjects,
        fixed_points=points if point_source == "fixed" else None,
    )
    params = cast("dict[str, Any]", model.get_params_template())
    params["acting"]["koopmans_aggregator"]["discount_factor"] = 0.5
    params["acting"]["utility"]["target_choice"] = target_choice
    if point_source != "fixed":
        params["acting"]["choice"]["points"] = (
            pd.Series([1.0, 2.0, 3.0]) if point_source == "series" else points
        )

    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(len(_WEALTH_VALUES)),
            "wealth": _WEALTH_VALUES,
            "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )
    return _period_zero_action_views(result=result, regime_name="acting")


@pytest.mark.parametrize("point_source", ["array", "series", "fixed"])
@pytest.mark.parametrize("target_choice", [1.0, 2.0, 3.0])
@pytest.mark.parametrize(
    "compiled_n_subjects", [None, len(_WEALTH_VALUES)], ids=["lazy", "aot"]
)
def test_public_runtime_action_points_reach_every_candidate_view(
    point_source: str,
    target_choice: float,
    compiled_n_subjects: int | None,
):
    """Array, Series, and fixed public points all complete the candidate menu."""
    observed = _simulate_runtime_action_model(
        point_source=point_source,
        n_subjects=compiled_n_subjects,
        target_choice=target_choice,
    )
    expected = {"choice": [target_choice, target_choice]}
    assert observed == (expected, expected, expected)


def _reference_for_mask(
    mask: tuple[bool, ...], ranks: tuple[float, ...]
) -> tuple[int, float, float, float]:
    """Return flat index, value, work, and consumption from the scalar oracle."""
    index, value = reference_masked_argmax(ranks, mask)
    work, consumption = _CANDIDATES[index]
    return index, value, work, consumption


def test_singleton_solve_matches_reference_over_every_nonempty_feasibility_mask(
    masked_singleton_model: Model,
):
    """Singleton solve agrees with the scalar oracle on every mask and ordering.

    Sweeping the ordering as well as the mask puts each candidate in the winning
    position, so omitting any one of them changes a published winner.
    """
    for ranks in _RANK_VECTORS:
        for mask in _NONEMPTY_FEASIBILITY_MASKS:
            _index, value, _work, _consumption = _reference_for_mask(mask, ranks)
            observed = _solve_mask_case(
                model=masked_singleton_model, mask=mask, ranks=ranks
            )
            aaae(observed, _WEALTH_VALUES + value, decimal=DECIMAL_PRECISION)


def test_singleton_simulate_matches_reference_over_every_nonempty_feasibility_mask(
    masked_singleton_model: Model,
):
    """Singleton simulation agrees with the scalar oracle on every mask and ordering."""
    for ranks in _RANK_VECTORS:
        for mask in _NONEMPTY_FEASIBILITY_MASKS:
            _index, value, work, consumption = _reference_for_mask(mask, ranks)
            observed = _simulate_mask_case(
                model=masked_singleton_model, mask=mask, ranks=ranks
            )
            assert observed["work"] == [work] * len(_WEALTH_VALUES), (mask, ranks)
            assert observed["consumption"] == [consumption] * len(_WEALTH_VALUES), (
                mask,
                ranks,
            )
            aaae(observed["value"], _WEALTH_VALUES + value, decimal=DECIMAL_PRECISION)


_TASTE_SCALE = 0.2
_STANDARDIZED_TASTE_NOISE = (-1.5, 1.5)


def _taste_masks(target: int) -> tuple[tuple[bool, ...], ...]:
    """Give one singleton, two intermediate, and the full support around a winner."""
    discrete, continuous = divmod(target, len(_CONSUMPTION_VALUES))
    other_discrete = 1 - discrete
    same_cell_rival = discrete * len(_CONSUMPTION_VALUES) + (continuous + 1) % 3
    other_cell_same = other_discrete * len(_CONSUMPTION_VALUES) + continuous
    other_cell_rival = other_discrete * len(_CONSUMPTION_VALUES) + (continuous + 2) % 3

    def mask(*indices: int) -> tuple[bool, ...]:
        return tuple(index in indices for index in range(len(_CANDIDATES)))

    return (
        mask(target),
        mask(target, same_cell_rival, other_cell_same),
        mask(target, same_cell_rival, other_cell_same, other_cell_rival),
        tuple(True for _ in _CANDIDATES),
    )


def _reference_taste_outcome(
    *, mask: tuple[bool, ...], ranks: tuple[float, ...]
) -> tuple[int, float, float]:
    """Return noisy flat winner, solved logsum, and selected unshocked value."""
    n_continuous = len(_CONSUMPTION_VALUES)
    continuous_winners: list[int] = []
    q_continuous: list[float] = []
    for discrete in range(len(_WORK_VALUES)):
        cell = range(discrete * n_continuous, (discrete + 1) * n_continuous)
        feasible = [index for index in cell if mask[index]]
        if feasible:
            winner = max(feasible, key=ranks.__getitem__)
            continuous_winners.append(winner)
            q_continuous.append(ranks[winner])
        else:
            continuous_winners.append(discrete * n_continuous)
            q_continuous.append(-math.inf)

    finite = [value for value in q_continuous if math.isfinite(value)]
    shift = max(finite)
    solved = shift + _TASTE_SCALE * math.log(
        sum(math.exp((value - shift) / _TASTE_SCALE) for value in finite)
    )
    noisy = [
        value + _TASTE_SCALE * noise
        for value, noise in zip(q_continuous, _STANDARDIZED_TASTE_NOISE, strict=True)
    ]
    discrete_winner = max(range(len(noisy)), key=noisy.__getitem__)
    flat_winner = continuous_winners[discrete_winner]
    return flat_winner, solved, ranks[flat_winner]


def _controlled_taste_noise(
    *, key: Any, shape: tuple[int, ...], scale: FloatND
) -> FloatND:
    """Supply deterministic, candidate-distinct noise at the production seam."""
    del key
    assert shape == (len(_WORK_VALUES),)
    return scale * jnp.asarray(_STANDARDIZED_TASTE_NOISE)


def _simulate_taste_mask_case(
    *, model: Model, mask: tuple[bool, ...], ranks: tuple[float, ...]
) -> dict[str, list[float]]:
    """Simulate one taste-shock neighborhood under deterministic per-cell noise."""
    params = _params_for_mask(model=model, mask=mask, ranks=ranks)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "draw_taste_shock_noise",
            _controlled_taste_noise,
        )
        result = model.simulate(
            params=params,
            initial_conditions={
                "age": jnp.zeros(len(_WEALTH_VALUES)),
                "wealth": _WEALTH_VALUES,
                "regime_id": jnp.full(len(_WEALTH_VALUES), RegimeId.acting),
            },
            period_to_regime_to_V_arr=None,
            log_level="debug",
            seed=409,
        )
    frame = result.to_dataframe(use_labels=False)
    period_0 = frame[frame["period"] == 0]
    return {name: period_0[name].to_numpy().tolist() for name in period_0.columns}


def test_aot_dedup_keeps_same_shaped_regime_reducers_distinct():
    """Different bound Q/F objects cannot share one compiled argmax program."""
    model = _build_dedup_collision_model()
    period_0 = _simulate_dedup_model(model, params=_dedup_params(model))

    assert period_0["work"].to_numpy().tolist() == [0, 1]
    assert period_0["consumption"].to_numpy().tolist() == [1.0, 3.0]


def test_collapsing_plain_callable_dedup_keys_changes_the_published_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    """The synchronized dedup-key defect is visible on the public AOT path."""
    model = _build_dedup_collision_model()
    params = _dedup_params(model)
    values = model.solve(params=params, log_level="debug")
    original = simulation_compile_module._func_dedup_key

    def collapsed_key(*, func: Callable[..., Any]):
        if isinstance(func, functools.partial):
            return original(func=func)
        return 0

    monkeypatch.setattr(simulation_compile_module, "_func_dedup_key", collapsed_key)
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(2),
            "wealth": jnp.ones(2),
            "regime_id": jnp.asarray(
                [DedupRegimeId.left, DedupRegimeId.right], dtype=jnp.int32
            ),
        },
        period_to_regime_to_V_arr=values,
        log_level="debug",
    )
    frame = result.to_dataframe(use_labels=False)
    period_0 = frame[frame["period"] == 0].sort_values("subject_id")

    observed = list(
        zip(
            period_0["work"].to_numpy().tolist(),
            period_0["consumption"].to_numpy().tolist(),
            strict=True,
        )
    )
    assert observed != [(0, 1.0), (1, 3.0)]


@pytest.mark.parametrize("compiled_n_subjects", [None, 3], ids=["lazy", "aot"])
def test_padded_heterogeneous_candidates_remain_subject_aligned(
    compiled_n_subjects: int | None,
):
    """Padding and trimming preserve each subject's selected candidate."""
    model = _build_dedup_collision_model(n_subjects=compiled_n_subjects)
    result = model.simulate(
        params=_dedup_params(model),
        initial_conditions={
            "age": jnp.zeros(3),
            "wealth": jnp.asarray([1.0, 2.0, 1.0]),
            "regime_id": jnp.full(3, DedupRegimeId.left, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        subject_batch_size=2,
    )

    raw = result.raw_results["left"][0]
    assert raw.actions["work"].tolist() == [0, 1, 0]
    assert raw.actions["consumption"].tolist() == [1.0, 3.0, 1.0]

    period_0 = result.to_dataframe(use_labels=False)
    period_0 = period_0[period_0["period"] == 0].sort_values("subject_id")
    assert period_0["work"].to_numpy().tolist() == [0, 1, 0]
    assert period_0["consumption"].to_numpy().tolist() == [1.0, 3.0, 1.0]

    enriched = result.to_dataframe(additional_targets=["utility"], use_labels=False)
    enriched = enriched[enriched["period"] == 0].sort_values("subject_id")
    assert enriched["work"].to_numpy().tolist() == [0, 1, 0]
    assert enriched["consumption"].to_numpy().tolist() == [1.0, 3.0, 1.0]


@pytest.mark.parametrize("compiled_n_subjects", [None, 3], ids=["lazy", "aot"])
def test_additional_targets_preserve_unbatched_selected_candidates(
    compiled_n_subjects: int | None,
):
    """Computed targets cannot overwrite an already published action column."""
    model = _build_dedup_collision_model(n_subjects=compiled_n_subjects)
    result = model.simulate(
        params=_dedup_params(model),
        initial_conditions={
            "age": jnp.zeros(3),
            "wealth": jnp.asarray([1.0, 2.0, 1.0]),
            "regime_id": jnp.full(3, DedupRegimeId.left, dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )

    raw = result.raw_results["left"][0]
    assert raw.actions["work"].tolist() == [0, 1, 0]
    assert raw.actions["consumption"].tolist() == [1.0, 3.0, 1.0]

    frame = result.to_dataframe(additional_targets=["utility"], use_labels=False)
    period_0 = frame[frame["period"] == 0].sort_values("subject_id")
    assert period_0["work"].to_numpy().tolist() == [0, 1, 0]
    assert period_0["consumption"].to_numpy().tolist() == [1.0, 3.0, 1.0]


def _route_to_folded(work: DiscreteAction) -> FloatND:
    return jnp.asarray(work, dtype=float)


def _route_to_dead(work: DiscreteAction) -> FloatND:
    return 1.0 - jnp.asarray(work, dtype=float)


def _fold_source_utility(work: DiscreteAction, wealth: ContinuousState) -> FloatND:
    return 0.5 * (1.0 - jnp.asarray(work, dtype=float)) + 0.0 * wealth


def _folded_terminal_utility(folded_shock: FloatND, work: DiscreteAction) -> FloatND:
    return 1.0 + folded_shock + 0.0 * jnp.asarray(work, dtype=float)


def _build_zero_weight_fold_model(*, n_subjects: int | None = None) -> Model:
    """Build a folded process whose quadrature weights are exactly ``[0, 1, 0]``."""
    src = Regime(
        transition={
            "folded": MarkovTransition(_route_to_folded),
            "dead": MarkovTransition(_route_to_dead),
        },
        active=lambda age: age < 1,
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"wealth": fixed_transition("wealth")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _fold_source_utility},
    )
    folded = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "folded_shock": NormalIIDProcess(
                n_points=3,
                gauss_hermite=False,
                mu=0.0,
                sigma=1.0,
                n_std=100.0,
                fold=True,
            )
        },
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _folded_terminal_utility},
    )
    dead = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"src": src, "folded": folded, "dead": dead},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=FoldRegimeId,
        n_subjects=n_subjects,
    )


def _simulate_zero_weight_fold(model: Model) -> tuple[list[int], list[int], list[int]]:
    params = cast("dict[str, Any]", model.get_params_template())
    params["src"]["koopmans_aggregator"]["discount_factor"] = 0.9
    result = model.simulate(
        params=params,
        initial_conditions={
            "age": jnp.zeros(1),
            "wealth": jnp.zeros(1),
            "regime_id": jnp.asarray([FoldRegimeId.src], dtype=jnp.int32),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=409,
    )
    raw = result.raw_results["src"][0].actions["work"].tolist()
    frame = result.to_dataframe(use_labels=False)
    frame = frame[frame["period"] == 0].sort_values("subject_id")
    enriched = result.to_dataframe(additional_targets=["utility"], use_labels=False)
    enriched = enriched[enriched["period"] == 0].sort_values("subject_id")
    return (
        raw,
        frame["work"].to_numpy().tolist(),
        enriched["work"].to_numpy().tolist(),
    )


@pytest.mark.parametrize("compiled_n_subjects", [None, 1], ids=["lazy", "aot"])
def test_zero_weight_fold_average_preserves_the_public_candidate(
    compiled_n_subjects: int | None,
):
    """The zero-mass fold nodes contribute nothing to the selected policy."""
    observed = _simulate_zero_weight_fold(
        _build_zero_weight_fold_model(n_subjects=compiled_n_subjects)
    )
    assert observed == ([1], [1], [1])


def test_negating_the_fold_average_reverses_the_public_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    """The synchronized folded-average mutation is behaviorally live."""
    baseline = _simulate_zero_weight_fold(_build_zero_weight_fold_model())
    original = max_Q_over_a_module.zero_safe_average

    def negated_average(*args: Any, **kwargs: Any) -> FloatND:
        return -original(*args, **kwargs)

    monkeypatch.setattr(max_Q_over_a_module, "zero_safe_average", negated_average)
    shifted = _simulate_zero_weight_fold(_build_zero_weight_fold_model())

    assert baseline == ([1], [1], [1])
    assert shifted == ([0], [0], [0])


@pytest.mark.parametrize("compiled_n_subjects", [None, 1], ids=["lazy", "aot"])
def test_negating_kernel_result_reverses_the_public_candidate(
    compiled_n_subjects: int | None,
    monkeypatch: pytest.MonkeyPatch,
):
    """The solved-value transport into backward induction is behaviorally live."""
    model = _build_zero_weight_fold_model(n_subjects=compiled_n_subjects)
    baseline = _simulate_zero_weight_fold(model)
    original = grid_search_module.KernelResult

    def negated_kernel_result(**kwargs: Any) -> Any:
        result = original(**kwargs)
        object.__setattr__(result, "V_arr", -result.V_arr)
        return result

    monkeypatch.setattr(grid_search_module, "KernelResult", negated_kernel_result)
    shifted = _simulate_zero_weight_fold(model)

    assert baseline == ([1], [1], [1])
    assert shifted == ([0], [0], [0])


_RNG_N_SUBJECTS = 20
_RNG_SUBJECT_BATCH_SIZE = 7
_RNG_RANKS = (2.0, 1.0, 0.0, 2.0, 1.0, 0.0)
_RNG_MASK = tuple(True for _ in _CANDIDATES)


def _simulate_seeded_taste_routing(
    *, model: Model, subject_batch_size: int
) -> tuple[list[int], list[int]]:
    """Publish raw and DataFrame choices for one fixed subject-key stream."""
    result = model.simulate(
        params=_params_for_mask(model=model, mask=_RNG_MASK, ranks=_RNG_RANKS),
        initial_conditions={
            "age": jnp.zeros(_RNG_N_SUBJECTS),
            "wealth": jnp.ones(_RNG_N_SUBJECTS),
            "regime_id": jnp.full(_RNG_N_SUBJECTS, RegimeId.acting),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
        seed=409,
        subject_batch_size=subject_batch_size,
    )
    raw = result.raw_results["acting"][0].actions["work"].tolist()
    frame = result.to_dataframe(additional_targets=["utility"], use_labels=False)
    period_0 = frame[frame["period"] == 0].sort_values("subject_id")
    return raw, period_0["work"].to_numpy().tolist()


@pytest.mark.parametrize(
    "compiled_n_subjects", [None, _RNG_N_SUBJECTS], ids=["lazy", "aot"]
)
def test_seeded_taste_keys_preserve_subject_identity_across_batching(
    compiled_n_subjects: int | None,
):
    """A fixed subject keeps its taste draw under chunking and AOT dispatch."""
    model = _build_model(
        utility=_ranked_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
        taste_shocks=True,
        n_subjects=compiled_n_subjects,
    )
    unbatched = _simulate_seeded_taste_routing(model=model, subject_batch_size=0)
    chunked = _simulate_seeded_taste_routing(
        model=model, subject_batch_size=_RNG_SUBJECT_BATCH_SIZE
    )

    assert unbatched == chunked
    assert unbatched[0] == unbatched[1]
    assert set(unbatched[0]) == {0, 1}


def test_reassigning_taste_keys_changes_the_public_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    """The synchronized key-permutation attack is behaviorally live."""
    model = _build_model(
        utility=_ranked_utility,
        constraints={"candidate_mask": _candidate_mask},
        terminal_utility=lambda: jnp.array(0.0),
        taste_shocks=True,
    )
    baseline = _simulate_seeded_taste_routing(
        model=model, subject_batch_size=_RNG_SUBJECT_BATCH_SIZE
    )
    original = simulation_module.generate_simulation_keys

    def reassigned_keys(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        next_key, keys = original(**kwargs)
        return next_key, {
            name: jnp.roll(values, 1, axis=0) for name, values in keys.items()
        }

    monkeypatch.setattr(simulation_module, "generate_simulation_keys", reassigned_keys)
    shifted = _simulate_seeded_taste_routing(
        model=model, subject_batch_size=_RNG_SUBJECT_BATCH_SIZE
    )

    assert shifted[0] == shifted[1]
    assert shifted[0] != baseline[0]


def test_taste_shock_solve_matches_reference_for_every_candidate_and_support(
    masked_taste_shock_model: Model,
):
    """The logsum retains every candidate over singleton through full supports."""
    covered: set[int] = set()
    for ranks in _RANK_VECTORS:
        target = max(range(len(ranks)), key=ranks.__getitem__)
        covered.add(target)
        for mask in _taste_masks(target):
            winner, solved, _selected = _reference_taste_outcome(mask=mask, ranks=ranks)
            assert winner == target
            observed = _solve_mask_case(
                model=masked_taste_shock_model, mask=mask, ranks=ranks
            )
            aaae(
                observed,
                _WEALTH_VALUES + solved,
                decimal=DECIMAL_PRECISION,
            )
    assert covered == set(range(len(_CANDIDATES)))


def test_taste_shock_simulate_matches_reference_for_every_candidate_and_support(
    masked_taste_shock_model: Model,
):
    """Seeded Gumbel-max preserves every flat index and its unshocked value."""
    covered: set[int] = set()
    for ranks in _RANK_VECTORS:
        target = max(range(len(ranks)), key=ranks.__getitem__)
        covered.add(target)
        for mask in _taste_masks(target):
            winner, _solved, selected = _reference_taste_outcome(mask=mask, ranks=ranks)
            assert winner == target
            work, consumption = _CANDIDATES[winner]
            observed = _simulate_taste_mask_case(
                model=masked_taste_shock_model, mask=mask, ranks=ranks
            )
            assert observed["work"] == [work] * len(_WEALTH_VALUES)
            assert observed["consumption"] == [consumption] * len(_WEALTH_VALUES)
            aaae(
                observed["value"],
                _WEALTH_VALUES + selected,
                decimal=DECIMAL_PRECISION,
            )
    assert covered == set(range(len(_CANDIDATES)))


def test_taste_shock_aot_simulation_preserves_the_selected_flat_index(
    masked_taste_shock_aot_model: Model,
):
    """The public ``Model(n_subjects=...)`` route publishes the same candidate."""
    mask = (False, False, False, False, True, False)
    ranks = (-3.0, -2.0, -1.0, 0.0, 2.0, 1.0)

    observed = _simulate_taste_mask_case(
        model=masked_taste_shock_aot_model, mask=mask, ranks=ranks
    )

    assert observed["work"] == [_WORK_VALUES[1]] * len(_WEALTH_VALUES)
    assert observed["consumption"] == [_CONSUMPTION_VALUES[1]] * len(_WEALTH_VALUES)
    aaae(
        observed["value"],
        _WEALTH_VALUES + ranks[4],
        decimal=DECIMAL_PRECISION,
    )


def test_taste_shock_simulation_applies_candidate_distinct_noise(
    masked_taste_shock_model: Model,
):
    """Controlled noise can move the discrete winner while preserving its Q value."""
    mask = (True, False, False, False, True, False)
    ranks = (2.0, 0.0, -1.0, -2.0, 1.5, -3.0)
    winner, _solved, selected = _reference_taste_outcome(mask=mask, ranks=ranks)
    assert winner == 4

    observed = _simulate_taste_mask_case(
        model=masked_taste_shock_model, mask=mask, ranks=ranks
    )

    assert observed["work"] == [_WORK_VALUES[1]] * len(_WEALTH_VALUES)
    assert observed["consumption"] == [_CONSUMPTION_VALUES[1]] * len(_WEALTH_VALUES)
    aaae(
        observed["value"],
        _WEALTH_VALUES + selected,
        decimal=DECIMAL_PRECISION,
    )


def test_collective_solve_matches_reference_over_every_nonempty_feasibility_mask(
    masked_collective_model: Model,
):
    """Collective solve selects the same strict winner for both stakeholders."""
    for ranks in _RANK_VECTORS:
        for mask in _NONEMPTY_FEASIBILITY_MASKS:
            _index, value, _work, _consumption = _reference_for_mask(mask, ranks)
            observed = _solve_mask_case(
                model=masked_collective_model, mask=mask, ranks=ranks
            )
            expected = jnp.stack(
                [_WEALTH_VALUES + value, _WEALTH_VALUES + value], axis=-1
            )
            aaae(observed, expected, decimal=DECIMAL_PRECISION)


def test_collective_simulate_matches_reference_over_every_nonempty_feasibility_mask(
    masked_collective_model: Model,
):
    """Collective simulation agrees on action and each stakeholder's value."""
    for ranks in _RANK_VECTORS:
        for mask in _NONEMPTY_FEASIBILITY_MASKS:
            _index, value, work, consumption = _reference_for_mask(mask, ranks)
            observed = _simulate_mask_case(
                model=masked_collective_model, mask=mask, ranks=ranks
            )
            assert observed["work"] == [work] * len(_WEALTH_VALUES), (mask, ranks)
            assert observed["consumption"] == [consumption] * len(_WEALTH_VALUES), (
                mask,
                ranks,
            )
            expected = _WEALTH_VALUES + value
            aaae(observed["value_f"], expected, decimal=DECIMAL_PRECISION)
            aaae(observed["value_m"], expected, decimal=DECIMAL_PRECISION)


def _argmax_masking_the_last_action_cell() -> Callable[..., Any]:
    """Return an `argmax_and_max` that hides the last cell of the action product.

    Patching the simulate-side reducer alone is what makes the control specific: the
    solve reduction is a different callable and keeps the full candidate set, so a
    green solve sweep beside a red simulate sweep is exactly the divergence a
    solve-only certificate cannot see.

    Returns:
        A drop-in replacement for `argmax_and_max`.
    """
    real = max_Q_over_a_module.argmax_and_max

    def patched(a: Any, *args: Any, where: Any = None, **kwargs: Any) -> Any:
        # `argmax_and_max` also reduces rank-0 masks elsewhere in the engine; those
        # carry no action axis to hide a cell along, so they pass through untouched.
        if where is not None and where.ndim >= 1:
            where = where.at[..., -1].set(False)  # noqa: PD008
        return real(a, *args, where=where, **kwargs)

    return patched


def test_masking_one_simulate_candidate_changes_the_published_action():
    """Hiding one cell from the simulate reducer alone moves the published action.

    Without this the simulate sweep's green would only say that nothing raised. The
    masked candidate is the one the constraint admits, so an exhaustive simulate
    search would publish it and this run publishes a different action.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "argmax_and_max",
            _argmax_masking_the_last_action_cell(),
        )
        model = _build_model(
            utility=_labelled_utility,
            constraints={"only_target": _only_target},
            terminal_utility=lambda: jnp.array(0.0),
        )
        rows = _simulate_acting(
            model=model,
            work=_WORK_VALUES[-1],
            consumption=_CONSUMPTION_VALUES[-1],
            function_name="only_target",
        )

    assert rows["consumption"] != [_CONSUMPTION_VALUES[-1]] * len(_WEALTH_VALUES)


def test_masking_one_simulate_candidate_leaves_the_others_alone():
    """The simulate mask removes one cell, not the simulation.

    A patch that broke the simulate reduction outright would move every published
    action and make the control above pass for the wrong reason.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "argmax_and_max",
            _argmax_masking_the_last_action_cell(),
        )
        model = _build_model(
            utility=_labelled_utility,
            constraints={"only_target": _only_target},
            terminal_utility=lambda: jnp.array(0.0),
        )
        rows = _simulate_acting(
            model=model,
            work=_WORK_VALUES[0],
            consumption=_CONSUMPTION_VALUES[0],
            function_name="only_target",
        )

    assert rows["consumption"] == [_CONSUMPTION_VALUES[0]] * len(_WEALTH_VALUES)


def _productmap_dropping_the_last_action_cell(
    *, action_names: frozenset[str]
) -> Callable[..., Any]:
    """Return a `productmap` that hides the last cell of the action product.

    Args:
        action_names: Names identifying the action product map, so the state
            product map built by the same function is left alone.

    Returns:
        A drop-in replacement for `productmap`.
    """
    real = max_Q_over_a_module.productmap

    def patched(
        *,
        func: Any,
        variables: tuple[str, ...],
        batch_sizes: dict[str, int],
    ) -> Any:
        mapped = real(func=func, variables=variables, batch_sizes=batch_sizes)
        if frozenset(variables) != action_names:
            return mapped

        @functools.wraps(mapped, assigned=("__name__", "__qualname__", "__doc__"))
        def masked(**call_kwargs: Any) -> Any:
            Q_arr, F_arr = mapped(**call_kwargs)
            return Q_arr, F_arr.at[..., -1].set(False)  # noqa: PD008

        return masked

    return patched


def _solve_with_the_last_action_cell_dropped(
    *, work: float, consumption: float
) -> FloatND:
    """Solve the unique-feasible model with one cell hidden from the search.

    Args:
        work: Target value of the discrete action.
        consumption: Target value of the continuous action.

    Returns:
        The acting regime's period-0 value array under the mask.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            max_Q_over_a_module,
            "productmap",
            _productmap_dropping_the_last_action_cell(
                action_names=frozenset({"work", "consumption"})
            ),
        )
        model = _build_model(
            utility=_labelled_utility,
            constraints={"only_target": _only_target},
            terminal_utility=lambda: jnp.array(0.0),
        )
        return _solve_acting(
            model=model,
            work=work,
            consumption=consumption,
            function_name="only_target",
        )


def test_dropping_one_candidate_is_visible_to_the_sweep():
    """Hiding one cell of the action product turns the sweep red on that cell.

    Without this the sweep's green would only say that nothing raised. The
    masked candidate is the one the params name, so an exhaustive search would
    still publish its value and this run publishes `-inf` instead.
    """
    V_arr = _solve_with_the_last_action_cell_dropped(
        work=_WORK_VALUES[-1], consumption=_CONSUMPTION_VALUES[-1]
    )

    assert bool(jnp.all(jnp.isneginf(V_arr)))


def test_dropping_one_candidate_leaves_every_other_candidate_alone():
    """The mask removes one cell, not the solve.

    A patch that broke the search outright would publish `-inf` everywhere and
    make the control above pass for the wrong reason. Naming a candidate the
    mask does not touch separates the two.
    """
    V_arr = _solve_with_the_last_action_cell_dropped(
        work=_WORK_VALUES[0], consumption=_CONSUMPTION_VALUES[0]
    )

    aaae(
        V_arr,
        _WEALTH_VALUES + _CONSUMPTION_VALUES[0] + 10.0 * _WORK_VALUES[0],
        decimal=DECIMAL_PRECISION,
    )
