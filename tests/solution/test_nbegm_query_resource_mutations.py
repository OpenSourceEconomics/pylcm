"""Installed source mutants of the actual-resource status are caught by the matrix.

Each mutant rewrites one expression in the production source of the NB query,
the choice seam or the status reductions, compiles it in that module's own
namespace and installs it in place of the original for one test. A mutant is
killed when at least one probe's host decision disagrees with the literal
budget class; an identical rewrite must disagree nowhere, which shows the
installation itself changes nothing. Probes run unjitted, so every trace
sees the installed source.
"""

import functools
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm import continuation as continuation_module
from _lcm.egm import nbegm_query as nbegm_query_module
from _lcm.egm.nbegm_query import NBEGMQuery
from _lcm.solution import feasibility
from _lcm.solution.feasibility import BoundedResult
from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution._nb_query_resource_oracle import ResourceKind
from tests.solution._nb_query_resource_seam import (
    JOB_CODES,
    LANES,
    Seam,
    build_seam,
    expected_lane_flag,
    host_rejects,
    permute_jobs,
    read_lane,
)
from tests.solution.test_nbegm_query_actual_resources import (
    PARAMS,
    _qualified_model,
)
from tests.solution.test_nbegm_query_resource_matrix import (
    LANE_ORDER,
    NODE_LANES,
    fold_nodes,
    lane_arrays,
    slot_of,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_PREDICATE = "            | ~jnp.isfinite(resources)\n"


@dataclass(frozen=True)
class _Mutant:
    """One textual rewrite of one production function."""

    module: ModuleType
    """The module whose namespace compiles the rewritten source."""
    owner: type[NBEGMQuery] | ModuleType
    """Where the rewritten function is installed."""
    name: str
    """The function's attribute name on `owner`."""
    old: str
    """The exact production text, which must occur once."""
    new: str
    """Its replacement."""
    installs: tuple[ModuleType, ...] = ()
    """Further modules that hold the function under the same imported name."""


def _query_mutant(new: str) -> _Mutant:
    return _Mutant(
        module=nbegm_query_module,
        owner=nbegm_query_module.NBEGMQuery,
        name="_read_at_floor_with_summary",
        old=_PREDICATE,
        new=new,
    )


MUTANTS = {
    "identity": _query_mutant(_PREDICATE),
    "delete_predicate": _query_mutant(""),
    "gate_on_domain": _query_mutant(
        "            | (domain & ~jnp.isfinite(resources))\n"
    ),
    "gate_on_eligible": _query_mutant(
        "            | (eligible & ~jnp.isfinite(resources))\n"
    ),
    "gate_on_corner": _query_mutant(
        "            | (corner.supported & ~jnp.isfinite(resources))\n"
    ),
    "gate_on_row_winner": _query_mutant(
        "            | (take_row & ~jnp.isfinite(resources))\n"
    ),
    "gate_on_runtime_checks": _query_mutant(
        "            | (~jnp.isfinite(resources)"
        " & (read_rows == self._read_rows_with_counts))\n"
    ),
    "optional_counter": _query_mutant(
        "            | (jnp.count_nonzero(~jnp.isfinite(resources)) > 1)\n"
    ),
    "final_action_only": _Mutant(
        module=continuation_module,
        owner=continuation_module,
        name="_read_feature_child_choices",
        old="coverage_missing=jnp.any(result.coverage_missing),",
        new="coverage_missing=result.coverage_missing[-1],",
    ),
    "winning_action_only": _Mutant(
        module=continuation_module,
        owner=continuation_module,
        name="_read_feature_child_choices",
        old="coverage_missing=jnp.any(result.coverage_missing),",
        new="coverage_missing=result.coverage_missing[jnp.argmax(result.value)],",
    ),
    "drop_from_map_reduction": _Mutant(
        module=feasibility,
        owner=feasibility,
        name="reduce_query_input_status",
        old="QueryInputStatus(*(jnp.any(active & flag) for flag in status))",
        new=(
            "QueryInputStatus(*(jnp.any(active & flag) for flag in status[:2]),"
            " jnp.any(active & jnp.zeros_like(status[2])))"
        ),
        installs=(continuation_module,),
    ),
    "drop_from_scan_merge": _Mutant(
        module=feasibility,
        owner=feasibility,
        name="merge_query_input_status",
        old="QueryInputStatus(*(a | b for a, b in zip(lhs, rhs, strict=True)))",
        new="QueryInputStatus(*(a for a, _ in zip(lhs, rhs, strict=True)))",
        installs=(continuation_module,),
    ),
    "flag_finite_empty_budget": _query_mutant(
        "            | ~jnp.isfinite(resources)\n            | ~domain\n"
    ),
    "flag_every_minus_infinity": _query_mutant(
        "            | ~jnp.isfinite(resources)\n"
        "            | jnp.isneginf(corner_value)\n"
    ),
}

KILLED = tuple(name for name in MUTANTS if name != "identity")


def install(*, mutant: _Mutant, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile the rewritten production source and install it for one test."""
    original = (
        vars(mutant.owner)[mutant.name]
        if isinstance(mutant.owner, type)
        else getattr(mutant.owner, mutant.name)
    )
    source = inspect.getsource(inspect.unwrap(original))
    if source.count(mutant.old) != 1:
        msg = f"{mutant.name} does not contain its mutated text exactly once."
        raise ValueError(msg)
    namespace = dict(vars(mutant.module))
    code = compile(
        textwrap.dedent(source.replace(mutant.old, mutant.new)),
        mutant.module.__file__ or "",
        "exec",
    )
    exec(code, namespace)  # noqa: S102
    replacement = namespace[mutant.name]
    monkeypatch.setattr(mutant.owner, mutant.name, replacement)
    for module in mutant.installs:
        monkeypatch.setattr(module, mutant.name, replacement)


@dataclass(frozen=True)
class _Probe:
    """One host decision and its literal expectation."""

    name: str
    """A label for the report."""
    observe: Callable[[], bool]
    """Return whether the mandatory host validation rejects."""
    expected: bool
    """Whether the literal budget class says it must reject."""


def _lane_probe(
    *,
    kind: ResourceKind,
    lane: str,
    collect_counts: bool = True,
    order: tuple[int, ...] = JOB_CODES,
) -> _Probe:
    point = LANES[lane]

    def observe() -> bool:
        seam = _seam(kind)
        carry = (
            seam.carry
            if order == JOB_CODES
            else permute_jobs(carry=seam.carry, order=order)
        )
        result = read_lane(
            seam=seam,
            liquid=point.liquid,
            stock=point.stock,
            z=point.z,
            collect_counts=collect_counts,
            carry=carry,
            jobs=order,
        )
        return host_rejects(seam=seam, status=slot_of(result))

    return _Probe(
        name=f"{kind.value}/{lane}/counts={collect_counts}/order={order}",
        observe=observe,
        expected=expected_lane_flag(kind=kind, lane=point),
    )


def _vmap_probe() -> _Probe:
    """Only the last lane is invalid and active; the first is valid and active."""
    active = (True, False, False, False, True)

    def observe() -> bool:
        seam = _seam(ResourceKind.NAN)
        liquid, stock, z = lane_arrays(LANE_ORDER)
        results = jax.vmap(functools.partial(read_lane, seam=seam))(
            liquid=liquid, stock=stock, z=z
        )
        reduced = feasibility.reduce_query_input_status(
            statuses=results.query_status, active=jnp.asarray(active)
        )
        assert reduced is not None
        return host_rejects(seam=seam, status=reduced[0])

    return _Probe(name="nan/lane-reduction", observe=observe, expected=True)


def _scan_probe() -> _Probe:
    """The only live invalid node sits in the partial final tile."""

    def observe() -> bool:
        seam = _seam(ResourceKind.NAN)

        def read(
            *, liquid: FloatND, stock: FloatND, z: FloatND
        ) -> BoundedResult[tuple[FloatND, FloatND]]:
            return read_lane(seam=seam, liquid=liquid, stock=stock, z=z)

        status = fold_nodes(
            read=read, node_weights=(0.0, 0.5, 0.0, 0.0, 0.5), collect_counts=False
        )
        return host_rejects(seam=seam, status=status)

    assert NODE_LANES[-1] == "exact_zero_factor"
    return _Probe(name="nan/partial-final-tile", observe=observe, expected=True)


def _seam_probes() -> tuple[_Probe, ...]:
    nan = ResourceKind.NAN
    return (
        _lane_probe(kind=nan, lane="positive_factors", collect_counts=False),
        _lane_probe(kind=nan, lane="positive_factors"),
        _lane_probe(kind=nan, lane="predicate_false"),
        _lane_probe(kind=nan, lane="negative_factor"),
        _lane_probe(kind=nan, lane="finite_node"),
        _lane_probe(kind=nan, lane="positive_factors", order=(1, 0)),
        _lane_probe(kind=ResourceKind.POSITIVE_INFINITY, lane="positive_factors"),
        _lane_probe(kind=ResourceKind.POSITIVE, lane="positive_factors"),
        _lane_probe(kind=ResourceKind.ZERO_AT_FLOOR, lane="positive_factors"),
        _lane_probe(kind=ResourceKind.NEGATIVE_FINITE, lane="positive_factors"),
        _vmap_probe(),
        _scan_probe(),
    )


def _seam(kind: ResourceKind) -> Seam:
    return build_seam(kind=kind, x64=bool(jax.config.read("jax_enable_x64")))


def _disagreements(probes: tuple[_Probe, ...]) -> list[str]:
    """List the probes whose host decision differs from the literal class."""
    return [probe.name for probe in probes if probe.observe() != probe.expected]


def test_identity_rewrite_disagrees_nowhere(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reinstalling the unchanged predicate source leaves every probe correct."""
    install(mutant=MUTANTS["identity"], monkeypatch=monkeypatch)
    assert _disagreements(_seam_probes()) == []


@pytest.mark.parametrize("name", KILLED)
def test_mutant_is_killed_by_a_probe(
    *, name: str, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Every gated, dropped, counted or over-flagging status rule fails a probe."""
    install(mutant=MUTANTS[name], monkeypatch=monkeypatch)
    killing = _disagreements(_seam_probes())
    request.node.user_properties.append(("killing_probes", ";".join(killing)))
    assert killing


def test_deleted_predicate_lets_public_nan_budget_solve(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the predicate the public NaN fixture solves, failing its test."""
    install(mutant=MUTANTS["delete_predicate"], monkeypatch=monkeypatch)
    model = _qualified_model(kind=ResourceKind.NAN)
    rejected = False
    try:
        model.solve(params=PARAMS, log_level="off", runtime_checks=False)
    except InvalidValueFunctionError:
        rejected = True
    assert not rejected


@pytest.mark.parametrize("lane", sorted(LANES))
def test_predicate_leaves_raw_read_counts_unchanged(
    *, lane: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counts with and without the predicate agree: the status is not a read."""
    point = LANES[lane]
    seam = _seam(ResourceKind.NAN)

    def counts() -> tuple[int, int]:
        result = read_lane(
            seam=seam,
            liquid=point.liquid,
            stock=point.stock,
            z=point.z,
            collect_counts=True,
        )
        assert result.continuation_read_counts is not None
        below, eligible = result.continuation_read_counts
        return int(below), int(eligible)

    with_predicate = counts()
    install(mutant=MUTANTS["delete_predicate"], monkeypatch=monkeypatch)
    assert counts() == with_predicate
