"""Consent pricing evaluates observed couple rows once and runs its cells in lanes.

A formation exit that observes every couple shock fixes the couple target's rows
to one physical point, so one representative row prices all of them. The cells of
a query chunk and the rows of an information support run as batched lanes; a lane
whose guard is false reads stand-in inputs and its answer is discarded.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm import policy_queries, query_information
from _lcm.typing import PytreeValue
from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    categorical,
)
from lcm.exceptions import RegimeInitializationError
from lcm.shocks import Discrete, GaussHermite, Law, Normal, Product, ShockGroup
from lcm.solver_api import KernelOutput
from lcm.typing import BoolND, FloatND, IntND, ScalarInt
from tests.conftest import assert_agrees_to_ulp

# Cells per query chunk; the source regime has two chunks of four cells.
_WIDTH = 4


@categorical(ordered=False)
class _Regimes:
    single_f: ScalarInt
    single_m: ScalarInt
    couple: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    couple_terminal: ScalarInt


def _single_f(*, experience: IntND, work: IntND, wage_f: FloatND) -> FloatND:
    return 1 + work * (2 + 0.2 * experience + wage_f) + 2.45 * (1 - work)


def _single_m(*, experience: IntND, work: IntND, wage_m: FloatND) -> FloatND:
    return 1 + work * (3 + 0.2 * experience + wage_m) + 2.45 * (1 - work)


def _consumption(
    *,
    experience_f: IntND,
    experience_m: IntND,
    work_f: IntND,
    work_m: IntND,
    wage_f: FloatND,
    wage_m: FloatND,
    match: FloatND,
) -> FloatND:
    return (
        1
        + work_f * (2 + 0.2 * experience_f + wage_f)
        + work_m * (3 + 0.2 * experience_m + wage_m)
        + match
    )


def _couple_f(*, consumption: FloatND, work_f: IntND) -> FloatND:
    return consumption / 2 + 2.45 * (1 - work_f) + 1


def _poisoned_couple_f(
    *, consumption: FloatND, work_f: IntND, experience_f: IntND
) -> FloatND:
    """Finite on the couple grid, -inf at the unavailable sentinel -1."""
    return _couple_f(consumption=consumption, work_f=work_f) + 0.1 * jnp.log1p(
        experience_f.astype(consumption.dtype)
    )


def _divergent_couple_f(
    *, consumption: FloatND, work_f: IntND, experience_f: IntND
) -> FloatND:
    """Counts up to experience; at the unavailable sentinel -1 it never stops."""
    target = experience_f.astype(consumption.dtype)
    steps = jax.lax.while_loop(
        lambda count: count != target, lambda count: count + 1, jnp.zeros_like(target)
    )
    return _couple_f(consumption=consumption, work_f=work_f) + 0 * steps


def _couple_m(*, consumption: FloatND, work_m: IntND) -> FloatND:
    return consumption / 2 + 2.5 * (1 - work_m) + 1


def _experience(experience: IntND) -> IntND:
    return experience


def _sentinel_experience(experience: IntND) -> IntND:
    """The couple state only where the formation exit is available."""
    return jnp.where(experience > 0, experience, -1)


def _experienced(experience: IntND) -> BoolND:
    return experience > 0


def _next_experience(*, experience: IntND, work: IntND) -> IntND:
    return jnp.minimum(experience + work, 5)


def _next_f(*, experience_f: IntND, work_f: IntND) -> IntND:
    return jnp.minimum(experience_f + work_f, 5)


def _next_m(*, experience_m: IntND, work_m: IntND) -> IntND:
    return jnp.minimum(experience_m + work_m, 5)


def _match_sigma(experience_f: IntND) -> FloatND:
    return 0.3 + 0.01 * experience_f


def _level(level: IntND) -> IntND:
    return level


def _flipped_level(level: IntND) -> IntND:
    return 1 - level


def _no_level(experience: IntND) -> IntND:
    return 0 * experience


def _level_sigma(level: IntND) -> FloatND:
    """Zero spread at level zero: the wage there reveals no base."""
    return 0.5 * level


def _group(*, name: str, n: int, law: Law) -> ShockGroup:
    if isinstance(law, Discrete):
        return ShockGroup(shocks={name: law})
    return ShockGroup(shocks={name: law}, nodes=Product(**{name: GaussHermite(n=n)}))


def _model(
    *,
    n_match: int = 2,
    match_law: Law | None = None,
    joint_match: bool = False,
    marry_available: Callable[..., BoolND] | None = None,
    couple_projection: Callable[..., IntND] = _experience,
    couple_f: Callable[..., FloatND] = _couple_f,
    wage_level: bool = False,
    width: int = _WIDTH,
) -> Model:
    """Singles may marry by consent, observing every shock of the couple.

    With `wage_level`, single_f and the couple carry a binary `level` whose value
    sets the wage spread, and marriage flips it.
    """
    experience = DiscreteGrid(values=jnp.arange(6))
    work = DiscreteGrid(values=jnp.arange(2))
    level = {"level": DiscreteGrid(values=jnp.arange(2))} if wage_level else {}
    keep_level = {"level": _level} if wage_level else {}
    singles = {
        f"single_{member}": Regime(
            states={"experience": experience, **(level if member == "f" else {})},
            actions={"work": work},
            functions={"utility": _single_f if member == "f" else _single_m},
            shocks=("wage_f",) if member == "f" else ("wage_m",),
            state_transitions={
                "experience": _next_experience,
                **(keep_level if member == "f" else {}),
            },
            exits={
                "marry": Exit(
                    targets={
                        "self": ExitTarget(
                            regime="couple",
                            role=member,
                            projection={
                                "experience_f": couple_projection
                                if member == "f"
                                else _experience,
                                "experience_m": _experience,
                                **(
                                    {
                                        "level": _flipped_level
                                        if member == "f"
                                        else _no_level
                                    }
                                    if wage_level
                                    else {}
                                ),
                            },
                        )
                    },
                    rule=Consent(
                        counterparty=Counterparty(
                            regime="single_m" if member == "f" else "single_f",
                            role="m" if member == "f" else "f",
                            projection={
                                "experience": _experience,
                                **(
                                    {"level": _no_level}
                                    if wage_level and member == "m"
                                    else {}
                                ),
                            },
                        )
                    ),
                    available=marry_available if member == "f" else None,
                    observes=("wage_f", "wage_m", "match"),
                )
            },
        )
        for member in ("f", "m")
    }
    regimes = {
        **singles,
        "couple": Regime(
            states={"experience_f": experience, "experience_m": experience, **level},
            actions={"work_f": work, "work_m": work},
            functions={
                "utility": CollectiveUtility(utilities={"f": couple_f, "m": _couple_m}),
                "consumption": _consumption,
            },
            state_transitions={
                "experience_f": _next_f,
                "experience_m": _next_m,
                **keep_level,
            },
            shocks=("wage_f", "wage_m", "match"),
        ),
    }
    for name, regime in tuple(regimes.items()):
        regimes[f"{name}_terminal"] = replace(
            regime,
            state_transitions={},
            exits={
                key: replace(
                    declaration,
                    targets={
                        member: replace(target, regime=f"{target.regime}_terminal")
                        for member, target in declaration.targets.items()
                    },
                    rule=replace(
                        declaration.rule,
                        counterparty=replace(
                            declaration.rule.counterparty,
                            regime=f"{declaration.rule.counterparty.regime}_terminal",
                        ),
                    ),
                )
                for key, declaration in regime.exits.items()
            },
        )
    law = Normal(mu=0.0, sigma=0.3) if match_law is None else match_law
    match = (
        ShockGroup(
            shocks={"match": law, "bonus": Normal(mu=0.0, sigma=0.1)},
            nodes=Product(match=GaussHermite(n=n_match), bonus=GaussHermite(n=2)),
        )
        if joint_match
        else _group(name="match", n=n_match, law=law)
    )
    return Model(
        regimes=regimes,
        shocks={
            "wage_f": _group(
                name="wage_f",
                n=2,
                law=Normal(mu=0.0, sigma=_level_sigma if wage_level else 0.5),
            ),
            "wage_m": _group(name="wage_m", n=2, law=Normal(mu=0.0, sigma=0.4)),
            "match": match,
        },
        n_periods=2,
        regime_id_class=_Regimes,
        initial_nodes=InitialNodes(by_period={0: ("couple", "single_f", "single_m")}),
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("couple", "single_f", "single_m")
        },
        execution_config=ExecutionConfig(
            device_memory_bytes=None, axis_widths={"query_cell": width}
        ),
    )


def _chunk_loop_trips(*, model: Model, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Known trip counts of every while loop in single_f's compiled chunk program."""
    texts = {}
    original = policy_queries._PolicyQueryKernel.__call__

    def capture(
        self: policy_queries._PolicyQueryKernel, **kwargs: PytreeValue
    ) -> KernelOutput:
        cores = cast(
            "Mapping[str, Callable[..., PytreeValue]]", kwargs["compiled_cores"]
        )
        compiled = cores["_chunks"].compiled  # ty: ignore[unresolved-attribute]
        texts[(self.root_context.regime_name, self.period)] = compiled.as_text()
        return original(self, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(policy_queries._PolicyQueryKernel, "__call__", capture)
    model.solve(params={"discount_factor": 0.9}, log_level="off")
    return sorted(
        (int(trip.group(1)) if trip else -1)
        for line in texts[("single_f", 0)].splitlines()
        if " while(" in line
        for trip in (re.search(r'known_trip_count":\{"n":"(\d+)"', line),)
    )


def test_couple_loop_does_not_grow_with_observed_couple_shocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every observed couple row is one point: no loop scales with the match grid."""
    small = _chunk_loop_trips(model=_model(n_match=2), monkeypatch=monkeypatch)
    large = _chunk_loop_trips(model=_model(n_match=4), monkeypatch=monkeypatch)
    assert small == large


def test_cells_of_a_chunk_run_as_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two chunks run in sequence; no loop walks the four cells of a chunk."""
    trips = _chunk_loop_trips(model=_model(), monkeypatch=monkeypatch)
    assert _WIDTH not in trips
    assert 2 in trips


def test_halving_the_chunk_doubles_the_support_rows_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step keeps its chunk-times-row lane count, so the row loop halves too."""
    monkeypatch.setattr(policy_queries, "_ROW_BATCH", 1)
    monkeypatch.setattr(policy_queries, "_STEP_LANES", _WIDTH)
    wide = _chunk_loop_trips(model=_model(width=_WIDTH), monkeypatch=monkeypatch)
    narrow = _chunk_loop_trips(model=_model(width=_WIDTH // 2), monkeypatch=monkeypatch)
    assert max(wide) > _WIDTH
    assert max(narrow) == max(wide) // 2


def _scalar_lanes(
    *,
    function: Callable[[PytreeValue], PytreeValue],
    operands: PytreeValue,
    batch_size: int | None = None,
) -> PytreeValue:
    """Every lane alone, so each guard branches instead of selecting."""
    del batch_size
    return jax.lax.map(function, operands)


def _values(model: Model) -> dict[str, np.ndarray]:
    solution = model.solve(params={"discount_factor": 0.9}, log_level="off")
    return {
        f"{period}/{regime}": np.asarray(value)
        for period, values in solution.values.items()
        for regime, value in values.items()
    }


def _assert_same_values(
    *, got: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> None:
    """Same arithmetic; XLA vectorizes a batch differently, so allow a few ULP.

    With the backend optimizer off both routes are bit-identical. A leaked -inf
    or a discarded lane's value would differ by far more than the bound.
    """
    assert got.keys() == expected.keys()
    for name, value in expected.items():
        assert np.all(np.isfinite(got[name])), name
        assert_agrees_to_ulp(got=got[name], expected=value, n_ulp=16, err_msg=name)


@pytest.mark.parametrize(
    "couple_f", [_poisoned_couple_f, _divergent_couple_f], ids=["inf", "divergent"]
)
@pytest.mark.timeout(600, method="thread")
def test_unavailable_lanes_never_reach_their_used_neighbours(
    *, couple_f: Callable[..., FloatND], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lane whose exit is unavailable maps to -1, where the couple utility is
    -inf or never stops; batched lanes finish and match the scalar branches."""

    def poisoned() -> Model:
        return _model(
            marry_available=_experienced,
            couple_projection=_sentinel_experience,
            couple_f=couple_f,
        )

    batched = _values(poisoned())
    monkeypatch.setattr(policy_queries, "map_lanes", _scalar_lanes)
    _assert_same_values(got=batched, expected=_values(poisoned()))


def test_pinned_rows_reproduce_the_full_row_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batched lanes over pinned rows equal the scalar branches over every row."""
    batched = _values(_model(n_match=3, marry_available=_experienced))
    monkeypatch.setattr(policy_queries, "map_lanes", _scalar_lanes)
    monkeypatch.setattr(query_information, "pinned_support", lambda **_: None)
    _assert_same_values(
        got=batched, expected=_values(_model(n_match=3, marry_available=_experienced))
    )


def _unpinned_scalar_values(
    *, model: Callable[[], Model], monkeypatch: pytest.MonkeyPatch
) -> dict[str, np.ndarray]:
    """Every support row evaluated in scalar branches: the reference computation."""
    with monkeypatch.context() as patch:
        patch.setattr(policy_queries, "map_lanes", _scalar_lanes)
        patch.setattr(query_information, "pinned_support", lambda **_: None)
        return _values(model())


def test_partly_observed_group_keeps_its_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exit observes match but not bonus, which shares match's joint group:
    those rows genuinely differ, so they are all evaluated."""

    def model() -> Model:
        return _model(joint_match=True)

    _assert_same_values(
        got=_values(model()),
        expected=_unpinned_scalar_values(model=model, monkeypatch=monkeypatch),
    )


def test_state_dependent_observed_law_pins_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observed couple law reading couple states is the producer's own law, so
    its rows are copies and pricing them once reproduces every row."""

    def model() -> Model:
        return _model(match_law=Normal(mu=0.0, sigma=_match_sigma))

    _assert_same_values(
        got=_values(model()),
        expected=_unpinned_scalar_values(model=model, monkeypatch=monkeypatch),
    )


def test_caller_law_that_can_differ_is_refused_at_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At level zero the single's wage has no spread, so it fixes no base, while
    the couple at the flipped level one draws a wage: the rows differ."""
    with pytest.raises(RegimeInitializationError, match="informativeness"):
        _model(wage_level=True)
    monkeypatch.setattr(query_information, "_refuse_pinning", lambda **_: None)
    pinned = _values(_model(wage_level=True))
    expected = _unpinned_scalar_values(
        model=lambda: _model(wage_level=True), monkeypatch=monkeypatch
    )
    assert any(not np.allclose(pinned[name], value) for name, value in expected.items())
