"""Default current-IID replay builds each frame once in ordinary arithmetic.

The simulate-time replay programs trace no certification work: no exact-rational
affine read outside a configured `ExactEnvelope`'s own construction, no
certified query reducer, and no provenance-owned query bracket.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm import asset_row, upper_envelope
from _lcm.egm.upper_envelope import query as envelope_query
from _lcm.egm.upper_envelope._exact_affine import ffi as exact_affine_ffi
from lcm import Model, NestedConsumptionSavingsRegime
from lcm.shocks import PhysicalValues
from lcm.solvers import NBEGM, NEGM, NNBEGM, FiniteOuterGrid
from lcm.typing import UserParams
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

_EXACT_READ = "_exact_affine_read_impl"
_CERTIFIED_REDUCER = "_reduce_envelope_exact"
_OWNED_BRACKET = "_bracket_from_owned_row"
_ORDINARY_REDUCER = "_reduce_envelope_ordinary"

type _Initial = dict[str, jax.Array | PhysicalValues]
type _Case = tuple[Model, UserParams, _Initial]


def _spy_on_certification(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count every trace of the certification primitives from here on."""
    counts: Counter[str] = Counter()
    for module, name in (
        (exact_affine_ffi, _EXACT_READ),
        (envelope_query, _CERTIFIED_REDUCER),
        (envelope_query, _ORDINARY_REDUCER),
        (upper_envelope, _OWNED_BRACKET),
        (asset_row, _OWNED_BRACKET),
    ):
        original = getattr(module, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    return counts


def _initial(*, model: Model, **states: np.ndarray) -> _Initial:
    n = next(iter(states.values())).shape[0]
    shocks = np.linspace(0.0, 1.0, n).astype(canonical_float_dtype())
    return {
        **{name: jnp.asarray(values) for name, values in states.items()},
        "age": jnp.ones((n,)),
        "regime_id": jnp.full((n,), model.regime_names_to_ids["child"], jnp.int32),
        "eps": PhysicalValues(values=shocks),
    }


def _dc_u() -> _Case:
    from tests.simulation.test_case_u_reconstruction import (  # noqa: PLC0415
        _model,
        _params,
    )

    model = _model()
    return (
        model,
        _params(),
        _initial(model=model, wealth=np.asarray([1.0, 12.5, 1.5, 1.75])),
    )


def _nb_u() -> _Case:
    from tests.egm.test_current_iid_nbegm_model import (  # noqa: PLC0415
        _model,
        _nb_params,
    )

    model = _model()
    return (
        model,
        _nb_params(),
        _initial(model=model, wealth=np.asarray([0.5, 1.5, 3.0, 1.5])),
    )


def _nb_ride_u() -> _Case:
    from tests.egm.test_current_iid_nbegm_ride_model import (  # noqa: PLC0415
        _model,
        _params,
    )

    model = _model()
    return (
        model,
        _params(),
        _initial(
            model=model,
            wealth=np.asarray([1.5, 1.5, 3.0]),
            q=np.asarray([0.0, 1.0, 0.25]),
        ),
    )


def _nb_r() -> _Case:
    from tests.egm.test_current_iid_nbegm_model import (  # noqa: PLC0415
        _nb_params,
        _r_declarations,
    )

    model = Model._from_call_owned_params(
        declarations=_r_declarations(), params=_nb_params()
    )
    return (
        model,
        _nb_params(),
        _initial(model=model, wealth=np.asarray([0.375, 1.0, 1.0, 2.0])),
    )


def _nnbegm_u() -> _Case:
    from tests.egm.test_current_iid_call_owned_generation import (  # noqa: PLC0415
        _public_model_from_declarations,
    )
    from tests.egm.test_current_iid_negm_model import (  # noqa: PLC0415
        _declarations,
        _params,
    )

    declarations = _declarations()
    original = declarations.regimes["child"]
    assert isinstance(original, NestedConsumptionSavingsRegime)
    assert isinstance(original.solver, NEGM)
    solver = NNBEGM(
        inner=NBEGM(
            savings_grid=original.solver.inner.savings_grid, jump_read="bridged"
        ),
        outer_search=FiniteOuterGrid(grid=original.solver.outer_grid),
    )
    declarations = replace(
        declarations,
        regimes={**declarations.regimes, "child": replace(original, solver=solver)},
    )
    model = _public_model_from_declarations(declarations=declarations)
    return (
        model,
        _params(),
        _initial(model=model, wealth=np.full((3,), 2.0), housing=np.ones((3,))),
    )


# Families whose configured envelope is ordinary trace no exact-rational read at
# all. DC sources keep `ExactEnvelope` (the DCEGM default), whose own candidate
# construction is exact, but replay requests no provenance from it. Each case
# names a primitive that must fire, so an untraced replay cannot pass.
_ORDINARY_NB = frozenset({_EXACT_READ, _CERTIFIED_REDUCER, _OWNED_BRACKET})
_CASES: dict[str, tuple[Callable[[], _Case], frozenset[str], str]] = {
    "dc_u": (_dc_u, frozenset({_CERTIFIED_REDUCER, _OWNED_BRACKET}), _EXACT_READ),
    "nb_u": (_nb_u, _ORDINARY_NB, _ORDINARY_REDUCER),
    "nb_ride_u": (_nb_ride_u, _ORDINARY_NB, _ORDINARY_REDUCER),
    "nb_r": (_nb_r, _ORDINARY_NB, _ORDINARY_REDUCER),
    "nnbegm_u": (_nnbegm_u, _ORDINARY_NB, _ORDINARY_REDUCER),
}


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("family", list(_CASES))
def test_default_current_iid_replay_traces_no_certification(
    *, family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate after solve; only the replay programs are traced under the spy."""
    build, absent, control = _CASES[family]
    model, params, initial = build()
    solution = model.solve(params=params, log_level="off")
    counts = _spy_on_certification(monkeypatch)
    model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        log_level="off",
        seed=17,
    )
    assert counts[control] > 0, counts
    assert {name: counts[name] for name in absent} == dict.fromkeys(absent, 0), counts
