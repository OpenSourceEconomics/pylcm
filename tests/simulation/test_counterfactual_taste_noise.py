"""Public counterfactuals share actual noise at matching semantic addresses."""

import dataclasses
import importlib
import threading
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, Model, categorical
from lcm.typing import FloatND, ScalarInt
from tests.test_models import taste_shocks_toy

max_q = importlib.import_module("_lcm.regime_building.max_Q_over_a")


@categorical(ordered=False)
class _RenamedRegimeId:
    student: ScalarInt
    absorbed: ScalarInt


def _enter_absorbed() -> ScalarInt:
    return _RenamedRegimeId.absorbed


@dataclasses.dataclass(kw_only=True)
class _NoiseRecords:
    """Call-local observations associate noise with keys, never callback order."""

    original: Callable[..., FloatND]
    records: list[tuple[tuple[int, ...], np.ndarray]] = dataclasses.field(
        default_factory=list
    )
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    # keyword-only-exempt: library-callback=jax.debug.callback
    def collect(self, key_data: np.ndarray, noise: np.ndarray) -> None:
        with self.lock:
            self.records.append(
                (tuple(int(word) for word in key_data), np.array(noise, copy=True))
            )

    def draw(
        self, *, key: jax.Array, shape: tuple[int, ...], scale: FloatND
    ) -> FloatND:
        noise = self.original(key=key, shape=shape, scale=scale)
        jax.debug.callback(self.collect, jax.random.key_data(key), noise)
        return noise

    def take(self) -> dict[tuple[int, ...], np.ndarray]:
        jax.effects_barrier()
        output: dict[tuple[int, ...], np.ndarray] = {}
        with self.lock:
            for key, noise in self.records:
                if key in output:
                    np.testing.assert_array_equal(output[key], noise)
                output[key] = noise
            self.records.clear()
        return output


def _counterfactual_model(*, renamed: bool, aot: bool, count: int) -> Model:
    if not renamed:
        return taste_shocks_toy.get_model(n_subjects=count if aot else None)
    return Model(
        regimes={
            "student": dataclasses.replace(
                taste_shocks_toy.alive, transition=_enter_absorbed
            ),
            "absorbed": taste_shocks_toy.done,
        },
        ages=AgeGrid(start=39, stop=42, step="Y"),
        regime_id_class=_RenamedRegimeId,
        n_subjects=count if aot else None,
    )


@pytest.mark.parametrize("aot", [False, True], ids=["lazy", "aot"])
@pytest.mark.parametrize("ambient_prng", ["threefry2x32", "rbg"])
def test_actual_noise_survives_renamed_regimes_and_a_longer_horizon(
    *, monkeypatch: pytest.MonkeyPatch, aot: bool, ambient_prng: str
) -> None:
    """Shared age-40 choices get identical EV1 draws; a new taste seed changes them."""
    count = 6
    observations = _NoiseRecords(original=max_q.draw_taste_shock_noise)
    monkeypatch.setattr(max_q, "draw_taste_shock_noise", observations.draw)
    observed_runs = []
    for renamed, ordinary_seed, taste_seed in (
        (False, 11, 721),
        (True, 92, 721),
        (True, 92, 722),
    ):
        # Compare against an actual Threefry baseline even in the ambient-RBG case.
        with jax.default_prng_impl(ambient_prng if renamed else "threefry2x32"):
            model = _counterfactual_model(renamed=renamed, aot=aot, count=count)
            params = taste_shocks_toy.get_params(scale=0.2)
            if renamed:
                params["student"] = params.pop("alive")
            initial = {
                "age": jnp.full(count, 40.0),
                "wealth": jnp.full(count, 4.6),
                "regime_id": jnp.zeros(count, dtype=jnp.int32),
            }
            original_wealth = np.asarray(initial["wealth"]).copy()
            result = model.simulate(
                params=params,
                initial_conditions=initial,
                log_level="debug",
                seed=ordinary_seed,
                taste_shock_seed=taste_seed,
            )
            observed_runs.append(observations.take())
            regime = "student" if renamed else "alive"
            period = 1 if renamed else 0
            np.testing.assert_array_equal(
                result.raw_results[regime][period].in_regime,
                np.ones(count, dtype=bool),
            )
            np.testing.assert_array_equal(initial["wealth"], original_wealth)
    baseline, counterfactual, changed_seed = observed_runs
    assert len(baseline) == count
    assert baseline.keys() <= counterfactual.keys()
    for key, noise in baseline.items():
        assert noise.shape == (2,)
        np.testing.assert_array_equal(noise, counterfactual[key])
    assert changed_seed
    assert not baseline.keys() & changed_seed.keys()
    assert {noise.tobytes() for noise in counterfactual.values()} != {
        noise.tobytes() for noise in changed_seed.values()
    }
