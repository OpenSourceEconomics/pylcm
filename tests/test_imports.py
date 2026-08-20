"""The engine package has no import-order cycles.

`_lcm.typing` imports type aliases from `lcm.typing`, and reaching
`lcm.typing` boots the public `lcm` package. When an `_lcm` submodule is
imported before `lcm` (as the benchmark suite does via `lcm_examples`),
that boot must not cycle back into a half-initialized `_lcm.typing`.
"""

import subprocess
import sys

import lcm
import lcm.consumption_savings_regime
import lcm.regime
import lcm.solvers
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    NestedConsumptionSavingsRegime,
)


def test_engine_submodule_imports_without_lcm_first():
    """An `_lcm` engine submodule imports cleanly in a fresh interpreter.

    Importing `_lcm.utils.dispatchers` as the first pylcm import — with no
    prior `import lcm` — must not raise a circular-import `ImportError`.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import _lcm.utils.dispatchers"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_every_public_name_is_bound():
    """Every name `lcm.__all__` advertises resolves, so `from lcm import *` works.

    A name left in `__all__` after its implementation is dropped turns a star
    import into an `AttributeError`, and the failure names only the first
    casualty rather than the list.
    """
    unbound = [name for name in lcm.__all__ if not hasattr(lcm, name)]
    assert unbound == []


def test_solver_names_live_only_in_the_solvers_module():
    """Solver declarations are published under `lcm.solvers`, not `lcm`."""
    solver_names = set(lcm.solvers.__all__)

    assert solver_names.isdisjoint(lcm.__all__)
    assert [name for name in solver_names if hasattr(lcm, name)] == []
    assert [name for name in solver_names if hasattr(lcm.regime, name)] == []
    assert [
        name for name in solver_names if hasattr(lcm.consumption_savings_regime, name)
    ] == []


def test_consumption_savings_declarations_share_their_own_module():
    """Specialized regime declarations are importable from their named module."""
    assert lcm.ConsumptionSavingsRegime is ConsumptionSavingsRegime
    assert lcm.NestedConsumptionSavingsRegime is NestedConsumptionSavingsRegime
    assert lcm.LiquidMargin is LiquidMargin
