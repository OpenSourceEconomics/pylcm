"""The engine package has no import-order cycles.

`_lcm.typing` imports type aliases from `lcm.typing`, and reaching
`lcm.typing` boots the public `lcm` package. When an `_lcm` submodule is
imported before `lcm` (as the benchmark suite does via `lcm_examples`),
that boot must not cycle back into a half-initialized `_lcm.typing`.
"""

import subprocess
import sys

import lcm


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
