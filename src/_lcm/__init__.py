"""Private implementation package for pylcm.

Every module here is engine-internal. The public surface — the classes and
helpers users construct or consume — lives in `lcm`.

`_lcm` modules import shared type aliases, exceptions, and boundary classes
from `lcm.*`, so `_lcm` is not standalone-importable. To keep imports working
regardless of which package a caller reaches first, this `__init__` applies
the jaxtyping patch and then bootstraps the full `lcm` package: any `_lcm`
submodule used as an entry point boots `lcm/__init__.py` before its own body
runs, so that boot never re-enters a half-initialized engine module.
`lcm/__init__.py` re-imports `_lcm` during its own bootstrap, so `import lcm`
here binds the in-progress module and returns immediately.
"""

# Import order is load-bearing and must not be sorted: the jaxtyping
# `"..."`-sentinel patch must be applied before `import lcm` triggers the
# bootstrap that creates jaxtyping-subscripted types.
# isort: off
from _lcm import jaxtyping_patch  # noqa: F401

import lcm  # noqa: F401

# isort: on
