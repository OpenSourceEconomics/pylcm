"""Private implementation package for pylcm.

Every module here is engine-internal. The public surface — the classes and
helpers users construct or consume — lives in `lcm`.

This `__init__` stays minimal: it applies the jaxtyping patch and nothing else,
so importing `_lcm` never triggers the public `lcm` package on its own. The
beartype claw for both packages is registered from `lcm/__init__.py`, the
single bootstrap entry point.
"""

# The jaxtyping `"..."`-sentinel patch must run before any jaxtyping-subscripted
# type is created anywhere in pylcm.
from _lcm import jaxtyping_patch  # noqa: F401
