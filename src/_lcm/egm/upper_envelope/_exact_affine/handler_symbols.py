"""The exact-affine kernel's FFI handler names, in one place.

Four consumers need this list and cannot share a runtime: the C++ and CUDA
translation units define the handlers, `ffi.py` registers them with XLA, and
the Windows build writes them into the module-definition file `link.exe`
consumes. A name present in some of those and absent from others produces a
library that loads and silently lacks a target, so the list lives here — a
module with no imports at all, which the build hook can load from the source
tree before the package is installed and before JAX exists.

`tests/test_exact_affine_handler_inventory.py` checks the C++ and CUDA sources
against this tuple, which is what makes it authoritative rather than merely
first.
"""

EXACT_AFFINE_HANDLER_SYMBOLS = (
    "CertifiedAffineCompareF32",
    "CertifiedAffineCompareF64",
    "ExactAffineReadF32",
    "ExactAffineReadF64",
    "ExactQueryWinnerF32",
    "ExactQueryWinnerF64",
    "ExactAffineHandoverF32",
    "ExactAffineHandoverF64",
    "ExactCellHullF32",
    "ExactCellHullF64",
)
