"""Every consumer of the exact-affine handler list sees the same names.

The C++ and CUDA translation units define the FFI handlers, `ffi.py` registers
them with XLA, and the Windows build writes them into the module-definition
file `link.exe` consumes. A name in some of those and not the others yields a
library that loads and silently lacks a target, which no other test can see:
the Linux and macOS builds export every symbol regardless.
"""

import re
from pathlib import Path

import pytest

import hatch_build
from _lcm.egm.upper_envelope._exact_affine import ffi
from _lcm.egm.upper_envelope._exact_affine.handler_symbols import (
    EXACT_AFFINE_HANDLER_SYMBOLS,
)

_PACKAGE = Path(hatch_build.__file__).resolve().parent / hatch_build.PACKAGE_DIR

# The symbol sits on the line *after* the macro opens, so a line-oriented
# pattern matches nothing and would certify an empty inventory as agreement.
_HANDLER = re.compile(r"XLA_FFI_DEFINE_HANDLER_SYMBOL\s*\(\s*([A-Za-z_]\w*)")


def _defined_in(filename: str) -> tuple[str, ...]:
    return tuple(sorted(_HANDLER.findall((_PACKAGE / filename).read_text())))


def test_the_extractor_finds_a_symbol_known_to_be_defined():
    """The pattern reads real handler names out of the C++ source."""
    assert "ExactCellHullF64" in _defined_in("certified_affine_ffi_cpu.cc")


def test_the_inventory_is_not_empty():
    """A silently emptied inventory would make every comparison below vacuous."""
    assert len(EXACT_AFFINE_HANDLER_SYMBOLS) > 0


@pytest.mark.parametrize(
    "source", ["certified_affine_ffi_cpu.cc", "certified_affine_ffi_cuda.cu"]
)
def test_each_translation_unit_defines_exactly_the_inventory(source):
    """The compiled sources define the handlers the inventory names, and no others."""
    assert _defined_in(source) == tuple(sorted(EXACT_AFFINE_HANDLER_SYMBOLS))


def test_xla_registration_uses_the_inventory():
    """The names registered with XLA are the inventory itself."""
    assert ffi._TARGETS == EXACT_AFFINE_HANDLER_SYMBOLS


def test_the_windows_export_list_uses_the_inventory():
    """The `.def` file handed to `link.exe` exports exactly those names."""
    assert hatch_build.EXACT_AFFINE_HANDLER_SYMBOLS == EXACT_AFFINE_HANDLER_SYMBOLS


def test_the_module_definition_exports_every_inventory_name():
    """Each name reaches the module-definition file the Windows linker reads."""
    exported = [
        line.strip()
        for line in hatch_build.windows_module_definition().splitlines()[2:]
    ]

    assert exported == list(EXACT_AFFINE_HANDLER_SYMBOLS)
