"""Adapt the independently supplied generated specification to the real codec."""
from .structural_check import decode, validate
from .structural_transport import capture_structure, reseal


def capture(*, source, source_anchor, before_final_validation=None):
    # This adapter accepts the external harness's explicit logical-array class.
    return capture_structure(source=source, source_anchor=source_anchor,
                             before_final_validation=before_final_validation,
                             logical_array_type=type(source['array']))
