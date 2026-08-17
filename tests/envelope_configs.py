"""Typed upper-envelope configurations used by parametrized model fixtures."""

from typing import Literal

from lcm.solvers import (
    EnvelopeConfig,
    ExactEnvelope,
    FUESEnvelope,
    LTMEnvelope,
    MSSEnvelope,
    RFCEnvelope,
)

type EnvelopeName = Literal["exact", "fues", "rfc", "ltm", "mss"]


def envelope_config(value: EnvelopeName | EnvelopeConfig) -> EnvelopeConfig:
    """Return the typed configuration denoted by a compact fixture name."""
    if not isinstance(value, str):
        return value
    constructors = {
        "exact": ExactEnvelope,
        "fues": FUESEnvelope,
        "rfc": RFCEnvelope,
        "ltm": LTMEnvelope,
        "mss": MSSEnvelope,
    }
    return constructors[value]()
