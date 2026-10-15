"""Canonical evaluator pool for model-bound DC-EGM queries."""

from dataclasses import dataclass

from _lcm.continuation import EGMQuerySpec
from _lcm.typing import EconFunction, EconFunctionsMapping
from lcm.typing import Float1D


@dataclass(frozen=True, kw_only=True)
class DCEGMQueryModel:
    """Canonical DC evaluator pool prepared before numerical kernel builders."""

    authority: EGMQuerySpec
    """Independent descriptor shared by the phase and consuming adapters."""
    functions: EconFunctionsMapping
    """Processed canonical functions with unresolved period variants."""
    koopmans_aggregator: EconFunction
    """Processed canonical aggregator reused by the solution phase."""
    offsets: Float1D
    """One supplied saving-offset vector, without cell or period expansion."""
