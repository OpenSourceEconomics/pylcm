"""Replication of Mahler & Yum (2024, Econometrica).

Lifecycle model from "Lifestyle Behaviors and Wealth-Health Gaps in Germany".
Two regimes (alive/dead), 8 states, 3 actions, stochastic transitions.
"""

from lcm_examples.mahler_yum_2024._model import (
    ALIVE_REGIME,
    DEAD_REGIME,
    MAHLER_YUM_MODEL,
    START_PARAMS,
    Education,
    Effort,
    Health,
    HealthType,
    LaborSupply,
    ProductivityShock,
    ProductivityType,
    RegimeId,
    ages,
    create_inputs,
)

__all__ = [
    "ALIVE_REGIME",
    "DEAD_REGIME",
    "MAHLER_YUM_MODEL",
    "START_PARAMS",
    "Education",
    "Effort",
    "Health",
    "HealthType",
    "LaborSupply",
    "ProductivityShock",
    "ProductivityType",
    "RegimeId",
    "ages",
    "create_inputs",
]
