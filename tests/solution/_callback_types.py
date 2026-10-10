"""Keyword arguments received by solution publication observers."""

from collections.abc import Mapping
from typing import NotRequired, TypedDict

from lcm.solver_api import ArtifactAuthority, ArtifactKey, KernelOutput
from lcm.typing import RegimeName


class ConsumeOutputKwargs(TypedDict):
    output: KernelOutput
    continuation_key: ArtifactKey | None
    regime_name: RegimeName
    period: int
    artifact_authorities: NotRequired[Mapping[ArtifactKey, ArtifactAuthority]]
