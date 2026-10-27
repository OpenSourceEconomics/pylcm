"""Model-authenticated inputs of the native current-IID reconstruction families."""

from collections.abc import Mapping

from _lcm.egm.current_iid_case_u_replay import (
    CASE_U_REPLAY_PROGRAM,
    CaseUReplayChildren,
    CaseUReplayInput,
)
from _lcm.egm.current_iid_replay import CASE_R_REPLAY_PROGRAM, CaseRReplayInput
from _lcm.egm.current_iid_resources_artifact import (
    ResourcesRowLeaf,
    resources_row_payload,
)
from lcm.typing import RegimeName

type CurrentIIDReplayInput = CaseRReplayInput | CaseUReplayInput
type CurrentIIDReplayInputs = Mapping[int, Mapping[RegimeName, CurrentIIDReplayInput]]


def native_replay_program(*, replay: CurrentIIDReplayInput) -> str:
    """Select the actual native family, preserving its own artifact locator."""
    if isinstance(replay, CaseUReplayInput):
        return CASE_U_REPLAY_PROGRAM
    return CASE_R_REPLAY_PROGRAM


def native_replay_payload(
    *, replay: CurrentIIDReplayInput
) -> CaseUReplayChildren | tuple[ResourcesRowLeaf, ...]:
    """Read the real payload; a U future bundle is never a resources row."""
    return (
        replay.payload
        if isinstance(replay, CaseUReplayInput)
        else resources_row_payload(row=replay.row)
    )
