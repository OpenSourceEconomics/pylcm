"""Fields consumed by the committed GitHub workflow contract tests."""

from typing import TypedDict

WorkflowStepInputs = TypedDict(
    "WorkflowStepInputs",
    {
        "name": str,
        "path": str,
        "persist-credentials": bool,
    },
    total=False,
)


# GitHub's YAML spelling is a Python keyword.
WorkflowStep = TypedDict(
    "WorkflowStep",
    {
        "name": str,
        "id": str,
        "run": str,
        "uses": str,
        "env": dict[str, str],
        "with": WorkflowStepInputs,
        "if": str,
    },
    total=False,
)


class MatrixEntry(TypedDict, total=False):
    os: str
    precision: int
    leg: str
    shard: int
    shards: int
    pytest_extra: str


class WorkflowMatrix(TypedDict, total=False):
    include: list[MatrixEntry]


class WorkflowStrategy(TypedDict):
    matrix: WorkflowMatrix


class WorkflowJob(TypedDict, total=False):
    steps: list[WorkflowStep]
    needs: str | list[str]
    strategy: WorkflowStrategy
    env: dict[str, str]


class Workflow(TypedDict):
    jobs: dict[str, WorkflowJob]
