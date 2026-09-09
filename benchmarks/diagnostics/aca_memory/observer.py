"""Durable scalar receipts around one diagnostic dispatch."""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path


class Observer:
    """Record dispatch provenance without retaining numerical payloads."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, event: str, **metadata: object) -> None:
        """Append one durable receipt."""
        record = {"event": event, "time_ns": time.time_ns(), **metadata}
        encoded = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def run(
        self,
        *,
        tag: dict,
        metadata: Callable,
        execute: Callable,
        wait: Callable,
        census: Callable,
        completion_confirmed: bool = True,
    ) -> object:
        """Execute one observed call and return its original result."""
        observed_at = time.perf_counter()
        metadata_snapshot = metadata()
        census_snapshot = census()
        self.emit(
            "dispatch_begin",
            tag=tag,
            metadata=metadata_snapshot,
            census=census_snapshot,
            preparation_seconds=time.perf_counter() - observed_at,
        )
        stage = "execute"
        started = time.perf_counter()
        try:
            result = execute()
            dispatch_seconds = time.perf_counter() - started
            self.emit("dispatch_returned", tag=tag, dispatch_seconds=dispatch_seconds)
            stage = "wait"
            waiting = time.perf_counter()
            wait(result)
            wait_seconds = time.perf_counter() - waiting
        except Exception as error:
            # Receipt failures must not replace the original dispatch exception.
            try:
                self.emit(
                    "dispatch_error",
                    tag=tag,
                    stage=stage,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            except Exception as receipt_error:
                error.add_note(f"Error receipt could not be written: {receipt_error}")
            raise
        self.emit(
            "dispatch_complete",
            tag=tag,
            dispatch_seconds=dispatch_seconds,
            wait_seconds=wait_seconds,
            completion_confirmed=completion_confirmed,
            diagnostic_call_seconds=dispatch_seconds + wait_seconds,
            census=census(),
        )
        return result
