"""Supervise a diagnostic child and preserve partial output on deadline or RSS cap."""

import argparse
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import psutil
from observer import Observer


def main() -> int:
    """Supervise one bounded child and preserve the cancellation exit status."""
    args, command = _parse_arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    observer = Observer(args.output / "supervisor.jsonl")
    observer.emit(
        "supervisor_start",
        command=command,
        seconds=args.seconds,
        rss_gib=args.rss_gib,
        note="RSS sums the child tree and may count shared pages more than once.",
    )
    termination_requested = False

    # keyword-only-exempt: library-callback=signal.signal
    def request_termination(_signum: int, _frame: object) -> None:
        """Record TERM without interrupting assignment of the newly launched child."""
        nonlocal termination_requested
        termination_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_termination)
    try:
        with (
            (args.output / "stdout.log").open("w") as stdout,
            (args.output / "stderr.log").open("w") as stderr,
        ):
            child = None
            started = time.monotonic()
            peak_rss = 0
            stop_reason = None
            try:
                child = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, start_new_session=True
                )
                observer.emit("child_started", pid=child.pid)
                while child.poll() is None:
                    rss = _child_tree_rss(child.pid)
                    peak_rss = max(peak_rss, rss)
                    if termination_requested:
                        stop_reason = "sigterm"
                    elif time.monotonic() - started >= args.seconds:
                        stop_reason = "deadline"
                    elif rss > args.rss_gib * 1024**3:
                        stop_reason = "rss_cap"
                    if stop_reason:
                        observer.emit(
                            "child_limit",
                            reason=stop_reason,
                            elapsed=time.monotonic() - started,
                            peak_rss=peak_rss,
                        )
                        _terminate(child)
                        break
                    time.sleep(0.5)
            except BaseException:
                if child is not None:
                    _terminate(child)
                raise
            code = child.wait()
        observer.emit(
            "child_finished",
            exit_code=code,
            reason=stop_reason,
            elapsed=time.monotonic() - started,
            peak_rss=peak_rss,
        )
        if termination_requested:
            return 143
        return (
            124
            if stop_reason == "deadline"
            else 137
            if stop_reason == "rss_cap"
            else code
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def _parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    """Parse and validate the bounded child command and its resource limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--rss-gib", type=int, default=32)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.seconds <= 1800 or not 1 <= args.rss_gib <= 64:
        parser.error(
            "Specify a command, a 1..1800 second deadline and a 1..64 GiB RSS cap."
        )
    return args, command


def _child_tree_rss(pid: int) -> int:
    """Return RSS of the running child tree, or zero after a process exits."""
    try:
        parent = psutil.Process(pid)
        rss = sum(
            process.memory_info().rss
            for process in [parent, *parent.children(recursive=True)]
            if process.is_running()
        )
    except psutil.NoSuchProcess:
        return 0
    else:
        return rss


def _terminate(child: subprocess.Popen) -> None:
    """Give the owned process group five seconds to exit, then kill survivors."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        child.poll()
        if not _group_has_live_members(child.pid):
            child.wait()
            return
        time.sleep(0.05)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def _group_has_live_members(group_id: int) -> bool:
    """Observe whether the owned group still contains an executing process."""
    for process in psutil.process_iter(["pid", "status"]):
        try:
            if (
                process.info["status"] != psutil.STATUS_ZOMBIE
                and os.getpgid(process.pid) == group_id
            ):
                return True
        except ProcessLookupError, PermissionError:
            continue
    return False


if __name__ == "__main__":
    raise SystemExit(main())
