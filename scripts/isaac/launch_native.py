"""Windows process boundary for one native Kit command; no shell or background service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil


def save(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def terminate_tree(process: psutil.Process) -> None:
    for child in [*reversed(process.children(recursive=True)), process]:
        with suppress(psutil.NoSuchProcess):
            child.terminate()


def launch(request_file: Path) -> int:
    request = json.loads(request_file.read_text(encoding="utf-8-sig"))
    state_file = Path(request["state"])
    timeout = request["timeout_seconds"]
    attempt = 1
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["status"] == "finished" and state["returncode"] == 0:
            return state["returncode"]
        try:
            process = psutil.Process(state["pid"])
            if abs(process.create_time() - state["created_at"]) < 0.01:
                remaining = max(1, timeout - (time.time() - state["started_at"]))
                code = process.wait(timeout=remaining)
                save(state_file, {**state, "status": "finished", "returncode": code})
                return code
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            terminate_tree(process)
            save(state_file, {**state, "status": "finished", "returncode": 124})
            return 124  # An existing active command is never launched a second time.
        save(
            state_file.with_name(f"{state_file.stem}-attempt-{state.get('attempt', 1)}.json"), state
        )
        attempt = state.get("attempt", 1) + 1
    env = os.environ | request.get("environment", {})
    log_file = Path(request["log"])
    if attempt > 1:
        log_file = log_file.with_name(f"{log_file.stem}-attempt-{attempt}.log")
    with log_file.open("wb") as log:
        child = subprocess.Popen(
            request["command"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        native = psutil.Process(child.pid)
        state = {
            "status": "running",
            "pid": child.pid,
            "created_at": native.create_time(),
            "started_at": time.time(),
            "request": str(request_file),
            "attempt": attempt,
            "log_name": log_file.name,
        }
        save(state_file, state)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_tree(native)
            child.wait()
            code = 124
        save(state_file, {**state, "status": "finished", "returncode": code})
        return code


if __name__ == "__main__":
    sys.exit(launch(Path(sys.argv[1])))
