#!/usr/bin/env python3
"""Persistent local-only supervisor for the Round 2 experiment queue.

The supervisor deliberately knows nothing about local_eval external_labels or Kaggle.  It
only adopts an already-running official-only process, waits for its immutable
metrics artifact, and then starts the next pre-created protocol-only child.
It uses a lock and atomic state writes so a service restart cannot create two
heavy jobs for the same experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = ROOT / "Polymer Prediction Challenge Round 2/research/watchdog-queue.json"
WATCHDOG_DIR = ROOT / "Polymer Prediction Challenge Round 2/experiments/CLEAN_OFFICIAL_ONLY/_watchdog"
STATE_PATH = WATCHDOG_DIR / "state.json"
EVENTS_PATH = WATCHDOG_DIR / "events.jsonl"
LOCK_PATH = WATCHDOG_DIR / "watchdog.lock"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def last_event_line_sha256() -> str | None:
    try:
        lines = EVENTS_PATH.read_bytes().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        if line.strip():
            return hashlib.sha256(line).hexdigest()
    return None


def append_event(event: dict[str, Any]) -> None:
    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event.setdefault("schema_version", "ppp.round2.watchdog-event.v1")
    event["previous_event_line_sha256"] = last_event_line_sha256()
    payload = json.dumps(event, sort_keys=True, allow_nan=False).encode("utf-8")
    event["event_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def find_existing_pid(run_id: str, script: str, own_pid: int) -> int | None:
    script_name = Path(script).name
    proc = Path("/proc")
    for candidate in proc.iterdir():
        if not candidate.name.isdigit() or int(candidate.name) == own_pid:
            continue
        command = process_cmdline(int(candidate.name))
        if script_name in command and run_id in command:
            return int(candidate.name)
    return None


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def metric_summary(run_dir: Path) -> dict[str, Any] | None:
    metrics = load_json(run_dir / "metrics.json", None)
    if not isinstance(metrics, dict):
        return None
    keep = ("schema_version", "experiment_id", "decision", "mean_parent_r2",
            "mean_candidate_r2", "mean_gain", "banked_targets",
            "complete_output_rows", "full_candidate_gate_pass")
    return {key: metrics[key] for key in keep if key in metrics}


def terminal_audit(python_bin: Path, run_dir: Path) -> dict[str, Any]:
    auditor = ROOT / "Polymer Prediction Challenge Round 2/tools/round2_terminal_artifact_audit.py"
    round2_root = ROOT / "Polymer Prediction Challenge Round 2"
    if not auditor.exists():
        return {"pass": False, "error": "missing_terminal_artifact_auditor", "tool": str(auditor)}
    command = [
        str(python_bin),
        str(auditor),
        "--root",
        str(round2_root),
        "--data-dir",
        "ppp-round-2",
        "--run-dir",
        str(run_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:  # pragma: no cover - defensive watchdog guard.
        return {"pass": False, "error": f"terminal_audit_exception:{type(exc).__name__}", "detail": str(exc), "command": command}
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "pass": False,
            "error": "terminal_audit_invalid_json",
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "command": command,
        }
    report["audit_exit_code"] = completed.returncode
    report["audit_stderr_tail"] = completed.stderr[-1000:]
    return report


def run_command(root: Path, python_bin: Path, entry: dict[str, Any], log_path: Path) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_bin),
        str(root / entry["script"]),
        "--root", str(root),
        "--data-dir", "Polymer Prediction Challenge Round 2/ppp-round-2",
        "--run-dir", str(root / entry["run_dir"]),
        "--canonical-run", str(root / entry["canonical_run"]),
    ]
    environment = os.environ.copy()
    # Do not override BLAS/OpenMP settings here.  The C050 replay gate is
    # exact to 1e-12, and changing thread counts can alter reduction order
    # enough to invalidate an otherwise identical parent reconstruction.
    environment["PYTHONUNBUFFERED"] = "1"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    handle.write(f"[{now()}] watchdog launch: {json.dumps(command)}\n")
    handle.flush()
    child = subprocess.Popen(
        command,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    # The child inherits the open descriptor; the watchdog does not need it.
    handle.close()
    return child


def protocol_only(run_dir: Path) -> bool:
    try:
        return {item.name for item in run_dir.iterdir()} == {"protocol.json"}
    except FileNotFoundError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--poll-seconds", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    queue_path = Path(args.queue).resolve()
    queue = load_json(queue_path, {})
    if not isinstance(queue, dict) or not isinstance(queue.get("entries"), list):
        raise SystemExit(f"invalid watchdog queue: {queue_path}")
    queue_sha256 = sha256_file(queue_path)
    poll_seconds = max(5, int(args.poll_seconds or queue.get("poll_seconds", 30)))
    recovery_wait = max(300, int(queue.get("recovery_wait_seconds", 7200)))
    python_bin = ROOT / "Polymer Prediction Challenge Round 2/.venv/bin/python"
    if not python_bin.exists():
        raise SystemExit(f"missing experiment interpreter: {python_bin}")

    WATCHDOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A second service instance is harmless; the first one owns the queue.
            return 0

        state = load_json(STATE_PATH, {
            "schema_version": "ppp.round2.local-watchdog-state.v1",
            "started_at": now(),
            "history": [],
        })
        state["watchdog_pid"] = os.getpid()
        state["service_started_at"] = now()
        state["queue_path"] = str(queue_path)
        state["queue_sha256"] = queue_sha256
        atomic_json(STATE_PATH, state)
        append_event({"at": now(), "event": "watchdog_started", "pid": os.getpid(), "queue_path": str(queue_path), "queue_sha256": queue_sha256})

        index = 0
        entries = queue["entries"]
        while index < len(entries):
            entry = entries[index]
            run_id = str(entry["run_id"])
            run_dir = ROOT / str(entry["run_dir"])
            metrics = metric_summary(run_dir)
            if metrics is not None:
                audit = terminal_audit(python_bin, run_dir)
                status = "completed" if audit.get("pass") is True else "failed_terminal_audit"
                state.setdefault("history", []).append({"run_id": run_id, "status": status, "metrics": metrics, "terminal_audit": audit, "at": now()})
                atomic_json(STATE_PATH, state)
                append_event({"at": now(), "event": "existing_metrics", "run_id": run_id, "status": status, "metrics": metrics, "terminal_audit": audit})
                index += 1
                continue

            previous_terminal = next(
                (
                    item for item in reversed(state.get("history", []))
                    if item.get("run_id") == run_id
                    and item.get("status") in {"failed", "failed_protocol_state", "failed_stale_launch", "adopted_failed", "failed_terminal_audit"}
                ),
                None,
            )
            if previous_terminal is not None:
                state.setdefault("history", []).append({"run_id": run_id, "status": "skipped_after_failure", "at": now()})
                state["last_heartbeat"] = now()
                atomic_json(STATE_PATH, state)
                append_event({"at": now(), "event": "skip_failed_child", "run_id": run_id})
                index += 1
                continue

            requirements = entry.get("requires_any_terminal", [])
            if requirements:
                terminal = False
                for required_id in requirements:
                    for prior in state.get("history", []):
                        if prior.get("run_id") == required_id and prior.get("status") in {"completed", "failed", "adopted_failed", "failed_terminal_audit"}:
                            terminal = True
                    required_dir = next((ROOT / str(candidate["run_dir"]) for candidate in entries if candidate.get("run_id") == required_id), None)
                    if required_dir is not None and metric_summary(required_dir) is not None:
                        terminal = True
                if not terminal:
                    state.update({"active_run": None, "queue_index": index, "last_heartbeat": now(), "status": "waiting_for_dependency"})
                    atomic_json(STATE_PATH, state)
                    append_event({"at": now(), "event": "waiting_for_dependency", "run_id": run_id, "requires_any_terminal": requirements})
                    if args.once:
                        return 0
                    time.sleep(poll_seconds)
                    continue

            # A completed primary skips its recovery child.  The fallback is
            # deliberately a separate pre-created experiment ID.
            fallback_for = entry.get("fallback_for")
            if fallback_for:
                primary_dir = next((ROOT / str(candidate["run_dir"]) for candidate in entries if candidate.get("run_id") == fallback_for), None)
                primary_metrics = metric_summary(primary_dir) if primary_dir is not None else None
                if primary_metrics is not None:
                    primary_audit = terminal_audit(python_bin, primary_dir)
                    if primary_audit.get("pass") is True:
                        state.setdefault("history", []).append({"run_id": run_id, "status": "skipped_primary_completed", "at": now()})
                        atomic_json(STATE_PATH, state)
                        append_event({"at": now(), "event": "skip_fallback", "run_id": run_id, "fallback_for": fallback_for, "primary_terminal_audit": primary_audit})
                        index += 1
                        continue
                    append_event({"at": now(), "event": "primary_terminal_audit_failed_fallback_allowed", "run_id": run_id, "fallback_for": fallback_for, "primary_terminal_audit": primary_audit})

            existing_pid = find_existing_pid(run_id, str(entry["script"]), os.getpid())
            if existing_pid is not None:
                state.update({"active_run": run_id, "active_pid": existing_pid, "adopted": True, "queue_index": index, "status": "adopted_running", "last_heartbeat": now()})
                atomic_json(STATE_PATH, state)
                append_event({"at": now(), "event": "adopted_existing_process", "run_id": run_id, "pid": existing_pid})
                while process_alive(existing_pid):
                    state["last_heartbeat"] = now()
                    state["process_alive"] = True
                    state["metrics_available"] = metric_summary(run_dir) is not None
                    atomic_json(STATE_PATH, state)
                    if args.once:
                        return 0
                    time.sleep(poll_seconds)
                summary = metric_summary(run_dir)
                state["process_alive"] = False
                state["metrics_available"] = summary is not None
                if summary is None:
                    state.setdefault("history", []).append({"run_id": run_id, "status": "adopted_failed", "at": now()})
                    append_event({"at": now(), "event": "adopted_process_ended_without_metrics", "run_id": run_id, "pid": existing_pid})
                else:
                    audit = terminal_audit(python_bin, run_dir)
                    status = "completed" if audit.get("pass") is True else "failed_terminal_audit"
                    state.setdefault("history", []).append({"run_id": run_id, "status": status, "metrics": summary, "terminal_audit": audit, "at": now()})
                    append_event({"at": now(), "event": "adopted_process_completed", "run_id": run_id, "status": status, "metrics": summary, "terminal_audit": audit})
                state["active_run"] = None
                atomic_json(STATE_PATH, state)
                index += 1
                continue

            if not protocol_only(run_dir):
                state.setdefault("history", []).append({"run_id": run_id, "status": "failed_protocol_state", "at": now()})
                append_event({"at": now(), "event": "protocol_state_invalid", "run_id": run_id, "run_dir": str(run_dir)})
                atomic_json(STATE_PATH, state)
                index += 1
                continue

            # If a prior watchdog instance recorded a recent launch, allow its
            # process a generous recovery window before creating a new child.
            prior_launch = next((item for item in reversed(state.get("history", [])) if item.get("run_id") == run_id and item.get("status") == "launched"), None)
            if prior_launch:
                try:
                    launch_age = time.time() - float(prior_launch["epoch"])
                except (KeyError, TypeError, ValueError):
                    launch_age = recovery_wait
                if launch_age < recovery_wait:
                    state.update({"active_run": run_id, "active_pid": None, "adopted": False, "queue_index": index, "status": "waiting_for_recent_launch", "last_heartbeat": now()})
                    atomic_json(STATE_PATH, state)
                    if args.once:
                        return 0
                    time.sleep(poll_seconds)
                    continue
                state.setdefault("history", []).append({"run_id": run_id, "status": "failed_stale_launch", "at": now()})
                state["last_heartbeat"] = now()
                atomic_json(STATE_PATH, state)
                append_event({"at": now(), "event": "stale_launch_recovery", "run_id": run_id, "recovery_wait_seconds": recovery_wait})
                index += 1
                continue

            log_path = WATCHDOG_DIR / "logs" / f"{run_id}.log"
            child = run_command(ROOT, python_bin, entry, log_path)
            state.setdefault("history", []).append({"run_id": run_id, "status": "launched", "pid": child.pid, "epoch": time.time(), "at": now()})
            state.update({
                "active_run": run_id,
                "active_pid": child.pid,
                "adopted": False,
                "queue_index": index,
                "status": "running",
                "last_heartbeat": now(),
                "process_alive": True,
                "metrics_available": metric_summary(run_dir) is not None,
            })
            atomic_json(STATE_PATH, state)
            append_event({"at": now(), "event": "launched", "run_id": run_id, "pid": child.pid, "log": str(log_path)})
            while child.poll() is None:
                state["last_heartbeat"] = now()
                state["process_alive"] = True
                state["metrics_available"] = metric_summary(run_dir) is not None
                atomic_json(STATE_PATH, state)
                if args.once:
                    return 0
                time.sleep(poll_seconds)
            summary = metric_summary(run_dir)
            audit = terminal_audit(python_bin, run_dir) if summary is not None else None
            if child.returncode == 0 and summary is not None and audit is not None and audit.get("pass") is True:
                status = "completed"
            elif summary is not None and audit is not None and audit.get("pass") is not True:
                status = "failed_terminal_audit"
            else:
                status = "failed"
            state.setdefault("history", []).append({"run_id": run_id, "status": status, "exit_code": child.returncode, "metrics": summary, "terminal_audit": audit, "at": now()})
            state["active_run"] = None
            state["active_pid"] = None
            state["status"] = status
            atomic_json(STATE_PATH, state)
            append_event({"at": now(), "event": "child_finished", "run_id": run_id, "exit_code": child.returncode, "status": status, "metrics": summary, "terminal_audit": audit})
            index += 1
            if args.once:
                return 0

        state.update({"active_run": None, "queue_index": len(entries), "status": "queue_idle", "last_heartbeat": now()})
        atomic_json(STATE_PATH, state)
        append_event({"at": now(), "event": "queue_idle", "entries": len(entries)})
        while not args.once:
            state["last_heartbeat"] = now()
            atomic_json(STATE_PATH, state)
            time.sleep(poll_seconds)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
