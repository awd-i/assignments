"""Launch A2 training configs one Modal app per run, keeping at most MAX_GPUS A2 runs active (course limit).

Counts every running run in the W&B project (including runs launched by other drivers) plus this driver's
launched-but-not-yet-visible runs. Retries a crashed run once.
usage: python -m experiments.a2.awd.queue_driver module:function [module:function ...]
"""

import importlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import wandb

from experiments.a2.modal_launcher import launch_training_jobs
from train import training_run_name

PROJECT = "whitedeer-stanford-university/assignments"
MAX_GPUS = 2
EXTRA_GPUS = Path(__file__).resolve().parent / ".extra_gpus"  # GPUs held by non-W&B jobs (P4.1 container)
DONE = ("finished", "crashed", "failed", "killed")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def api_call(fn, limit=180):
    """W&B query with a hard wall-clock limit (a sleeping laptop can leave sockets hung forever)."""
    signal.signal(signal.SIGALRM, _alarm)
    while True:
        signal.alarm(limit)
        try:
            return fn(wandb.Api(timeout=60))
        except Exception as e:  # network blips / hangs: wait and retry
            print("wandb error:", repr(e)[:150], flush=True)
            time.sleep(60)
        finally:
            signal.alarm(0)


def states(names):
    """State of each named run. Avoids `display_name $in` filters, which hang on long lists."""
    names = set(names)
    runs = api_call(lambda api: list(api.runs(PROJECT, filters={"created_at": {"$gt": "2026-09-30T00:00:00"}},
                                              per_page=500)))
    out = {}
    for r in runs:  # a retried name can have several runs; prefer an active/finished one
        if r.name in names and out.get(r.name) not in ("finished", "running"):
            out[r.name] = r.state
    return out


def n_running_elsewhere(mine):
    runs = api_call(lambda api: list(api.runs(PROJECT, filters={"state": "running"})))
    return sum(r.name not in mine for r in runs)


LAUNCH_SNIPPET = """
import importlib, sys
from experiments.a2.modal_launcher import launch_training_jobs
from train import training_run_name
mod, fn, name = sys.argv[1:4]
cs = [c for c in getattr(importlib.import_module(mod), fn)() if training_run_name(c) == name]
launch_training_jobs(cs, max_parallel_runs=1)
"""


def launch(spec, name):
    """Launch one config in a fresh subprocess with a timeout (stale Modal connections can hang forever)."""
    mod, fn = spec.split(":")
    while True:
        try:
            r = subprocess.run([sys.executable, "-c", LAUNCH_SNIPPET, mod, fn, name], capture_output=True, text=True,
                               timeout=900)
            if r.returncode == 0:
                return
            print("launch error:", r.stderr[-300:], flush=True)
        except subprocess.TimeoutExpired:
            print("launch error: timeout", flush=True)
        time.sleep(60)


def main(specs):
    by_name, spec_of = {}, {}
    for spec in specs:
        mod, fn = spec.split(":")
        for c in getattr(importlib.import_module(mod), fn)():
            by_name[training_run_name(c)] = c
            spec_of[training_run_name(c)] = spec
    st = states(by_name)
    todo = [n for n in by_name if st.get(n) not in ("finished", "running")]
    launched = {n: time.time() for n in by_name if st.get(n) == "running"}  # already in flight: just track
    attempts = {n: 1 for n in launched}
    print(f"{len(by_name)} configs, {len(launched)} running, {len(todo)} to run", flush=True)
    hb = os.environ.get("HEARTBEAT")
    while todo or launched:
        if hb:
            Path(hb).touch()
        st = states(launched) if launched else {}
        for n in list(launched):
            s = st.get(n, "pending")
            if s in DONE:
                print(time.ctime(), "DONE", n, s, flush=True)
                del launched[n]
                if s != "finished" and attempts[n] < 2:
                    todo.insert(0, n)
        extra = int(EXTRA_GPUS.read_text()) if EXTRA_GPUS.exists() else 0
        busy = len(launched) + n_running_elsewhere(set(launched)) + extra
        while todo and busy < MAX_GPUS:
            n = todo.pop(0)
            attempts[n] = attempts.get(n, 0) + 1
            print(time.ctime(), "LAUNCH", n, f"(attempt {attempts[n]})", flush=True)
            launch(spec_of[n], n)
            launched[n] = time.time()
            busy += 1
        time.sleep(120)
    print("== queue done", time.ctime(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
