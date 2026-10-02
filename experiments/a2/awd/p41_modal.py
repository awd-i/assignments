"""A2 Problem 4.1: run many five-step stress configurations sequentially in ONE detached GPU container.

Each result is written to the private volume at /a2-stress/awd/<name>.json (existing files are skipped).
usage: python -m experiments.a2.awd.p41_modal width|depth|<extra-jobs.json>
"""

import json
import sys

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS, MODAL_ENVIRONMENT, MODAL_SHARED_DATASETS_DIR,
                         timestamped_modal_app_name)

OUT_DIR = "/root/data/a2-stress/awd"
WIDTH_LRS = [1e-3 * 2**k for k in range(-4, 4)]   # 6.25e-5 .. 8e-3
DEPTH_LRS = [1e-3 * 2**k for k in range(-2, 6)]   # 2.5e-4 .. 3.2e-2


def job_name(j):
    return f"{j['policy']}-w{j['width']}-d{j['depth']}-{j['precision']}-lr{j['lr']:.3g}"


def width_jobs(lrs=WIDTH_LRS):
    return [dict(policy=p, width=w, depth=2, precision="fp32", lr=lr, alignment=True)
            for w in (640, 2560, 5120) for p in ("kaiming", "mup") for lr in lrs]


def depth_jobs(lrs=DEPTH_LRS):
    jobs = [dict(policy="mup", width=64, depth=2, precision="mp", lr=lr, alignment=False) for lr in lrs]
    return jobs + [dict(policy=p, width=64, depth=d, precision="mp", lr=lr, alignment=False)
                   for d in (100, 1000) for p in ("mup", "depth-mup", "completep") for lr in lrs]


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", retries=0, max_containers=1, timeout=6 * 3600,
              memory=65536)
def run_jobs(jobs):
    import gc
    import os
    import time
    from pathlib import Path

    import torch

    from experiments.a2.awd.p41_policies import make
    from experiments.a2.stress import StressConfig, load_tokens, run

    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    base = StressConfig()
    train = load_tokens(str(data / "train"), base.steps * base.batch, base.context)
    val = load_tokens(str(data / "val"), base.batch, base.context)
    os.makedirs(OUT_DIR, exist_ok=True)
    for j in jobs:
        out = Path(OUT_DIR) / (job_name(j) + ".json")
        if out.exists():
            continue
        t0 = time.time()
        c = StressConfig(width=j["width"], depth=j["depth"], precision=j["precision"],
                         microbatch=j.get("microbatch", 8))
        init, groups = make(j["policy"])
        try:
            r = run(c, train, val, base_lr=j["lr"], initialize_fn=init, groups_fn=groups, device="cuda",
                    alignment=j["alignment"])
            r["job"] = j
        except Exception as e:  # record divergence/OOM/other failures and continue
            r = dict(job=j, error=repr(e)[:300])
        out.write_text(json.dumps(r, allow_nan=False))
        user_volume.commit()
        last = r.get("history", [{}])[-1].get("val_loss")
        print(f"{job_name(j)}: final val {last} ({time.time() - t0:.0f}s)", flush=True)
        del r
        gc.collect()
        torch.cuda.empty_cache()


def main(which):
    jobs = {"width": width_jobs, "depth": depth_jobs}.get(which)
    jobs = jobs() if jobs else json.load(open(which))
    import modal
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name("a2-p41"), detach=True, environment_name=MODAL_ENVIRONMENT):
            call = run_jobs.spawn(jobs)
            print("spawned", call.object_id, len(jobs), "jobs", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
