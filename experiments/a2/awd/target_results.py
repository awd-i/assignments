"""Fetch finished A2 target runs from W&B (cached to plots/target_runs.json) for analysis and plots."""

import json
from pathlib import Path

from experiments.a2.awd.queue_driver import PROJECT, api_call

CACHE = Path(__file__).resolve().parent / "plots" / "target_runs.json"
TAG_PREFIXES = ("a2-p1d-hyperball-D1229m", "a2-p1-D4915m", "a2-p2c", "a2-p32", "a2-p32c", "a2-p42", "a2-p5")


def fetch():
    runs = api_call(lambda api: list(api.runs(PROJECT, filters={"state": "finished"})))
    out = []
    for r in runs:
        if not any(t.startswith(TAG_PREFIXES) for t in r.tags):
            continue
        c = r.config
        out.append(dict(name=r.name, tags=r.tags, url=r.url, lr=c.get("learning_rate"), wd=c.get("weight_decay"),
                        batch=c.get("batch_size"), beta1=c.get("beta1"), tokens=c.get("num_train_sequences", 0) * 1024,
                        model=(c.get("model_config") or {}).get("name"), kwargs=c.get("model_builder_kwargs"),
                        val_loss=r.summary.get("val_loss"), runtime_min=r.summary.get("_runtime", 0) / 60))
    CACHE.write_text(json.dumps(out, indent=1))
    return out


def load(tag=None):
    rows = json.loads(CACHE.read_text())
    return [r for r in rows if tag is None or any(t.startswith(tag) for t in r["tags"])]


if __name__ == "__main__":
    rows = fetch()
    for r in sorted(rows, key=lambda r: r["name"]):
        print(f"{r['name']:70s} {r['val_loss']:.4f}  {r['runtime_min']:.0f} min")
