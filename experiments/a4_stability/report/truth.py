"""Recover which training actually produced each W&B run's log (containers were reused and
wandb.init returned the previous run after a crash), from the stdout logs."""
import json, os, re, wandb
api = wandb.Api()
out = {}
for r in api.runs('whitedeer-stanford-university/assignments'):
    if not r.name.startswith('a4-'):
        continue
    d = f'/tmp/wb/{r.id}'
    try:
        r.file('output.log').download(d, replace=True)
        log = open(f'{d}/output.log', errors='replace').read()
    except Exception as e:
        log = ''
    # split by "Using custom model builder" segments; each segment belongs to one training
    segments = re.split(r'(?=Using custom model builder)', log)
    for seg in segments:
        m = re.search(r'Saving async checkpoint for step \d+ to /root/data/ckpts/([^/]+)/', seg)
        name = m.group(1) if m else None
        finals = re.findall(r'Step 9375/9375.*?val Loss: ([0-9.]+)', seg)
        saved = 'Saving model to /root/data/ckpts/' in seg
        crash = 'Traceback' in seg or 'RecompileLimitExceeded' in seg
        nan = 'Loss: nan' in seg.lower()
        if name or finals:
            out.setdefault(name, []).append(dict(wandb_run=r.name, wandb_id=r.id, final_val=float(finals[-1]) if finals else None, saved=saved, crash=crash, nan=nan))
json.dump(out, open(os.path.expanduser('~/a4work/my_runs_truth.json'), 'w'), indent=1)
for k, v in sorted(out.items(), key=lambda kv: str(kv[0])):
    print(k, v)
