import wandb, sys
api = wandb.Api()
rows = []
for r in api.runs('whitedeer-stanford-university/assignments', order='-created_at', per_page=100):
    if r.name.startswith('a4-'):
        s = r.summary
        rows.append((r.name, r.state, s.get('step'), s.get('train_loss'), s.get('val_loss'), s.get('_runtime', 0)/60))
for n, st, step, tl, vl, rt in sorted(rows):
    fmt = lambda v: v if not isinstance(v, (int, float)) else round(v, 3)
    print(f"{n:32s} {st:9s} step={step} train={fmt(tl)} val={fmt(vl)} {rt:.0f}min")
