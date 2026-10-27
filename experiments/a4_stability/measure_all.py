"""Probe many checkpoints (init, step 100, final) on one Modal GPU.

    uv run python -m experiments.a4_stability.measure_all shared        # supplied P1/P2 runs
    uv run python -m experiments.a4_stability.measure_all mine a4-gru-lr0.003 ...

Each (run, stage) writes results/a4_stability/<run>/<stage>.json on the user
volume, so a crash keeps earlier work. Download with
    modal volume get --env cs312-awd <user volume> results/a4_stability ~/a4work/results
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

SHARED_ROOT = Path('/root/shared_data/a4_stability')
MY_ROOT = Path('/root/data/ckpts')
OUT_ROOT = Path('/root/data/results/a4_stability')
PROFILE_BATCH = 8
CONTEXT_BATCH = 64
KEEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1023)


def variant_and_lr(run):
    name = run[3:]
    stem, lr = name.rsplit('-lr', 1)
    return stem, float(lr)


def early_path(run_dir):
    for candidate in (run_dir/'early', run_dir/'step_100.pt'):
        if candidate.exists():
            return candidate
    return None


LAGS = (1, 4, 16, 64, 256)


@torch.no_grad()
def position_profile(model, tokens):
    """Per depth index: activation RMS at every position and cosine between positions at fixed lags."""
    import torch.nn.functional as F
    from experiments.a4_stability.p2_rnn_depth import Recurrent
    stream, handles = [], []
    for branch in model.branches():
        handles.append(branch.register_forward_pre_hook(lambda m, a: stream.append(a[0].detach().float())))
    handles.append(model.output_norm.register_forward_pre_hook(lambda m, a: stream.append(a[0].detach().float())))
    training = model.training
    model.eval()
    try:
        model(tokens[:, :-1])
    finally:
        for h in handles:
            h.remove()
        model.train(training)
    out = []
    for x in stream:
        row = dict(rms_by_position=x.square().mean((0, 2)).sqrt().tolist(),
                   cosine_by_lag={str(l): F.cosine_similarity(x[:, l:], x[:, :-l], dim=-1).mean().item() for l in LAGS if l < x.shape[1]},
                   cosine_to_first={str(l): F.cosine_similarity(x[:, l], x[:, 0], dim=-1).mean().item() for l in LAGS if l < x.shape[1]})
        out.append(row)
    return dict(depth=out, length=stream[0].shape[1])


def measure(jobs, device='cuda'):
    """jobs: list of dict(run, root, stages, context). Returns list of done keys."""
    from data import dclm_val_dataset, load_token_dataset
    from model_io import load_model
    from train import build_model
    from experiments.a4_stability.sweeps import config
    from experiments.a4_stability.diagnostics import profile, context_loss, save
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    all_tokens = torch.tensor(validation[:CONTEXT_BATCH]['input_ids'].astype('int64'), device=device)
    tokens = all_tokens[:PROFILE_BATCH]
    done = []
    for job in jobs:
        run, root = job['run'], Path(job['root'])
        out_dir = OUT_ROOT/run
        for stage in job['stages']:
            out = out_dir/f'{stage}.json'
            if out.exists():
                done.append(f'{run}/{stage} (cached)')
                continue
            start = time.time()
            try:
                if stage == 'val':
                    import torch.nn.functional as F
                    model = load_model(root/run, device=device, precision='fp32').float().to(device)
                    model.eval()
                    total, count = 0., 0
                    with torch.no_grad():
                        for i in range(0, len(validation), 32):
                            ids = torch.tensor(validation[i:i+32]['input_ids'].astype('int64'), device=device)
                            logits = model(ids[:, :-1])
                            total += F.cross_entropy(logits.flatten(0, 1).float(), ids[:, 1:].flatten(), reduction='sum').item()
                            count += ids[:, 1:].numel()
                    stats = dict(val_loss=total/count, sequences=len(validation), seconds=time.time()-start)
                    save(out, stats)
                    done.append(f'{run}/{stage} {stats["val_loss"]:.4f} {stats["seconds"]:.0f}s')
                    continue
                if stage == 'position':
                    model = load_model(root/run, device=device, precision='fp32').float().to(device)
                    stats = position_profile(model, tokens)
                    stats['seconds'] = time.time()-start
                    save(out, stats)
                    done.append(f'{run}/{stage} {stats["seconds"]:.0f}s')
                    continue
                if stage == 'init':
                    variant, lr = variant_and_lr(run)
                    cfg = replace(config(variant, lr), precision='fp32')
                    torch.manual_seed(cfg.model_seed)
                    model = build_model(cfg, device)
                else:
                    path = root/run if stage == 'final' else early_path(root/run)
                    if path is None:
                        done.append(f'{run}/{stage} (missing)')
                        continue
                    model = load_model(path, device=device, precision='fp32')
                model = model.float().to(device)
                stats = profile(model, tokens)
                stats['num_parameters'] = sum(p.numel() for p in model.parameters())
                if stage in job.get('context', ()):
                    with torch.no_grad():
                        full = torch.cat([context_loss(model, all_tokens[i:i+32], keep=all_tokens.shape[1]-1)
                                          for i in range(0, CONTEXT_BATCH, 32)])
                        ctx = {}
                        for keep in KEEPS:
                            for first in (False, True):
                                if keep == KEEPS[-1] and first:
                                    continue
                                losses = torch.cat([context_loss(model, all_tokens[i:i+32], keep, first)
                                                    for i in range(0, CONTEXT_BATCH, 32)])
                                ctx[f'{keep}_first_{first}'] = dict(mean_increase=(losses-full).mean().item(),
                                                                    mean_loss=losses.mean().item())
                        stats['context'] = dict(full_last_token_loss=full.mean().item(), tests=ctx)
                stats['seconds'] = time.time()-start
                save(out, stats)
                done.append(f'{run}/{stage} {stats["seconds"]:.0f}s')
            except Exception as e:  # keep going; record the failure
                save(out_dir/f'{stage}.error.json', dict(error=repr(e)))
                done.append(f'{run}/{stage} ERROR {e!r}')
            finally:
                model = None
                torch.cuda.empty_cache()
            print(done[-1], flush=True)
    return done


def shared_jobs(manifest):
    """Final profile for every supplied run; init, early and context tests for the best rate per variant."""
    runs = {}
    for entry in manifest['runs']:
        runs[entry['run_name']] = entry
    best = {}
    for run, entry in runs.items():
        loss = entry.get('validation_loss')
        if loss is None or entry.get('status') != 'complete':
            continue
        variant = entry['variant']
        if variant not in best or loss < best[variant][1]:
            best[variant] = (run, loss)
    chosen = {run for run, _ in best.values()}
    jobs = []
    for run in sorted(runs):
        if run in chosen:
            jobs.append(dict(run=run, root=str(SHARED_ROOT), stages=('init', 'early', 'final'), context=('final',)))
        elif runs[run].get('status') == 'complete':
            jobs.append(dict(run=run, root=str(SHARED_ROOT), stages=('final',)))
    return sorted(jobs, key=lambda j: j['run'] not in chosen)


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    mode, names = sys.argv[1], sys.argv[2:]
    if mode in ('position', 'val'):
        root = SHARED_ROOT if os.environ.get('A4_ROOT', 'shared') == 'shared' else MY_ROOT
        jobs = [dict(run=run, root=str(root), stages=(mode,)) for run in names]
    elif mode == 'shared':
        manifest = json.loads(Path(os.environ.get('A4_MANIFEST', os.path.expanduser('~/a4work/shared/manifest.json'))).read_text())
        jobs = shared_jobs(manifest)
    else:
        jobs = [dict(run=run, root=str(MY_ROOT), stages=('init', 'early', 'final'), context=('final',)) for run in names]
    print(f'{len(jobs)} runs')
    # The course GPU limit covers all GPU types, so while training occupies both
    # slots the FP32 probes run on CPU containers instead (A4_GPU=cpu), in chunks.
    gpu = os.environ.get('A4_GPU', 'H100')
    chunk = int(os.environ.get('A4_CHUNK', 4 if gpu == 'cpu' else len(jobs)))
    chunks = [jobs[i:i+chunk] for i in range(0, len(jobs), chunk)]
    options = dict(gpu=None, cpu=8, memory=24*1024) if gpu == 'cpu' else dict(gpu=gpu)
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=4*3600, **options)(measure)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        for lines in probe.map(chunks, ['cpu' if gpu == 'cpu' else 'cuda']*len(chunks)):
            for line in lines:
                print(line)
