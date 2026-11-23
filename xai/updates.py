"""Updates along the inversions (Section IV-B3 and supplementary Section S4, Table S6).

A1 (simulation at the stored checkpoints of the max-pooling and frozen-network runs, six shots): position of the
   maximum of each 16 x 16 block of the normalized simulated gather, fraction of the blocks whose maximum stays at the
   same sample, fraction of the block maxima in the early arrivals, and concentration of the adjoint source.
A2 (checkpoints only): actual update between successive checkpoints D = m_{k+1} - m_k and error e = m_true - m_k,
   below the water layer. The error decreases if and only if c = <D, e> / (|D| |e|) exceeds t = |D| / (2 |e|).
A4: alignment of the velocity gradient of each single shot (all shots) at a few checkpoints, in the whole model, in the
   shallowest kilometre and for its sign (Adam updates the model after every shot with a short memory).

    python -m xai.updates --model marm
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from xai import common
from xai.adjoint import partitions
from utils import protocol
from utils.misfits import POOL, to_image

A1_RUNS = ('maxpool', 'frozen')
A2_RUNS = ('maxpool', 'frozen', 'siamesefit', 'mpbae', 'euclidean', 'skip128', 'maxpool_shift', 'maxpool_top16')
A4_RUNS = ('maxpool', 'frozen', 'skip128', 'maxpool_shift', 'maxpool_top16', 'siamesefit')
BANDS = ((0.0, 1000.0), (1000.0, 2000.0), (2000.0, np.inf))


def update_stats(models, true, water, depth):
    """A2 quantities for a sequence of velocity models [n, nz, nx]."""
    m = models[:, water:].reshape(len(models), -1).astype(np.float64)
    t = true[water:].ravel().astype(np.float64)
    nx = models.shape[2]
    band_idx = [np.repeat((depth[water:] >= lo) & (depth[water:] < hi), nx) for lo, hi in BANDS]
    out = {k: [] for k in ('c', 'threshold', 'coherence', 'band_share', 'snr')}
    prev_perp = None
    for k in range(len(m) - 1):
        d, e = m[k + 1] - m[k], t - m[k]
        ue = e / np.linalg.norm(e)
        perp = d - np.dot(d, ue) * ue
        out['c'].append(common.cos(d, e))
        out['threshold'].append(np.linalg.norm(d) / (2 * np.linalg.norm(e)))
        out['coherence'].append(common.cos(perp, prev_perp) if prev_perp is not None else np.nan)
        out['band_share'].append([np.sum(perp[b] ** 2) / np.sum(perp ** 2) for b in band_idx])
        out['snr'].append(10 * np.log10(np.sum(t ** 2) / np.sum((m[k + 1] - t) ** 2)))
        prev_perp = perp
    return {k: np.array(v) for k, v in out.items()}


def a1(model, seed, s, dev):
    shots = list(common.SHOTS)
    masks = partitions(s, shots)
    out = {}
    for label in A1_RUNS:
        run = common.run_path(model, label, seed)
        f = common.misfit_of(run, model, dev)
        first, prev_idx, prev_adj = {}, {}, {}
        rows = {k: [] for k in ('epochs', 'same_prev', 'same_first', 'early_share', 'adj_cos', 'half_energy')}
        for epoch, path in common.checkpoints(run):
            state = torch.load(path, map_location=dev)
            if f.net is not None:
                f.net.load_state_dict(state['net'])
            sp, sf, ee, ac, he = [], [], [], [], []
            for j, shot in enumerate(shots):
                with torch.no_grad():
                    d_sim = common.simulate(state['model'].to(dev), s, [shot])
                d_sim = d_sim.detach().requires_grad_(True)
                f(s['d_obs'][:, shot:shot + 1], d_sim).backward()
                a = d_sim.grad[:, 0].cpu().numpy().ravel()
                x = to_image(d_sim.detach())
                _, idx = F.max_pool2d(x / x.norm(), POOL, stride=POOL, return_indices=True)
                idx = idx.flatten().cpu().numpy()
                ee.append(float(masks[j]['early'][:, 0].numpy().ravel()[idx].mean()))
                if shot in prev_idx:
                    sp.append(float(np.mean(idx == prev_idx[shot])))
                    sf.append(float(np.mean(idx == first[shot])))
                    ac.append(common.cos(a, prev_adj[shot]))
                else:
                    first[shot] = idx
                prev_idx[shot], prev_adj[shot] = idx, a
                e2 = np.sort(a ** 2)[::-1]
                he.append(float(np.searchsorted(np.cumsum(e2), 0.5 * e2.sum()) / e2.size))
            rows['epochs'].append(epoch)
            for k, v in (('same_prev', sp), ('same_first', sf), ('early_share', ee), ('adj_cos', ac), ('half_energy', he)):
                rows[k].append(np.mean(v) if v else np.nan)
        for k, v in rows.items():
            out['A1/%s/%s' % (label, k)] = np.array(v)
        print('A1 %-8s maxima unchanged since epoch 0 %.3f, in the early arrivals %.3f, half of the adjoint energy in '
              '%.4f of the samples' % (label, rows['same_first'][-1], np.mean(rows['early_share']),
                                       np.mean(rows['half_energy'])), flush=True)
    return out


def a4(model, seed, s, dev, true, water, depth, epochs, labels=A4_RUNS):
    shallow = (depth < 1000) & (np.arange(len(depth)) >= water)
    out = {}
    for label in labels:
        run = common.run_path(model, label, seed)
        if not os.path.exists(os.path.join(run, 'model_final.npy')):
            continue
        for epoch in epochs:
            state = common.checkpoint(run, epoch, dev)
            f = common.misfit_of(run, model, dev, state)
            e = true - state['model'].cpu().numpy()
            c, c_shallow, c_sign = [], [], []
            for shot in range(protocol.NUM_SHOTS):
                torch.manual_seed(shot)  # offsets of the shifted grid
                m = state['model'].to(dev).clone().requires_grad_(True)
                f(s['d_obs'][:, shot:shot + 1], common.simulate(m, s, [shot])).backward()
                g = m.grad.detach().cpu().numpy()
                c.append(common.cos(-g[water:], e[water:]))
                c_shallow.append(common.cos(-g[shallow], e[shallow]))
                c_sign.append(common.cos(-np.sign(g[water:]), e[water:]))
            for k, v in (('c', c), ('c_shallow', c_shallow), ('c_sign', c_sign)):
                out['A4/%s@%d/%s' % (label, epoch, k)] = np.array(v)
            print('A4 %-14s epoch %3d: single-shot alignment %.3f, shallowest km %.3f, sign %.3f'
                  % (label, epoch, np.mean(c), np.mean(c_shallow), np.mean(c_sign)), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='marm', choices=list(protocol.MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--parts', nargs='+', default=['A1', 'A2', 'A4'])
    p.add_argument('--a4-runs', nargs='+', default=list(A4_RUNS))
    p.add_argument('--a4-epochs', type=int, nargs='+', default=[0, 10, 50])
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    true = s['model_true'].cpu().numpy()
    water = s['water_rows']
    depth = np.arange(true.shape[0]) * protocol.DX
    d = common.out_dir('updates', args.model)
    path = '%s/updates_seed%d.npz' % (d, args.seed)
    results = dict(np.load(path)) if os.path.exists(path) else {}
    if 'A2' in args.parts:
        for label in A2_RUNS:
            run = common.run_path(args.model, label, args.seed)
            if not os.path.exists(os.path.join(run, 'model_final.npy')):
                continue
            ck = common.checkpoints(run)
            r = update_stats(np.array([torch.load(q, map_location='cpu')['model'].numpy() for _, q in ck]), true, water, depth)
            for k, v in r.items():
                results['A2/%s/%s' % (label, k)] = v
            results['A2/%s/epochs' % label] = np.array([e for e, _ in ck])
            print('A2 %-14s alignment %s threshold %s' % (label, np.round(r['c'][:3], 2), np.round(r['threshold'][:3], 2)))
    if 'A1' in args.parts:
        results.update(a1(args.model, args.seed, s, dev))
    if 'A4' in args.parts:
        results.update(a4(args.model, args.seed, s, dev, true, water, depth, args.a4_epochs, args.a4_runs))
    np.savez(path, **results)
    print('saved in', d)


if __name__ == '__main__':
    main()
