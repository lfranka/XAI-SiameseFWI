"""Occlusion (Sections IV-A3 and IV-B3): effect of correcting one region of the simulated data on the misfit and on the
direction of the update.

For each run and stored epoch and six shots, each region of the simulated gather (early arrivals, later arrivals,
offset bands of xai.adjoint) is replaced by the observed data, and the fraction of the misfit removed and the alignment
of the velocity gradient with the direction to the true model are recorded.

    python -m xai.occlusion --model over
"""
import argparse
import time

import numpy as np
import torch

from xai import common
from xai.adjoint import OFFSETS, RUNS, partitions
from utils import protocol

REGIONS = ['early', 'late'] + ['offset_%g-%g' % (lo / 1000, hi / 1000) for lo, hi in zip(OFFSETS[:-1], OFFSETS[1:])]


def region_masks(masks):
    """Masks [nt, 1, nrec] of each region, per shot, in the order of REGIONS."""
    return [[m['early'], m['late']] + list(m['offset']) for m in masks]


def run_case(f, m0, s, shots, masks, region):
    """Total misfit and gradient alignment with one region (or none) replaced by the observed data."""
    m = m0.clone().requires_grad_(True)
    total = 0.0
    for j, shot in enumerate(shots):
        d_sim = common.simulate(m, s, [shot])
        obs = s['d_obs'][:, shot:shot + 1]
        if region is not None:
            mask = masks[j][region].to(d_sim.device).float()
            d_sim = d_sim * (1 - mask) + obs * mask
        loss = f(obs, d_sim)
        loss.backward()
        total += loss.item()
    return total, common.alignment(m.grad.detach(), m0, s['model_true'], s['water_rows'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, nargs='+', default=[0, 10, -1], help='-1: last epoch')
    p.add_argument('--runs', nargs='+', default=list(RUNS))
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    shots = list(common.SHOTS)
    masks = region_masks(partitions(s, shots))
    results, t0 = {}, time.time()
    for label in args.runs:
        run = common.run_path(args.model, label, args.seed)
        last = common.config(run)['epochs']
        for epoch in args.epochs:
            epoch = last if epoch < 0 else epoch
            state = common.checkpoint(run, epoch, dev)
            f, m0 = common.misfit_of(run, args.model, dev, state), state['model'].to(dev)
            j_full, cos_full = run_case(f, m0, s, shots, masks, None)
            removed, cos_occ = [], []
            for r in range(len(REGIONS)):
                j_occ, c = run_case(f, m0, s, shots, masks, r)
                removed.append((j_full - j_occ) / j_full)
                cos_occ.append(c)
            key = '%s@%d' % (label, epoch)
            results[key + '/removed'] = np.array(removed)
            results[key + '/cos_occluded'] = np.array(cos_occ)
            results[key + '/cos_full'] = cos_full
            print('%s: removed %s | alignment full %.3f occluded %s | %.0f s' % (
                key, np.round(removed, 2), cos_full, np.round(cos_occ, 3), time.time() - t0), flush=True)
    out = common.out_dir('occlusion', args.model)
    np.savez('%s/occlusion_seed%d.npz' % (out, args.seed), regions=np.array(REGIONS), **results)
    print('saved in', out)


if __name__ == '__main__':
    main()
