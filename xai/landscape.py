"""Misfit landscape (Sections IV-A3 and IV-B3): time-shift and model-line tests.

  1) time shift: misfit between each observed gather and the same gather shifted in time (3 shots);
  2) model line: misfit between the observed data and the data simulated in m(gamma) = m_init + gamma (m_true -
     m_init), for gamma from 0 (linear initial model) to 1 (true model) (6 shots).
Fixed misfits are built directly; the Siamese networks are read from the checkpoints of one inversion (epoch 0: random
initialization, which is also the network of the frozen ablation).

    python -m xai.landscape --model over
"""
import argparse
import time

import numpy as np
import torch

from xai import common
from utils import protocol
from utils.misfits import Misfit

LINE_SHOTS = (0, 5, 10, 15, 20, 25)
SHIFT_SHOTS = (0, 15, 29)
FIXED = ('l2', 'euclidean', 'identity', 'avgpool', 'randproj', 'maxpool', 'mpbae')
NETWORKS = {'siamesefit': (0, 10, 50, -1), 'siamesefwi': (0, -1)}  # -1: last epoch


def build_misfits(model, seed, dev):
    """(label, misfit) of the fixed operators and of the checkpointed networks."""
    out = []
    for name in FIXED:
        torch.manual_seed(seed)  # random projection
        out.append((name, Misfit(name, dev, model)))
    for method, epochs in NETWORKS.items():
        run = common.run_path(model, method, seed)
        last = common.config(run)['epochs']
        for epoch in epochs:
            epoch = last if epoch < 0 else epoch
            out.append(('%s@%d' % (method, epoch), common.misfit_of(run, model, dev, common.checkpoint(run, epoch, dev))))
    return out


def shift(d, n):
    """Shift a gather [nt, 1, nrec] by n time samples (positive: later), filling with zeros."""
    out = torch.zeros_like(d)
    if n > 0:
        out[n:] = d[:-n]
    elif n < 0:
        out[:n] = d[-n:]
    else:
        out = d.clone()
    return out


def basin_half_width(shifts, j):
    """Distance from zero shift to the nearest local maximum of the misfit (inf if none in the range)."""
    i0 = int(np.argmin(np.abs(shifts)))
    right = next((shifts[i] for i in range(i0 + 1, len(j) - 1) if j[i + 1] < j[i]), np.inf)
    left = next((-shifts[i] for i in range(i0 - 1, 0, -1) if j[i - 1] < j[i]), np.inf)
    return min(left, right)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--max-shift', type=float, default=0.5, help='largest time shift (s)')
    p.add_argument('--shift-step', type=int, default=4, help='step of the time shift (samples)')
    p.add_argument('--num-gamma', type=int, default=21)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    d_obs = s['d_obs']
    misfits = build_misfits(args.model, args.seed, dev)

    n_max = int(round(args.max_shift / protocol.DT))
    shifts = np.arange(-n_max, n_max + 1, args.shift_step)
    j_shift = np.zeros((len(misfits), len(shifts)))
    t0 = time.time()
    with torch.no_grad():
        for i, n in enumerate(shifts):
            for shot in SHIFT_SHOTS:
                obs = d_obs[:, shot:shot + 1]
                sim = shift(obs, int(n))
                for k, (_, f) in enumerate(misfits):
                    j_shift[k, i] += f(obs, sim).item()
        print('time shift done, %.0f s' % (time.time() - t0), flush=True)

        m_init = s['model'].detach().clone()
        m_init[:s['water_rows']] = protocol.WATER_VELOCITY
        gammas = np.linspace(0, 1, args.num_gamma)
        j_line = np.zeros((len(misfits), len(gammas)))
        for i, g in enumerate(gammas):
            d_sim = common.simulate(m_init + float(g) * (s['model_true'] - m_init), s, LINE_SHOTS)
            for j, shot in enumerate(LINE_SHOTS):
                for k, (_, f) in enumerate(misfits):
                    j_line[k, i] += f(d_obs[:, shot:shot + 1], d_sim[:, j:j + 1]).item()
        print('model line done, %.0f s' % (time.time() - t0), flush=True)

    labels = [label for label, _ in misfits]
    for k, label in enumerate(labels):
        print('%-16s basin half-width %.3f s' % (label, basin_half_width(shifts * protocol.DT, j_shift[k])))
    out = common.out_dir('landscape', args.model)
    np.savez('%s/landscape_seed%d.npz' % (out, args.seed), labels=np.array(labels), shifts=shifts * protocol.DT,
             j_shift=j_shift, gammas=gammas, j_line=j_line)
    print('saved in', out)


if __name__ == '__main__':
    main()
