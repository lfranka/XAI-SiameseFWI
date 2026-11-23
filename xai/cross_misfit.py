"""Cross-evaluation of the misfits at the final models of other inversions (Section IV-B3, Table 5).

The 16 x 16 max-pooling and the frozen SiameseFit network (both with the SiameseFit loss), MPBAE and the squared L2 norm
of the data residual are evaluated over all shots at the initial model, at the true model and at the final models of
several inversions (seed 0), relative to their values at the initial model. For the runs analysed by xai.adjoint, the
alignment of the stored velocity gradient at epoch 10 is also computed in depth bands and for the sign of the gradient,
which is the direction of the first Adam update of each cell when eps is negligible.

    python -m xai.cross_misfit --model marm
"""
import argparse
import os

import numpy as np
import torch

from xai import common
from utils import protocol
from utils.fwi import model_metrics
from utils.misfits import Misfit

RUNS = ('maxpool', 'maxpool_lr20', 'frozen', 'siamesefit', 'mpbae', 'euclidean')  # final models
ADJOINT = ('maxpool', 'frozen', 'siamesefit', 'mpbae', 'euclidean')              # runs of xai.adjoint
BANDS = ((0.0, 1000.0), (1000.0, 2000.0), (2000.0, np.inf))
ALIGN_EPOCH = 10


def band_alignment(model, seed, s):
    """Alignment of the stored gradient at ALIGN_EPOCH per depth band and for its sign (rows: bands, then sign)."""
    path = os.path.join(common.XAI_ROOT, 'adjoint', '%s_linear' % model, 'adjoint_seed%d.npz' % seed)
    d = np.load(path) if os.path.exists(path) else {}
    true, water = s['model_true'].cpu().numpy(), s['water_rows']
    depth = np.arange(true.shape[0]) * protocol.DX
    out = {}
    for label in ADJOINT:
        k = '%s@%d/grad' % (label, ALIGN_EPOCH)
        if k not in d:
            continue
        m = common.checkpoint(common.run_path(model, label, seed), ALIGN_EPOCH, 'cpu')['model'].numpy()
        g, direction = d[k], true - m
        rows = [(depth >= lo) & (depth < hi) & (np.arange(len(depth)) >= water) for lo, hi in BANDS]
        out[label] = [common.cos(-g[r], direction[r]) for r in rows] + [common.cos(-np.sign(g[water:]), direction[water:])]
    return out


def evaluate(m, s, misfits):
    """Sum over all shots of each misfit between the observed data and the data simulated in m."""
    out = {k: 0.0 for k in misfits}
    with torch.no_grad():
        for start in range(0, protocol.NUM_SHOTS, 6):
            shots = list(range(start, min(start + 6, protocol.NUM_SHOTS)))
            d_sim = common.simulate(m, s, shots)
            for j, shot in enumerate(shots):
                for k, f in misfits.items():
                    out[k] += float(f(s['d_obs'][:, shot:shot + 1], d_sim[:, j:j + 1]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='marm', choices=list(protocol.MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    frozen_run = common.run_path(args.model, 'frozen', args.seed)
    state0 = common.checkpoint(frozen_run, 0, dev)
    misfits = {'max-pool': Misfit('maxpool', dev, args.model), 'frozen encoder': common.misfit_of(frozen_run, args.model, dev, state0),
               'MPBAE': Misfit('mpbae', dev, args.model), 'data L2': Misfit('l2', dev, args.model)}
    models = {'initial': state0['model'].to(dev), 'true': s['model_true']}
    for label in RUNS:
        run = common.run_path(args.model, label, args.seed)
        if os.path.exists(os.path.join(run, 'model_final.npy')):
            models[label] = common.checkpoint(run, common.config(run)['epochs'], dev)['model'].to(dev)
    values = {label: evaluate(m, s, misfits) for label, m in models.items()}
    snr = {label: model_metrics(m.cpu(), s['model_true'].cpu())[0] for label, m in models.items()}
    align = band_alignment(args.model, args.seed, s)
    ref = values['initial']
    print('%-14s %8s ' % ('final model', 'SNR') + ' '.join('%15s' % k for k in misfits))
    for label in models:
        print('%-14s %8.2f ' % (label, snr[label]) + ' '.join('%15.3f' % (values[label][k] / ref[k]) for k in misfits)
              + ('   alignment (bands, sign) %s' % np.round(align[label], 3) if label in align else ''))
    d = common.out_dir('cross_misfit', args.model)
    np.savez('%s/cross_misfit_seed%d.npz' % (d, args.seed), models=np.array(list(models)), misfits=np.array(list(misfits)),
             snr=np.array([snr[m] for m in models]), values=np.array([[values[m][k] for k in misfits] for m in models]),
             align=np.array([align.get(m, [np.nan] * (len(BANDS) + 1)) for m in models]), align_epoch=ALIGN_EPOCH)
    print('saved in', d)


if __name__ == '__main__':
    main()
