"""Training on fixed data (Sections IV-A3 and IV-B3): adjoint source of the Siamese networks along the training.

The network stored at each checkpoint is applied to the same pair of gathers (observed data and data simulated in the
initial model, middle shot), so that the change of the adjoint source comes from the training only. The Euclidean and
16 x 16 max-pooling misfits on the same data are stored as references.

    python -m xai.fixed_data --model over
"""
import argparse
import time

import numpy as np
import torch

from xai import common
from utils import protocol
from utils.misfits import Misfit

TRAINED = ('siamesefwi', 'siamesefit')
REFERENCES = ('euclidean', 'maxpool')
EPOCHS_SHOW = (0, 10, 50, 100, 200)


def adjoint(f, obs, sim):
    """Adjoint source dJ/d(sim) and misfit value."""
    x = sim.clone().requires_grad_(True)
    loss = f(obs, x)
    g, = torch.autograd.grad(loss, x)
    return g.detach().cpu().numpy()[:, 0], loss.item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shot', type=int, default=common.MIDDLE_SHOT)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    m0 = common.checkpoint(common.run_path(args.model, 'siamesefit', args.seed), 0, dev)['model'].to(dev)
    with torch.no_grad():
        sim = common.simulate(m0, s, [args.shot])
    obs = s['d_obs'][:, args.shot:args.shot + 1]
    out = {'shot': args.shot, 'epochs_show': np.array(EPOCHS_SHOW)}
    for name in REFERENCES:
        out['ref_%s' % name], _ = adjoint(Misfit(name, dev, args.model), obs, sim)
    t0 = time.time()
    for label in TRAINED:
        run = common.run_path(args.model, label, args.seed)
        last = common.config(run)['epochs']
        epochs = list(range(0, last + 1, 10))
        f = Misfit(label, dev, args.model)
        stats = {k: [] for k in ('cos0', 'cos_euclidean', 'cos_maxpool', 'norm', 'loss')}
        for epoch in epochs:
            f.net.load_state_dict(common.checkpoint(run, epoch, dev)['net'])
            a, loss = adjoint(f, obs, sim)
            a0 = a if epoch == 0 else a0
            if epoch in EPOCHS_SHOW or epoch == last:
                out['%s@%d' % (label, epoch)] = a
            stats['cos0'].append(common.cos(a, a0))
            stats['cos_euclidean'].append(common.cos(a, out['ref_euclidean']))
            stats['cos_maxpool'].append(common.cos(a, out['ref_maxpool']))
            stats['norm'].append(np.linalg.norm(a) / np.linalg.norm(a0))
            stats['loss'].append(loss)
        out['%s/epochs' % label] = np.array(epochs)
        for k, v in stats.items():
            out['%s/%s' % (label, k)] = np.array(v)
        print('%s: cosine with epoch 0 at the last epoch %.3f, norm %.3g, misfit %.3g of epoch 0 | %.0f s'
              % (label, stats['cos0'][-1], stats['norm'][-1], stats['loss'][-1] / stats['loss'][0],
                 time.time() - t0), flush=True)
    d = common.out_dir('fixed_data', args.model)
    np.savez('%s/fixed_data_seed%d.npz' % (d, args.seed), **out)
    print('saved in', d)


if __name__ == '__main__':
    main()
