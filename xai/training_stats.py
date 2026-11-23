"""Data-independent output of the trained SiameseFit networks (Section IV-A3, Table S5).

For each seed, the networks of epoch 0 and of the last epoch are applied to the normalized gather simulated in the
initial model for the middle shot, x, and to a zero input: reported are |phi(0)| / |phi(x)| at the last epoch (1: the
output is dominated by its data-independent part) and the growth of the output norm, |phi_last(x)| / |phi_0(x)|.

    python -m xai.training_stats --model over
"""
import argparse

import numpy as np
import torch

from xai import common
from utils import protocol
from utils.misfits import to_image


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    with torch.no_grad():
        m0 = common.checkpoint(common.run_path(args.model, 'siamesefit', 0), 0, dev)['model'].to(dev)
        x = to_image(common.simulate(m0, s, [common.MIDDLE_SHOT]))
        x = x / x.norm()
        zero_ratio, growth = [], []
        for seed in args.seeds:
            run = common.run_path(args.model, 'siamesefit', seed)
            last = common.config(run)['epochs']
            norms = []
            for epoch in (0, last):
                f = common.misfit_of(run, args.model, dev, common.checkpoint(run, epoch, dev))
                norms.append(float(f.net(x).norm()))
                if epoch == last:
                    zero_ratio.append(float(f.net(torch.zeros_like(x)).norm()) / norms[-1])
            growth.append(norms[1] / norms[0])
            print('seed %d: |phi(0)|/|phi(x)| %.5f, output norm x%.1f' % (seed, zero_ratio[-1], growth[-1]), flush=True)
    d = common.out_dir('training', args.model)
    np.savez('%s/training_stats.npz' % d, seeds=np.array(args.seeds), zero_ratio=np.array(zero_ratio),
             growth=np.array(growth))
    print('saved in', d)


if __name__ == '__main__':
    main()
