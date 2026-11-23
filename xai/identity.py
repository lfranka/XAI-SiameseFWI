"""Departure of the SiameseFWI network from the identity (Sections IV-A3 and IV-B3, Table S5).

The network returns o = x + R(x), where x is the input gather. For the checkpoints of the inversions and six shots
simulated in the model of the same epoch, this measures ||R(x)|| / ||x|| for the observed and simulated inputs, ||o|| /
||x||, the ratio between the SiameseFWI loss and the Euclidean loss on x (identity network), and the cosine between
their adjoint sources (1: the network acts as the identity).

    python -m xai.identity --model over
"""
import argparse
import time

import numpy as np
import torch

from xai import common
from utils import protocol
from utils.misfits import euclidean, to_image


def analyse(net, m0, s, shots):
    """Mean over the shots of the identity-departure measures at one checkpoint."""
    out = {'r_obs': [], 'r_sim': [], 'o_norm': [], 'loss_ratio': [], 'cos_adjoint': []}
    with torch.no_grad():
        d_sims = common.simulate(m0, s, shots)
    for j, shot in enumerate(shots):
        d_sim = d_sims[:, j:j + 1].clone().requires_grad_(True)
        x_obs, x_sim = to_image(s['d_obs'][:, shot:shot + 1]), to_image(d_sim)
        o_obs, o_sim = net.forward_once(x_obs)[1], net.forward_once(x_sim)[1]
        j_net, j_id = euclidean(o_sim, o_obs), euclidean(x_sim, x_obs)
        a_net, = torch.autograd.grad(j_net, d_sim, retain_graph=True)
        a_id, = torch.autograd.grad(j_id, d_sim)
        out['r_obs'].append(float((o_obs - x_obs).norm() / x_obs.norm()))
        out['r_sim'].append(float((o_sim - x_sim).norm() / x_sim.norm()))
        out['o_norm'].append(float(o_sim.norm() / x_sim.norm()))
        out['loss_ratio'].append(float(j_net / j_id))
        out['cos_adjoint'].append(float((a_net * a_id).sum() / (a_net.norm() * a_id.norm())))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--every', type=int, default=10, help='epoch step of the checkpoints for the first seed')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    shots = list(common.SHOTS)
    results, t0 = {}, time.time()
    for i, seed in enumerate(args.seeds):
        run = common.run_path(args.model, 'siamesefwi', seed)
        last = common.config(run)['epochs']
        for epoch in (range(0, last + 1, args.every) if i == 0 else (0, last)):
            state = common.checkpoint(run, epoch, dev)
            f = common.misfit_of(run, args.model, dev, state)
            r = analyse(f.net, state['model'].to(dev), s, shots)
            for key, value in r.items():
                results['siamesefwi/seed%d@%d/%s' % (seed, epoch, key)] = value
            print('seed%d epoch %d: |R|/|x| %.3f, loss ratio %.3f, cosine of the adjoint sources %.3f, %.0f s'
                  % (seed, epoch, r['r_sim'], r['loss_ratio'], r['cos_adjoint'], time.time() - t0), flush=True)
    out = common.out_dir('identity', args.model)
    np.savez('%s/identity.npz' % out, **results)
    print('saved in', out)


if __name__ == '__main__':
    main()
