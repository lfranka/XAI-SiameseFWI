"""What the training changes in the SiameseFit encoder (Sections IV-A3 and IV-B3): linear CKA and weight changes.

For the observed gathers and the gathers simulated in the initial model (30 shots each, normalized as in the loss), the
Gram matrices of the encoder blocks are stored for every checkpoint, together with fixed references (input, 16 x 16
max-pooling and average pooling). Gram matrices of the observed-minus-simulated differences, which is what the loss
compares, are stored as well; linear CKA between any two representations follows from their Gram matrices.

    python -m xai.cka --model over
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from xai import common
from utils import protocol
from utils.misfits import Misfit, to_image

LAYERS = ('output1', 'output2', 'output3', 'output4', 'deep4', 'skip4')


def linear_cka(k, l):
    """Linear CKA from two Gram matrices."""
    n = k.shape[0]
    h = np.eye(n) - 1.0 / n
    kc, lc = h @ k @ h, h @ l @ h
    return float(np.sum(kc * lc) / np.sqrt(np.sum(kc * kc) * np.sum(lc * lc)))


def features(net, x):
    """Intermediate representations of the encoder for one input [1, 1, nt, nrec] (same as forward_once)."""
    xx = [net.act(c(x)) for c in (net.cnnXX1, net.cnnXX2, net.cnnXX3, net.cnnXX4)]
    out, feats = x, {}
    for k, cnn in enumerate((net.cnn1, net.cnn2, net.cnn3, net.cnn4)):
        deep = net.pool(net.act(cnn(out)))
        skip = net.pool_skip[k](xx[k])
        out = deep + skip
        feats['output%d' % (k + 1)] = out
    feats['deep4'], feats['skip4'] = deep, skip
    return feats


def gram(rows):
    """Gram matrix (float64) of a list of flattened representations."""
    x = torch.stack([r.flatten().double().cpu() for r in rows])
    return (x @ x.T).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--every', type=int, default=10, help='epoch step of the checkpoints for the first seeds')
    p.add_argument('--full-seeds', type=int, default=3,
                   help='number of seeds (from the first) analysed at every checkpoint; the others at the first and last')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    m0 = s['model'].detach().clone()
    m0[:s['water_rows']] = protocol.WATER_VELOCITY
    with torch.no_grad():
        d_sim = common.simulate(m0, s, range(protocol.NUM_SHOTS))
    inputs = []
    for d in (s['d_obs'], d_sim):
        for shot in range(protocol.NUM_SHOTS):
            x = to_image(d[:, shot:shot + 1])
            inputs.append(x / torch.norm(x, p=2))
    n = protocol.NUM_SHOTS  # inputs[:n] observed, inputs[n:] simulated, same shot order
    out = {'n_inputs': len(inputs)}
    with torch.no_grad():
        for name, op in (('input', nn.Identity()), ('maxpool16', nn.MaxPool2d(16, 16)),
                         ('avgpool16', nn.AvgPool2d(16, 16))):
            r = [op(x) for x in inputs]
            out['ref/' + name] = gram(r)
            out['refdiff/' + name] = gram([r[i] - r[i + n] for i in range(n)])
    t0 = time.time()
    for i, seed in enumerate(args.seeds):
        run = common.run_path(args.model, 'siamesefit', seed)
        last = common.config(run)['epochs']
        w0 = None
        for epoch in (range(0, last + 1, args.every) if i < args.full_seeds else (0, last)):
            state = common.checkpoint(run, epoch, dev)['net']
            f = Misfit('siamesefit', dev, args.model)
            f.net.load_state_dict(state)
            w0 = state if w0 is None else w0
            key = 'seed%d@%d' % (seed, epoch)
            with torch.no_grad():
                feats = [features(f.net, x) for x in inputs]
            for layer in LAYERS:
                out['%s/%s' % (key, layer)] = gram([fe[layer] for fe in feats])
                out['%s/diff_%s' % (key, layer)] = gram([feats[i][layer] - feats[i + n][layer] for i in range(n)])
            for name in state:
                if name.endswith('weight') and not name.startswith('cnn5'):
                    out['%s/wdist_%s' % (key, name.split('.')[0])] = float((state[name] - w0[name]).norm() / w0[name].norm())
            print('%s done, %.0f s' % (key, time.time() - t0), flush=True)
    d = common.out_dir('cka', args.model)
    np.savez('%s/cka_grams.npz' % d, **out)
    print('saved in', d)


if __name__ == '__main__':
    main()
