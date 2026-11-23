"""Adjoint sources, attributions and alignment of the update (Sections IV-A3 and IV-B3).

For each run and stored epoch, the velocity model and the network are read from the checkpoint and, for six shots:
  - adjoint source a = dJ/dd_sim (gradient saliency); its energy is split by time window (early arrivals, up to 0.4 s
    after the first break picked on the observed data, and later arrivals), by offset band and by frequency band;
  - integrated gradients with the observed data as reference, split in the same way; by completeness they add up to
    J(d_sim) - J(d_obs);
  - velocity gradient summed over the shots and its alignment with the direction to the true model, over the cells
    updated by the inversion (below the water layer of the Marmousi2 model).

    python -m xai.adjoint --model over --ig-steps 32
"""
import argparse
import time

import numpy as np
import torch

from xai import common
from utils import protocol

EARLY = 0.4                                   # length of the early-arrival window after the first break (s)
OFFSETS = (0.0, 3000.0, 6000.0, np.inf)       # offset bands (m)
FREQS = (0.0, 3.0, 6.0, 10.0, np.inf)         # frequency bands (Hz)
RUNS = ('l2', 'euclidean', 'siamesefwi', 'siamesefit', 'mpbae', 'maxpool', 'frozen')


def first_break(d, threshold=0.02):
    """Index of the first sample of each trace above threshold x trace maximum. d: [nt, nrec]."""
    a = d.abs()
    above = a > threshold * a.max(0, keepdim=True).values
    return above.float().argmax(0)


def partitions(s, shots):
    """Boolean masks [nt, 1, nrec] per shot, on the CPU: early/late arrivals and offset bands."""
    nt = s['d_obs'].shape[0]
    t = torch.arange(nt).view(-1, 1)
    masks = []
    for shot in shots:
        fb = first_break(s['d_obs'][:, shot].cpu())
        early = (t < fb.view(1, -1) + int(EARLY / protocol.DT)).unsqueeze(1)
        offset = (s['x_r'][shot, :, 1] - s['x_s'][shot, 0, 1]).abs().cpu()
        bands = [((offset >= lo) & (offset < hi)).view(1, 1, -1).expand(nt, 1, -1)
                 for lo, hi in zip(OFFSETS[:-1], OFFSETS[1:])]
        masks.append({'early': early, 'late': ~early, 'offset': bands})
    return masks


def split(x, mask):
    """Sums of x (one shot, [nt, 1, nrec]) over the time windows and offset bands."""
    return ([float(x[mask['early']].sum()), float(x[mask['late']].sum())],
            [float(x[m].sum()) for m in mask['offset']])


def band_energy(a):
    """Energy of a [nt, 1, nrec] in the frequency bands, and its amplitude spectrum averaged over traces."""
    spec = np.abs(np.fft.rfft(a[:, 0].numpy(), axis=0)) ** 2
    f = np.fft.rfftfreq(a.shape[0], protocol.DT)
    return [float(spec[(f >= lo) & (f < hi)].sum()) for lo, hi in zip(FREQS[:-1], FREQS[1:])], spec.mean(1)


def integrated_gradients(f, obs, sim, steps):
    """Integrated gradients of J(obs, .) from obs (reference) to sim, midpoint rule."""
    diff = sim - obs
    acc = torch.zeros_like(sim)
    for b in (np.arange(steps) + 0.5) / steps:
        x = (obs + float(b) * diff).requires_grad_(True)
        g, = torch.autograd.grad(f(obs, x), x)
        acc += g
    return diff * acc / steps


def analyse(f, m0, s, shots, masks, ig_steps):
    """Adjoint source, attributions and gradient of one misfit at one velocity model."""
    m = m0.clone().requires_grad_(True)
    out = {'win': np.zeros(2), 'off': np.zeros(len(OFFSETS) - 1), 'freq': np.zeros(len(FREQS) - 1),
           'ig_win': np.zeros(2), 'ig_off': np.zeros(len(OFFSETS) - 1), 'loss': 0.0, 'ig_sum': 0.0}
    spectra = []
    for j, shot in enumerate(shots):
        d_sim = common.simulate(m, s, [shot])
        d_sim.retain_grad()
        obs = s['d_obs'][:, shot:shot + 1]
        loss = f(obs, d_sim)
        loss.backward()  # accumulates the velocity gradient over the shots
        a = d_sim.grad.detach().cpu()
        win, off = split(a ** 2, masks[j])
        freq, spec = band_energy(a)
        out['win'] += win
        out['off'] += off
        out['freq'] += freq
        out['loss'] += loss.item()
        spectra.append(spec)
        if j == len(shots) // 2:
            out['adjoint_mid'] = a[:, 0].numpy()
        if ig_steps > 0:
            with torch.no_grad():  # misfit at the reference; not zero for the SiameseFit loss (eps of the distance)
                out['loss_ref'] = out.get('loss_ref', 0.0) + f(obs, obs.clone()).item()
            ig = integrated_gradients(f, obs, d_sim.detach(), ig_steps).cpu()
            win, off = split(ig, masks[j])
            out['ig_win'] += win
            out['ig_off'] += off
            out['ig_sum'] += float(ig.sum())
    g = m.grad.detach()
    out['grad'] = g.cpu().numpy()
    out['cos_true'] = common.alignment(g, m0, s['model_true'], s['water_rows'])
    out['spectrum'] = np.mean(spectra, 0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, nargs='+', default=[0, 10, 50, 100, -1], help='-1: last epoch')
    p.add_argument('--ig-steps', type=int, default=32, help='steps of the integrated gradients (0: skip)')
    p.add_argument('--runs', nargs='+', default=list(RUNS))
    p.add_argument('--device', default=None)
    args = p.parse_args()
    dev = common.device(args.device)
    s = protocol.setup(args.model, 'linear', dev)
    shots = list(common.SHOTS)
    masks = partitions(s, shots)
    results, t0 = {}, time.time()
    for label in args.runs:
        run = common.run_path(args.model, label, args.seed)
        last = common.config(run)['epochs']
        for epoch in args.epochs:
            epoch = last if epoch < 0 else epoch
            state = common.checkpoint(run, epoch, dev)
            r = analyse(common.misfit_of(run, args.model, dev, state), state['model'].to(dev), s, shots, masks,
                        args.ig_steps)
            for key, value in r.items():
                results['%s@%d/%s' % (label, epoch, key)] = value
            print('%s epoch %d: alignment %.3f, early fraction %.2f, IG completeness %.3g / %.3g, %.0f s'
                  % (label, epoch, r['cos_true'], r['win'][0] / r['win'].sum(), r['ig_sum'],
                     r['loss'] - r.get('loss_ref', 0.0), time.time() - t0), flush=True)
    out = common.out_dir('adjoint', args.model)
    np.savez('%s/adjoint_seed%d.npz' % (out, args.seed), shots=np.array(shots), freq_bands=np.array(FREQS),
             offset_bands=np.array(OFFSETS), early_window=EARLY,
             freqs=np.fft.rfftfreq(s['d_obs'].shape[0], protocol.DT), **results)
    print('saved in', out)


if __name__ == '__main__':
    main()
