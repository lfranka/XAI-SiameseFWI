"""SNR and SSIM of the final velocity models per depth band (0-1, 1-2 and below 2 km).

The SSIM map uses the window and constants of utils.utils.ComputeSSIM in double precision; its mean over the whole model
equals the SSIM reported by the inversions. Writes results/xai/depth/<model>_linear/depth_metrics.csv.

    python -m xai.depth --model over
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from xai import common
from utils import protocol
from utils.utils import create_window, loadtruemodel

BANDS_KM = (0.0, 1.0, 2.0, np.inf)
RUNS = ('l2', 'euclidean', 'siamesefwi', 'siamesefit', 'mpbae', 'maxpool', 'frozen', 'skip128')


def ssim_map(rec, true, window_size=11):
    """SSIM map of two 2D arrays, same definition as utils.utils.ComputeSSIM (L = 255)."""
    a = torch.from_numpy(rec).double().view(1, 1, *rec.shape)
    b = torch.from_numpy(true).double().view(1, 1, *true.shape)
    w = create_window(window_size, 1).double()
    pad = window_size // 2
    mu1, mu2 = F.conv2d(a, w, padding=pad), F.conv2d(b, w, padding=pad)
    s11 = F.conv2d(a * a, w, padding=pad) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=pad) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=pad) - mu1 * mu2
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m[0, 0].numpy()


def snr(rec, true):
    return 10 * np.log10(np.sum(true ** 2) / np.sum((rec - true) ** 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    args = p.parse_args()
    cfg = protocol.MODELS[args.model]
    nz, ny = cfg['shape']
    true = loadtruemodel(os.path.join(protocol.ROOT, 'data', cfg['true']), 2, np.array([nz, ny])).numpy().astype(np.float64)
    depth = (np.arange(nz) + 0.5) * protocol.DX / 1000
    bands = [(lo, hi, (depth >= lo) & (depth < hi)) for lo, hi in zip(BANDS_KM[:-1], BANDS_KM[1:])]
    init = protocol.initial_model(args.model, 'linear').numpy().astype(np.float64)
    init[:cfg['water_rows']] = protocol.WATER_VELOCITY
    rows = []
    for name, seed, rec in [('initial', '-', init)] + [
            (method, os.path.basename(r), np.load(os.path.join(r, 'model_final.npy')).astype(np.float64))
            for method in RUNS for r in common.finished(args.model, method)]:
        smap = ssim_map(rec, true)
        for lo, hi, mask in bands:
            rows.append([name, seed, '%g-%g' % (lo, hi), snr(rec[mask], true[mask]), smap[mask].mean()])
        rows.append([name, seed, 'all', snr(rec, true), smap.mean()])
    d = common.out_dir('depth', args.model)
    with open(os.path.join(d, 'depth_metrics.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'seed', 'depth_km', 'snr_db', 'ssim'])
        w.writerows(rows)
    for name in ['initial'] + list(RUNS):
        v = [r for r in rows if r[0] == name]
        if v:
            print('%-12s' % name + ''.join('  %s km %.2f' % (b, np.mean([r[3] for r in v if r[2] == b]))
                                           for b in ('0-1', '1-2', '2-inf')))
    print('saved in', d)


if __name__ == '__main__':
    main()
