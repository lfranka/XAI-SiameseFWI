"""Full-waveform inversion with any misfit of the article, under the unified protocol.

In each epoch the shots are visited once in a fixed order; for each shot the network (if any) is updated,
the velocity gradient is clipped to [-1e3, 1e3], the velocity model is updated with Adam, velocities are
clamped to positive values and the water layer (Marmousi2) is reset to 1500 m/s.

From a notebook:
    from utils.fwi import run
    metrics = run('marm', 'linear', 'siamesefit', seed=0)
From the command line (repository root):
    python -m utils.fwi --model marm --init linear --method siamesefit --seed 0

Outputs (in out): config.json, metrics.csv, model_final.npy and checkpoints of the velocity model and the
network weights every 10 epochs, used by the explainability analyses.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import optim

import deepwave
from utils import protocol
from utils.misfits import ABLATIONS, MAXPOOL_TESTS, METHODS, Misfit
from utils.utils import ComputeSSIM

# Adam step (m/s) and eps of the velocity model selected by the calibration of the article (supplementary Section S2, Table S3) for
# the linear initial model; the other initial models use the same settings, and the ablations and the tests of the
# max-pooling use those of SiameseFit
CALIBRATED = {
    ('over', 'l2'): (10, 1e-8), ('over', 'euclidean'): (20, 1e-8), ('over', 'siamesefwi'): (10, 1e-8),
    ('over', 'siamesefit'): (40, 1e-22), ('over', 'mpbae'): (20, 1e-8),
    ('marm', 'l2'): (10, 1e-8), ('marm', 'euclidean'): (10, 1e-8), ('marm', 'siamesefwi'): (10, 1e-8),
    ('marm', 'siamesefit'): (50, 1e-22), ('marm', 'mpbae'): (40, 1e-8),
}
DECAY_EVERY, DECAY_FACTOR = 100, 0.5  # the step is halved every 100 epochs
CKPT_EVERY = 10


def calibrated(model, init, method):
    """Calibrated (step, eps) of the velocity model (the same for every initial model); the ablations and the tests of
    the max-pooling use those of SiameseFit."""
    return CALIBRATED[(model, 'siamesefit' if method in ABLATIONS + MAXPOOL_TESTS else method)]


def run_dir(model, init, method, seed, test=None):
    """Default output folder of a run: results/<model>_<init>[_<test>]/<method>/seed<seed>, with test 'noise' or
    'lowcut' for the robustness tests."""
    case = '%s_%s' % (model, init) + ('_%s' % test if test else '')
    return os.path.join(protocol.ROOT, 'results', case, method, 'seed%d' % seed)


def test_name(noise_snr=None, lowcut=None):
    """Name of the robustness test of a run ('noise', 'lowcut' or None)."""
    return '_'.join(([] if noise_snr is None else ['noise']) + ([] if not lowcut else ['lowcut'])) or None


def model_metrics(m, model_true):
    """SNR (dB), SSIM and relative error of the velocity model, computed in float64."""
    rec = m.detach().cpu().numpy().astype(np.float64)
    true = model_true.detach().cpu().numpy().astype(np.float64)
    err = np.sum((rec - true) ** 2)
    snr = 10 * np.log10(np.sum(true ** 2) / err)
    ssim = float(ComputeSSIM(rec, true))
    return snr, ssim, np.sqrt(err / np.sum(true ** 2))


def save_checkpoint(out, epoch, m, misfit):
    """Store the velocity model and the network weights of one epoch."""
    state = {'epoch': epoch, 'model': m.detach().cpu()}
    if misfit.net is not None:
        state['net'] = {k: v.cpu() for k, v in misfit.net.state_dict().items()}
    torch.save(state, os.path.join(out, 'ckpt_epoch%04d.pt' % epoch))


def run(model, init, method, seed=0, lr=None, eps=None, epochs=None, decay_every=DECAY_EVERY,
        decay_factor=DECAY_FACTOR, noise_snr=None, lowcut=None, water_after_update=False, out=None, verbose=True,
        cudnn_benchmark=None):
    """Run one inversion and return its metrics per epoch.

    Args:
        model: 'over' or 'marm'; init: 'linear', 'smooth', 'smooth_top' or 'linear_top'
        method: a misfit of METHODS ('l2', 'euclidean', 'siamesefwi', 'siamesefit', 'mpbae'), of ABLATIONS
            ('identity', 'avgpool', 'randproj', 'maxpool', 'skip128', 'frozen', 'maxpool_l2') or of MAXPOOL_TESTS ('maxpool_shift', 'maxpool_top16', 'maxpool_abs',
            'maxpool_mute')
        seed: random seed (initialization of the network)
        lr, eps: Adam step (m/s) and eps of the velocity model (default: calibrated values)
        epochs: default 200 (Overthrust) or 300 (Marmousi2)
        decay_every, decay_factor: schedule of the step (decay_every = 0 keeps it constant)
        noise_snr, lowcut: robustness tests (noise SNR in dB per shot; high-pass corner in Hz)
        water_after_update: reset the water layer only after each update, as the official SiameseFit
            Marmousi2 notebook (supplementary Section S1)
        out: output folder (default: run_dir(model, init, method, seed, test))
        cudnn_benchmark: cuDNN autotuner (default: on for the Overthrust and off for the Marmousi2 model, as in the
            main runs of the article; the robustness tests were run with it off)

    Returns:
        dict of numpy arrays: epoch, loss, data_l2, snr, ssim, rel_error, lr, time_s
    """
    assert method in METHODS + ABLATIONS + MAXPOOL_TESTS, method
    cal_lr, cal_eps = calibrated(model, init, method)
    lr = cal_lr if lr is None else lr
    eps = cal_eps if eps is None else eps
    epochs = protocol.MODELS[model]['epochs'] if epochs is None else epochs

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    # cuDNN autotuner on as in the official notebooks, except for the Marmousi2 model, where its workspace
    # does not leave room for SiameseFit in 16 GB
    torch.backends.cudnn.benchmark = (model != 'marm') if cudnn_benchmark is None else cudnn_benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = out or run_dir(model, init, method, seed, test_name(noise_snr, lowcut))
    os.makedirs(out, exist_ok=True)

    # data, geometry and trainable velocity model
    s = protocol.setup(model, init, device, noise_snr, lowcut)
    m, model_true, d_obs = s['model'], s['model_true'], s['d_obs']
    water_rows = s['water_rows']
    if not water_after_update:
        m.data[:water_rows] = protocol.WATER_VELOCITY  # known water layer (no rows for the Overthrust)

    # misfit (network created after the data, as in the notebooks) and optimizer of the velocity model
    misfit = Misfit(method, device, model)
    optimizer = optim.Adam([{'params': m, 'lr': lr, 'betas': (0.5, 0.99), 'eps': eps, 'weight_decay': 0}])
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: decay_factor)

    config = dict(model=model, init=init, method=method, seed=seed, lr=lr, eps=eps, epochs=epochs,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  decay_every=decay_every, decay_factor=decay_factor, noise_snr=noise_snr, lowcut=lowcut,
                  water_after_update=water_after_update, water_rows=water_rows,
                  net_parameters=0 if misfit.net is None else sum(p.numel() for p in misfit.net.parameters()))
    json.dump(config, open(os.path.join(out, 'config.json'), 'w'), indent=1)

    header = 'epoch,loss,data_l2,snr,ssim,rel_error,lr,time_s'
    rows = []
    save_checkpoint(out, 0, m, misfit)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_l2 = 0.0, 0.0

        # one shot per batch: every epoch updates the model once per shot
        for it in range(protocol.NUM_SHOTS):
            optimizer.zero_grad()
            if misfit.net is not None:
                misfit.net.train()
                misfit.net.zero_grad()

            # forward modelling in the current model
            prop = deepwave.scalar.Propagator({'vp': m}, protocol.DX, protocol.PML_WIDTH, protocol.ORDER, None)
            d_sim = prop(s['source'], s['x_s'][it:it + 1], s['x_r'][it:it + 1], protocol.DT)
            d_shot = d_obs[:, it:it + 1]

            loss = misfit(d_shot, d_sim)
            epoch_loss += loss.item()
            epoch_l2 += torch.sum((d_sim.detach() - d_shot) ** 2).item()
            loss.backward()

            # update the network, then the velocity model
            if misfit.net_optimizer is not None:
                misfit.net_optimizer.step()
            torch.nn.utils.clip_grad_value_(m, 1e3)
            optimizer.step()
            m.data = torch.clamp(m.data, min=1e-12)
            m.data[:water_rows] = protocol.WATER_VELOCITY

        # schedule of the velocity step (same rule as the notebooks)
        if decay_every > 0 and epoch > 1 and (epoch - 1) % decay_every == 0:
            scheduler.step()

        snr, ssim, rel_error = model_metrics(m, model_true)
        rows.append([epoch, epoch_loss / protocol.NUM_SHOTS, epoch_l2 / protocol.NUM_SHOTS, snr, ssim, rel_error,
                     optimizer.param_groups[0]['lr'], time.time() - t_start])
        if verbose:
            print('Epoch: %d Loss: %.8e data_l2: %.6e SNR: %.4f SSIM: %.4f lr: %g'
                  % (epoch, rows[-1][1], rows[-1][2], snr, ssim, rows[-1][6]), flush=True)

        if epoch % CKPT_EVERY == 0 or epoch == epochs:
            save_checkpoint(out, epoch, m, misfit)
            np.savetxt(os.path.join(out, 'metrics.csv'), np.array(rows), delimiter=',', header=header, comments='')

    np.save(os.path.join(out, 'model_final.npy'), m.detach().cpu().numpy())
    if verbose:
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
        print('Running complete in %.1f min, peak GPU memory %.2f GiB' % ((time.time() - t_start) / 60, peak))
    return dict(zip(header.split(','), np.array(rows).T))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='over', choices=list(protocol.MODELS))
    p.add_argument('--init', default='linear', choices=list(protocol.INITS))
    p.add_argument('--method', required=True, choices=list(METHODS + ABLATIONS + MAXPOOL_TESTS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=None, help='Adam step of the velocity model (default: calibrated)')
    p.add_argument('--eps', type=float, default=None, help='Adam eps of the velocity model (default: calibrated)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--decay-every', type=int, default=DECAY_EVERY, help='epochs between halvings of the step (0: none)')
    p.add_argument('--noise-snr', type=float, default=None, help='SNR (dB) of white noise added to the observed data')
    p.add_argument('--lowcut', type=float, default=None, help='remove frequencies below this value (Hz)')
    p.add_argument('--water-after-update', action='store_true')
    p.add_argument('--no-cudnn-benchmark', action='store_true', help='cuDNN autotuner off (also on the Overthrust)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    run(a.model, a.init, a.method, a.seed, a.lr, a.eps, a.epochs, a.decay_every, DECAY_FACTOR, a.noise_snr, a.lowcut,
        a.water_after_update, a.out, cudnn_benchmark=False if a.no_cudnn_benchmark else None)


if __name__ == '__main__':
    main()
