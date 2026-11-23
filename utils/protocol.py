"""Acquisition and modelling protocol shared by all experiments.

Models are selected with the keys 'over' (Overthrust) and 'marm' (Marmousi2), and initial models with
'linear' (velocity increasing linearly with depth, the initial model of the article), 'smooth' (smoothed true model
distributed with the reference implementations) and the hybrids 'smooth_top' (smooth above 2 km, linear below) and
'linear_top' (the reverse), used to test what limits the deepest band of the Marmousi2 model (Section IV-B1).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.utils import createSR, createdata, loadinitmodel, loadtruemodel, ricker, seismic_filter  # noqa: E402

# fixed modelling parameters (same values as the official notebooks)
DX = 30.0                       # grid spacing (m)
DT = 0.003                      # time sampling (s)
TOTAL_T = 6.0                   # recording time (s)
NT = int(TOTAL_T / DT)          # number of time samples
NUM_SHOTS = 30
ORDER = 8                       # finite-difference order in space
PML_WIDTH = [0, 10, 10, 10, 0, 0]  # free surface on top, absorbing elsewhere
PEAK_FREQ = 5.0                 # Ricker source (Hz)

# velocity models: shape (nz, ny), files of the true and initial models, number of epochs,
# water_rows (top rows kept at WATER_VELOCITY during the inversion, as in the official SiameseFit
# notebooks for the Marmousi2) and window of MPBAE (q + 1 for q successive 2 x 2 max-poolings)
MODELS = {
    'over': {'name': 'Overthrust', 'shape': (94, 400), 'true': 'over_94_400.bin',
             'smooth': 'over_initmodel.mat', 'linear': 'over_initmodel_FLAT.mat',
             'epochs': 200, 'water_rows': 0, 'mpbae_window': 11},
    'marm': {'name': 'Marmousi2', 'shape': (117, 567), 'true': 'mar_big_117_567.bin',
             'smooth': 'mar_big_initmodel.mat', 'linear': 'Marm_initmodel_FLAT.mat',
             'epochs': 300, 'water_rows': 16, 'mpbae_window': 19},
}
INITS = ('linear', 'smooth', 'smooth_top', 'linear_top')
HYBRID_DEPTH, HYBRID_TAPER = 2000.0, 300.0  # depth of the transition of the hybrid models and its width (m)
WATER_VELOCITY = 1500.0  # m/s
LOWCUT_CORNERS = 6       # order of the Butterworth high-pass filter of the source


def make_source(device, lowcut=None):
    """Ricker wavelet [nt, 1, 1] with the peak at 1/PEAK_FREQ seconds.

    With lowcut (Hz), the wavelet is high-pass filtered (causal Butterworth, as in utils), so that
    observed and simulated data both lack the frequencies below lowcut.
    """
    wavelet = ricker(PEAK_FREQ, NT, DT, 1.0 / PEAK_FREQ)
    if lowcut:
        wavelet = seismic_filter(wavelet, 'highpass', lowcut, None, 1.0 / DT, LOWCUT_CORNERS)
    return torch.from_numpy(np.float32(wavelet)).reshape(-1, 1, 1).to(device)


def add_noise(d, snr_db, seed=0):
    """Add white Gaussian noise to each shot of d [nt, shots, nrec] with the given SNR (dB).

    The noise power of each shot is its mean power divided by 10^(snr/10). The realization depends
    only on seed, so every misfit and run sees the same noisy data.
    """
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(d.shape, generator=gen, dtype=torch.float32).to(d.device)
    power = d.pow(2).mean(dim=(0, 2), keepdim=True)
    return d + noise * torch.sqrt(power / 10 ** (snr_db / 10))


def initial_model(model, init):
    """Initial velocity model (CPU tensor, without the water layer of the inversions)."""
    cfg = MODELS[model]
    data_dir = os.path.join(ROOT, 'data')
    if init in ('linear', 'smooth'):
        return loadinitmodel(os.path.join(data_dir, cfg[init]), 'cpu')[1]
    smooth = loadinitmodel(os.path.join(data_dir, cfg['smooth']), 'cpu')[1]
    linear = loadinitmodel(os.path.join(data_dir, cfg['linear']), 'cpu')[1]
    depth = torch.arange(smooth.shape[0], dtype=torch.float32).view(-1, 1) * DX
    top = torch.clamp((HYBRID_DEPTH + HYBRID_TAPER / 2 - depth) / HYBRID_TAPER, 0.0, 1.0)  # 1 above, 0 below
    upper, lower = (smooth, linear) if init == 'smooth_top' else (linear, smooth)
    return top * upper + (1 - top) * lower


def setup(model, init, device, noise_snr=None, lowcut=None, noise_seed=0):
    """Build everything an inversion needs.

    Args:
        model: 'over' or 'marm'
        init: initial model, one of INITS
        noise_snr: if given, white Gaussian noise with this SNR (dB) is added to the observed data
        lowcut: if given, frequencies below it (Hz) are removed from the source, hence from all data
        noise_seed: seed of the noise realization

    Returns:
        dict with the true model, trainable model, source, geometry and observed data
    """
    cfg = MODELS[model]
    nz, ny = cfg['shape']
    data_dir = os.path.join(ROOT, 'data')

    model_true = loadtruemodel(os.path.join(data_dir, cfg['true']), 2, np.array([nz, ny])).to(device)
    model_init = initial_model(model, init)
    m = torch.nn.Parameter(model_init.clone().to(device))

    # sources and receivers evenly spread on the surface, one receiver per grid point
    source_spacing = np.floor(DX * ny / (NUM_SHOTS + 1))
    receiver_spacing = np.floor(DX * ny / (ny + 1))
    x_s, x_r = createSR(NUM_SHOTS, 1, ny, 2, source_spacing, receiver_spacing, 0, 0)
    x_s, x_r = x_s.to(device), x_r.to(device)

    # observed data modelled in the true model
    source = make_source(device, lowcut)
    d_obs = createdata(model_true, DX, source.repeat(1, NUM_SHOTS, 1), x_s, x_r, DT,
                       PML_WIDTH, ORDER, None, device).to(device)
    if noise_snr is not None:
        d_obs = add_noise(d_obs, noise_snr, noise_seed)

    return {'model_true': model_true, 'model': m, 'model_init': model_init, 'source': source,
            'x_s': x_s, 'x_r': x_r, 'd_obs': d_obs, 'shape': (nz, ny),
            'water_rows': cfg['water_rows']}
