"""Plots of the inversions: velocity models and convergence curves."""
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import protocol
from utils.utils import loadtruemodel

NAMES = {'l2': '$L_2$', 'euclidean': 'Euclidean', 'siamesefwi': 'SiameseFWI', 'siamesefit': 'SiameseFit',
         'mpbae': 'MPBAE', 'identity': 'No encoder', 'avgpool': 'Average pooling',
         'randproj': 'Random projection', 'maxpool': 'Max-pooling', 'frozen': 'Frozen network',
         'maxpool_l2': 'Max-pooling, $L_2$ loss', 'skip128': 'Skip branch, 128 filters',
         'maxpool_shift': 'Max-pooling, shifted grid', 'maxpool_top16': 'Mean of the 16 largest',
         'maxpool_abs': 'Max-pooling of |x|', 'maxpool_mute': 'Max-pooling, early arrivals muted'}
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'lines.linewidth': 1.5})


def true_and_initial(model, init):
    """True and initial velocity models (m/s) as numpy arrays, with the water layer of the inversions."""
    cfg = protocol.MODELS[model]
    data = os.path.join(protocol.ROOT, 'data')
    true = loadtruemodel(os.path.join(data, cfg['true']), 2, np.array(cfg['shape'])).numpy()
    initial = protocol.initial_model(model, init).numpy().copy()
    initial[:cfg['water_rows']] = protocol.WATER_VELOCITY
    return true, initial


def plot_models(model, init, inverted, title):
    """True, initial and inverted velocity models with the same colour scale (km/s)."""
    true, initial = true_and_initial(model, init)
    nz, ny = true.shape
    extent = [0, ny * protocol.DX / 1000, nz * protocol.DX / 1000, 0]
    vmin, vmax = true.min() / 1000, true.max() / 1000
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for k, (ax, v, name) in enumerate(zip(axes, (true, initial, inverted),
                                          ('True model', 'Initial model (%s)' % init, title))):
        im = ax.imshow(v / 1000, cmap='turbo', vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
        ax.set_title(name)
        ax.set_ylabel('Depth (km)')
        ax.text(-0.09, 1.02, '(%s)' % 'abc'[k], transform=ax.transAxes, fontweight='bold')
    axes[-1].set_xlabel('Distance (km)')
    fig.colorbar(im, ax=axes, label='Velocity (km/s)', shrink=0.8)
    return fig


def plot_curves(metrics, title):
    """SNR of the velocity model and misfit along the inversion."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics['epoch'], metrics['snr'])
    axes[0].set_ylabel('SNR (dB)')
    axes[1].semilogy(metrics['epoch'], metrics['loss'] / metrics['loss'][0])
    axes[1].set_ylabel('Misfit / first epoch')
    for k, ax in enumerate(axes):
        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.text(-0.15, 1.02, '(%s)' % 'ab'[k], transform=ax.transAxes, fontweight='bold')
    fig.tight_layout()
    return fig
