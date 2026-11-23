"""Figures and tables of the article, built from the saved runs and analyses.

Every figure takes the list of models; a model whose results are not available is skipped. Figures are written to
results/figures (PDF) and tables to results/tables (LaTeX), with the file names used in the article.

    python -m utils.figures                       # everything
    python -m utils.figures --only performance results
"""
import argparse
import csv
import functools
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deepwave  # noqa: E402
from utils import protocol  # noqa: E402
from utils.fwi import model_metrics  # noqa: E402
from utils.misfits import Misfit, to_image  # noqa: E402
from utils.utils import loadtruemodel  # noqa: E402
from xai.cka import linear_cka  # noqa: E402

ROOT = protocol.ROOT
FIG_DIR = os.path.join(ROOT, 'results', 'figures')
TAB_DIR = os.path.join(ROOT, 'results', 'tables')
KEY = {'overthrust': 'over', 'marmousi2': 'marm'}  # names used in the article files -> keys of utils.protocol
WIDTH = 6.85  # full text width of GJI (inches)
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
NAMES = {'overthrust': 'Overthrust', 'marmousi2': 'Marmousi2'}
SHOT = 15  # middle shot of the adjoint analysis

# Colour-blind safe palette (Okabe-Ito). Dashed: misfits that compare the waveforms; solid: max-based.
STYLE = {
    'l2': (r'$L_2$', '#000000', '--'),
    'euclidean': ('Euclidean', '#999999', '--'),
    'siamesefwi': ('SiameseFWI', '#0072B2', '--'),
    'siamesefit': ('SiameseFit', '#D55E00', '-'),
    'frozen': ('SiameseFit (frozen)', '#E69F00', '-'),
    'maxpool': ('Max-pool 16/16', '#CC79A7', '-'),
    'mpbae': ('MPBAE', '#009E73', '-'),
    'skip128': ('Skip branch, 128 filters', '#56B4E9', '-'),
}
MAIN = ['l2', 'euclidean', 'siamesefwi', 'siamesefit', 'mpbae']  # F-SiameseFWI left out of the article
WHERE = {m: 'A2' for m in MAIN}
WHERE.update({'frozen': 'ablations', 'maxpool': 'ablations'})
BRANCH = 'skip128'  # skip branch shown with the main ablations (curves, depth bands, run-away cells)
LADDER = [  # (group title, [(results subdirectory, method directory, label), ...])
    ('Encoder (SiameseFit loss and optimizer)', [
        ('A2', 'euclidean', 'Euclidean'),
        ('ablations', 'identity', 'No encoder'),
        ('ablations', 'avgpool', 'Average pool 16/16'),
        ('ablations', 'randproj', 'Random projection'),
        ('ablations', 'maxpool', 'Max-pool 16/16'),
        ('ablations', 'skip128', 'Skip branch, 128 filters'),
        ('ablations', 'frozen', 'SiameseFit, frozen'),
        ('A2', 'siamesefit', 'SiameseFit, trained')]),
    ('Loss form and window (max-pooling)', [
        ('ablations', 'maxpool', 'Max-pool 16/16, SiameseFit loss'),
        ('ablations', 'maxpool_l2', 'Max-pool 16/16, $L_2$ loss'),
        ('A2', 'mpbae', 'MPBAE (own window and step)')]),
    ('Tests of the max-pooling', [
        ('ablations', 'maxpool_lr20', r'Max-pool 16/16, $\alpha=20$'),
        ('ablations', 'maxpool_shift', 'Max-pool, shifted grid'),
        ('ablations', 'maxpool_top16', 'Mean of the 16 largest'),
        ('ablations', 'maxpool_abs', 'Max-pool of $|x|$'),
        ('ablations', 'maxpool_mute', 'Max-pool, early arrivals muted')]),
]
BANDS = {'0-1': '0-1 km', '1-2': '1-2 km', '2-inf': '> 2 km'}


# ----------------------------------------------------------------------------- helpers

def set_style():
    """Matplotlib settings shared by all figures (sizes are final print sizes)."""
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.5, 'lines.linewidth': 1.5, 'axes.titlelocation': 'left',
        'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'savefig.dpi': 300, 'pdf.fonttype': 42})


def title(ax, k, text):
    """Panel label (a), (b), ... followed by the panel title."""
    ax.set_title('(%s) %s' % (LETTERS[k], text), loc='left')


def save(fig, name):
    """Save one figure in results/figures as PDF (as in the article) and PNG (for the notebooks)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, name + '.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('saved', name)


def case(model, init='linear'):
    """Name of the results subfolder of one model and initial model (utils.fwi.run_dir)."""
    return '%s_%s' % (KEY[model], init)


def runs(sub, model, method, init='linear'):
    """Seed folders of the finished runs of one misfit (sub, the group of the run in the research layout, is kept in
    the calls for readability and is not part of the path)."""
    paths = sorted(glob.glob(os.path.join(ROOT, 'results', case(model, init), method, 'seed*')))
    return [p for p in paths if os.path.exists(os.path.join(p, 'model_final.npy'))]


def metrics(path):
    """Metrics of one run: epochs x (epoch, loss, data_l2, snr, ssim, rel_error, lr, time)."""
    return np.loadtxt(os.path.join(path, 'metrics.csv'), delimiter=',', skiprows=1)


def npz(*parts):
    """Load an analysis file if it exists, else None."""
    path = os.path.join(ROOT, 'results', 'xai', *parts)
    return np.load(path) if os.path.exists(path) else None


XAI_SEEDS = (0, 1, 2)  # seeds of the explainability analyses; the images of gathers and adjoint sources are seed 0


def npz_seeds(analysis, model, fmt):
    """Analysis files of the XAI seeds that exist (fmt contains %d for the seed)."""
    found = [npz(analysis, case(model), fmt % sd) for sd in XAI_SEEDS]
    return [d for d in found if d is not None]


def band_plot(ax, x, curves, color, ls='-', label=None, marker=None):
    """Mean of the curves of the seeds (line) and their range (shaded)."""
    v = np.array(curves, dtype=float)
    ax.plot(x, v.mean(0), color=color, ls=ls, label=label, marker=marker, ms=3)
    if len(v) > 1:
        ax.fill_between(x, v.min(0), v.max(0), color=color, alpha=0.15, lw=0)


def true_model(model):
    """True velocity model (nz x ny)."""
    nz, ny = protocol.MODELS[KEY[model]]['shape']
    m = loadtruemodel(os.path.join(ROOT, 'data', protocol.MODELS[KEY[model]]['true']), 2, np.array([nz, ny]))
    return m.numpy() if torch.is_tensor(m) else m


def initial_model(model, init='linear'):
    """Initial velocity model with the water layer fixed, as in the inversions."""
    m = protocol.initial_model(KEY[model], init).clone()
    m[:protocol.MODELS[KEY[model]]['water_rows']] = protocol.WATER_VELOCITY
    return m


@functools.lru_cache(maxsize=None)
def initial_metrics(model, init='linear'):
    """SNR (dB) and SSIM of the initial model."""
    snr, ssim, _ = model_metrics(initial_model(model, init), torch.from_numpy(true_model(model)))
    return snr, ssim


@functools.lru_cache(maxsize=None)
def observed(model):
    """Observed data of all shots (nt x shots x receivers), computed on the CPU."""
    return protocol.setup(KEY[model], 'linear', torch.device('cpu'))


def initial_residual(model, init='linear'):
    """Mean over shots of the squared data residual of the initial model (cached in a json file)."""
    cache = os.path.join(FIG_DIR, 'initial_residual.json')
    store = json.load(open(cache)) if os.path.exists(cache) else {}
    key = case(model, init)
    if key not in store:
        s = observed(model)
        m = initial_model(model, init)
        total = 0.0
        with torch.no_grad():
            for it in range(protocol.NUM_SHOTS):
                prop = deepwave.scalar.Propagator({'vp': m}, protocol.DX, protocol.PML_WIDTH, protocol.ORDER, None)
                d = prop(s['source'], s['x_s'][it:it + 1], s['x_r'][it:it + 1], protocol.DT)
                total += torch.sum((d - s['d_obs'][:, it:it + 1]) ** 2).item()
        store[key] = total / protocol.NUM_SHOTS
        os.makedirs(FIG_DIR, exist_ok=True)
        json.dump(store, open(cache, 'w'), indent=1)
    return store[key]


def median_run(paths):
    """Seed folder with the median final SNR."""
    snr = [metrics(p)[-1, 3] for p in paths]
    return paths[int(np.argsort(snr)[len(snr) // 2])]


def extent_model(model):
    """imshow extent of a velocity model in km."""
    nz, ny = protocol.MODELS[KEY[model]]['shape']
    return [0, ny * protocol.DX / 1000, nz * protocol.DX / 1000, 0]


def extent_gather(g):
    """imshow extent of a shot gather (receiver position in km, time in s)."""
    return [0, g.shape[1] * protocol.DX / 1000, g.shape[0] * protocol.DT, 0]


def show_gather(ax, g):
    """Gather clipped at its own 99th percentile of the absolute value."""
    clip = np.percentile(np.abs(g), 99)
    ax.imshow(g, cmap='seismic', vmin=-clip, vmax=clip, aspect='auto', extent=extent_gather(g))


def depth_rows(model):
    """Rows of the depth-band metrics file, or [] if not computed."""
    path = os.path.join(ROOT, 'results', 'xai', 'depth', case(model), 'depth_metrics.csv')
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def available(models):
    """Models with finished main-regime runs of every method."""
    return [m for m in models if all(runs('A2', m, k) for k in MAIN)]


# ----------------------------------------------------------------------------- figures

def fig_models(models):
    """True, linear initial and smooth initial (control regime) models: one column per model with its
    own colour scale below the column."""
    rows = [('True model', None), ('Linear initial model', 'linear')]
    fig, ax = plt.subplots(len(rows), len(models), figsize=(WIDTH if len(models) > 1 else WIDTH / 2, 0.8 + 1.2 * len(rows)),
                           squeeze=False, constrained_layout=True)
    k = 0
    for i, (name, init) in enumerate(rows):
        for j, model in enumerate(models):
            a = ax[i, j]
            true = true_model(model)
            m = true if init is None else initial_model(model, init).numpy()
            a.imshow(m, cmap='turbo', vmin=true.min(), vmax=true.max(), extent=extent_model(model))
            title(a, k, name)
            k += 1
            if j == 0:  # one depth label per row
                a.set_ylabel('Depth (km)')
            a.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    for j, model in enumerate(models):
        ax[0, j].set_title(NAMES[model], loc='center', y=1.22, fontweight='bold')
        ax[-1, j].set_xlabel('Distance (km)')
        cb = fig.colorbar(ax[-1, j].images[0], ax=ax[:, j], orientation='horizontal', fraction=0.05, aspect=30)
        cb.set_label('%s velocity (m s$^{-1}$)' % NAMES[model])
    save(fig, 'fig_models')


def fig_performance(models):
    """One figure per model: linear initial model and inverted models of the seed with the median final
    SNR (two columns), SNR along the inversion (mean and range of seeds) and SNR per depth band."""
    ncol = 2
    nrow = int(np.ceil((len(MAIN) + 1) / ncol))
    for model in models:
        true = true_model(model)
        fig = plt.figure(figsize=(WIDTH, 7.4), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[4.3, 2.9])
        top = outer[0].subgridspec(nrow, ncol)
        items = [('Linear initial model', initial_model(model).numpy())]
        items += [(STYLE[method][0], np.load(os.path.join(median_run(runs('A2', model, method)), 'model_final.npy')))
                  for method in MAIN]
        maps = []
        for j, (name, m) in enumerate(items):
            a = fig.add_subplot(top[j // ncol, j % ncol])
            im = a.imshow(m, cmap='turbo', vmin=true.min(), vmax=true.max(), extent=extent_model(model))
            title(a, j, name)
            maps.append(a)
            if j % ncol == 0:
                a.set_ylabel('Depth (km)')
            if j // ncol == nrow - 1:
                a.set_xlabel('Distance (km)')
        cb = fig.colorbar(im, ax=maps, orientation='horizontal', fraction=0.05, aspect=40)
        cb.set_label('Velocity (m s$^{-1}$)')
        k = len(items)
        bottom = outer[1].subgridspec(1, 3, width_ratios=[1.15, 1, 1])
        a = fig.add_subplot(bottom[0, 0])
        for method in MAIN:
            snr_band(a, runs('A2', model, method), STYLE[method][1], STYLE[method][2], STYLE[method][0])
        a.axhline(initial_metrics(model)[0], color='0.4', ls=':', lw=1.0, label='Initial model')
        a.set_xlabel('Epoch')
        a.set_ylabel('SNR (dB)')
        title(a, k, 'SNR along the inversion')
        handles, labels = a.get_legend_handles_labels()
        a = fig.add_subplot(bottom[0, 1])
        if depth_rows(model):
            bars_by_depth(a, model, MAIN)
            a.set_ylabel('')  # one SNR label for the row
            title(a, k + 1, 'SNR per depth band')
        else:
            a.axis('off')
        a = fig.add_subplot(bottom[0, 2])
        band_curves(a, model, runs('A2', model, 'siamesefit'))
        a.set_ylabel('')
        title(a, k + 2, 'SiameseFit per depth band')
        handles.append(matplotlib.patches.Patch(facecolor='white', edgecolor='k', hatch='///'))
        labels.append('Initial model (bars)')
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
        save(fig, 'fig_performance_%s' % model)


def band_masks(model):
    """Row masks of the depth bands, as in xai.depth (cell centres; the Marmousi2 water layer
    is in the first band)."""
    nz = protocol.MODELS[KEY[model]]['shape'][0]
    depth = (np.arange(nz) + 0.5) * protocol.DX / 1000
    return [(depth >= lo) & (depth < hi) for lo, hi in ((0, 1), (1, 2), (2, np.inf))]


def band_snr(m, true, mask):
    return 10 * np.log10(np.sum(true[mask] ** 2) / np.sum((m[mask] - true[mask]) ** 2))


def band_curves(ax, model, paths):
    """SNR of each depth band at the stored checkpoints (every 10 epochs; epoch 0 is the initial model): mean of the
    runs (line) and range (shaded)."""
    true = true_model(model).astype(np.float64)
    for mask, (band, ls) in zip(band_masks(model), zip(BANDS.values(), ('-', '--', ':'))):
        curves = []
        for p in paths:
            ck = sorted(glob.glob(os.path.join(p, 'ckpt_epoch*.pt')))
            curves.append([band_snr(torch.load(c, map_location='cpu')['model'].numpy().astype(np.float64), true, mask)
                           for c in ck])
        epochs = [int(os.path.basename(c)[10:14]) for c in ck]
        v = np.array(curves)
        ax.fill_between(epochs, v.min(0), v.max(0), color=STYLE['siamesefit'][1], alpha=0.15, lw=0)
        ax.plot(epochs, v.mean(0), color=STYLE['siamesefit'][1], ls=ls, label=band)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SNR (dB)')
    ax.legend(loc='best', frameon=False)


def bars_by_depth(ax, model, methods):
    """Grouped bars of the SNR per depth band (mean and standard deviation over seeds)."""
    rows = depth_rows(model)
    width = 0.8 / (len(methods) + 1)
    for b, band in enumerate(BANDS):
        init = [float(r['snr_db']) for r in rows if r['method'] == 'initial' and r['depth_km'] == band]
        ax.bar(b - 0.4 + width / 2, init[0], width, color='white', edgecolor='k', hatch='///', lw=0.8,
               label='Initial model (bars)' if b == 0 else None)
        for k, method in enumerate(methods):
            v = [float(r['snr_db']) for r in rows if r['method'] == method and r['depth_km'] == band]
            if not v:  # runs not finished yet
                continue
            label, color, _ = STYLE[method]
            ax.bar(b - 0.4 + width * (k + 1.5), np.mean(v), width, yerr=std(v), color=color, ecolor='k',
                   error_kw={'lw': 0.8}, label=label if b == 0 else None)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(list(BANDS.values()))
    ax.set_xlabel('Depth band')
    ax.set_ylabel('SNR (dB)')


def snr_band(ax, paths, color, ls, label):
    """Mean SNR of several seeds along the inversion (line) and their range (shaded)."""
    d = np.stack([metrics(p) for p in paths])
    ax.fill_between(d[0, :, 0], d[:, :, 3].min(0), d[:, :, 3].max(0), color=color, alpha=0.25, lw=0)
    ax.plot(d[0, :, 0], d[:, :, 3].mean(0), color=color, ls=ls, label=label)


def fig_ablations(models):
    """One figure per model: ablation ladder (final SNR), SNR along the inversion of the max-pooling,
    frozen and trained encoders against the Euclidean misfit, and their SNR per depth band."""
    for model in models:
        if not runs('ablations', model, 'maxpool'):
            continue
        finals = {(sub, method): [metrics(p)[-1, 3] for p in runs(sub, model, method)]
                  for _, items in LADDER for sub, method, _ in items}
        groups = [(g, [it for it in items if finals[it[:2]]]) for g, items in LADDER]
        groups = [(g, items) for g, items in groups if items]
        n_rows = sum(1 + len(items) for _, items in groups)
        fig = plt.figure(figsize=(WIDTH, max(4.4, 0.8 + 0.2 * n_rows)), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
        a = fig.add_subplot(grid[:, 0])
        low = min([10] + [np.floor(np.mean(v) - std(v)) - 1 for v in finals.values() if v])  # left end of the bars
        names, y, row = [], [], 0
        for group, items in groups:  # a title row, then one bar per finished method (from the top)
            a.text(low + 0.2, -row, group, fontsize=7.5, fontstyle='italic', va='center')
            row += 1
            for sub, method, label in items:
                v = finals[(sub, method)]
                if v:
                    color = STYLE[method][1] if method in STYLE else '0.6'
                    a.barh(-row, np.mean(v), xerr=std(v), color=color, ecolor='k', error_kw={'lw': 0.8})
                    names.append('%s (%d)' % (label, len(v)))
                    y.append(-row)
                    row += 1
            if group != groups[-1][0]:
                a.axhline(-row + 0.5, color='0.4', lw=0.8)
        a.set_ylim(-row + 0.4, 0.7)
        a.axvline(initial_metrics(model)[0], color='k', ls=':', lw=1.0)
        a.set_yticks(y)
        a.set_yticklabels(names)
        a.set_xlabel('Final SNR (dB)')
        a.set_xlim(low, None)
        title(a, 0, 'Ablation ladder')
        a = fig.add_subplot(grid[0, 1])
        for sub, method in (('A2', 'euclidean'), ('ablations', 'maxpool'), ('ablations', BRANCH),
                            ('ablations', 'frozen'), ('A2', 'siamesefit')):
            if runs(sub, model, method):
                snr_band(a, runs(sub, model, method), STYLE[method][1], STYLE[method][2], STYLE[method][0])
        a.axhline(initial_metrics(model)[0], color='0.4', ls=':', lw=1.0, label='Initial model')
        a.set_xlabel('Epoch')
        a.set_ylabel('SNR (dB)')
        title(a, 1, 'SNR along the inversion')
        handles, labels = a.get_legend_handles_labels()
        a = fig.add_subplot(grid[1, 1])
        if depth_rows(model):
            methods = [m for m in ('maxpool', BRANCH, 'frozen', 'siamesefit')
                       if any(r['method'] == m for r in depth_rows(model))]
            bars_by_depth(a, model, methods)
            v = [float(r['snr_db']) for r in depth_rows(model) if r['method'] in methods]
            a.set_ylim(min(10, np.floor(min(v)) - 1), None)
            title(a, 2, 'SNR per depth band')
        handles.append(matplotlib.patches.Patch(facecolor='white', edgecolor='k', hatch='///'))
        labels.append('Initial model (bars)')
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.62, 0.0))
        save(fig, 'fig_ablations_%s' % model)


def fig_step_schedule(models):
    """Euclidean misfit with the step schedule of the protocol and with the constant step of the
    SiameseFWI implementation (Appendix on the calibration)."""
    for model in models:
        constant = runs('ablations', model, 'euclidean_lr80_nodecay')
        if not constant:
            continue
        fig, a = plt.subplots(figsize=(0.55 * WIDTH, 2.6), constrained_layout=True)
        snr_band(a, runs('A2', model, 'euclidean'), STYLE['euclidean'][1], '--', 'Protocol: 20 m s$^{-1}$, halved every 100 epochs')
        snr_band(a, constant, '0.2', '-', 'Constant step of 80 m s$^{-1}$')
        a.axhline(initial_metrics(model)[0], color='0.4', ls=':', lw=1.0, label='Initial model')
        a.set_xlabel('Epoch')
        a.set_ylabel('SNR (dB)')
        a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), frameon=False)
        save(fig, 'fig_step_schedule_%s' % model)


def best_setting(folder):
    """(step, eps, final SNR) of the complete calibration run with the highest final SNR in folder/lr*_eps*/seed0."""
    best = None
    for run in glob.glob(os.path.join(folder, 'lr*_eps*', 'seed0')):
        m = np.loadtxt(os.path.join(run, 'metrics.csv'), delimiter=',', skiprows=1, ndmin=2)
        if len(m) == json.load(open(os.path.join(run, 'config.json')))['epochs']:
            lr, eps = os.path.basename(os.path.dirname(run))[2:].split('_eps')
            if best is None or m[-1, 3] > best[2]:
                best = (lr, eps, m[-1, 3])
    return best


def calibration_runs(model, init, method):
    """Complete calibration runs of one misfit at its selected eps: sorted list of (step, final SNR)."""
    folder = os.path.join(ROOT, 'results', 'calibration', case(model, init), method)
    if not os.path.isdir(folder):
        return []
    eps = best_setting(folder)[1]
    out = []
    for run in glob.glob(os.path.join(folder, 'lr*_eps%s' % eps, 'seed0')):
        m = metrics(run)
        if len(m) == json.load(open(os.path.join(run, 'config.json')))['epochs']:
            out.append((float(os.path.basename(os.path.dirname(run))[2:].split('_eps')[0]), m[-1, 3]))
    return sorted(out)


def fig_calibration(models):
    """Calibration of the velocity step: final SNR of seed 0 against the initial step, for each misfit at its
    selected eps, in the main and control regimes (Appendix on the calibration). Filled marker: selected step."""
    steps = [10, 20, 40, 50, 80]
    regimes = [('linear', 'linear')]
    fig, ax = plt.subplots(len(regimes), len(models), figsize=(0.5 * WIDTH * len(models), 0.3 + 2.3 * len(regimes)),
                           squeeze=False, sharex=True, constrained_layout=True)
    for j, model in enumerate(models):
        for i, (init, regime) in enumerate(regimes):
            a = ax[i, j]
            for method in MAIN:
                pts = calibration_runs(model, init, method)
                if not pts:
                    continue
                label, color, ls = STYLE[method]
                x = [steps.index(int(s)) for s, _ in pts]
                y = [v for _, v in pts]
                a.plot(x, y, color=color, ls=ls, marker='o', ms=4, mfc='w', label=label)
                k = int(np.argmax(y))
                a.plot(x[k], y[k], color=color, marker='o', ms=4, ls='none')
            a.axhline(initial_metrics(model, init)[0], color='0.4', ls=':', lw=1.0, label='Initial model')
            a.set_xticks(range(len(steps)))
            a.set_xticklabels(steps)
            a.set_ylabel('Final SNR (dB)')
            title(a, len(models) * i + j, '%s, %s initial model' % (NAMES[model], regime))
        ax[-1, j].set_xlabel('Initial step (m s$^{-1}$)')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, 'fig_calibration')


LANDSCAPE = [  # (label in the landscape file, label in the adjoint file, legend, colour, line style); @-1: last epoch
    ('l2', 'l2') + STYLE['l2'],
    ('euclidean', 'euclidean') + STYLE['euclidean'],
    ('siamesefwi@-1', 'siamesefwi', 'SiameseFWI (trained)') + STYLE['siamesefwi'][1:],
    ('maxpool', 'maxpool') + STYLE['maxpool'],
    ('mpbae', 'mpbae') + STYLE['mpbae'],
    ('siamesefit@0', 'frozen', 'SiameseFit (random/frozen)') + STYLE['frozen'][1:],
    ('siamesefit@-1', 'siamesefit', 'SiameseFit (trained)') + STYLE['siamesefit'][1:],
]
OCCLUDED = ['l2', 'euclidean', 'siamesefwi', 'maxpool', 'mpbae', 'siamesefit']  # bars of fig_alignment


def network_epochs(labels, method):
    """Epochs of the networks of one method stored in a landscape file (labels 'method@epoch')."""
    return sorted(int(l.split('@')[1]) for l in labels if l.startswith(method + '@'))


def resolve(labels, label):
    """Convexity label with '@-1' replaced by the last stored epoch of that network."""
    if not label.endswith('@-1'):
        return label
    method = label.split('@')[0]
    return '%s@%d' % (method, network_epochs(labels, method)[-1])


def fig_landscape(models):
    """One figure per model: misfit along a time shift (waveform and max-based misfits in separate
    panels) and along the model line, legend below."""
    for model in models:
        cs = npz_seeds('landscape', model, 'landscape_seed%d.npz')
        if not cs:
            continue
        c = cs[0]
        fig, (a_wave, a_max, a_line) = plt.subplots(1, 3, figsize=(WIDTH, 2.5), constrained_layout=True)
        for conv, _, legend, color, ls in LANDSCAPE:
            shift, line = [], []
            for d in cs:  # mean and range over the seeds (the same curve for the misfits without a network)
                labels = list(d['labels'])
                k = labels.index(resolve(labels, conv))
                shift.append(d['j_shift'][k] / d['j_shift'][k].max())
                line.append(d['j_line'][k] / d['j_line'][k][0])
            band_plot(a_wave if ls == '--' else a_max, c['shifts'], shift, color, ls, legend)
            band_plot(a_line, c['gammas'], line, color, ls, legend)
        for a in (a_wave, a_max):
            for x in (-0.1, 0.1):  # half period of the 5 Hz source
                a.axvline(x, color='0.5', lw=0.8, ls=':')
            a.set_xlabel('Time shift (s)')
            a.set_ylim(-0.03, 1.05)
        a_wave.set_ylabel('Normalized misfit')
        a_line.set_xlabel(r'$\gamma$ (0: initial, 1: true model)')
        a_line.set_ylabel(r'Misfit / misfit at $\gamma=0$')
        title(a_wave, 0, 'Time shift, waveform')
        title(a_max, 1, 'Time shift, max-based')
        title(a_line, 2, 'Model line')
        fig.legend(*a_line.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
        save(fig, 'fig_landscape_%s' % model)


def grouped_bars(ax, values, styles, labels, legend_loc='best'):
    """Bars grouped by misfit (colour of the misfit) with one hatch per quantity. values: quantity x misfit, or
    seed x quantity x misfit, drawn as the mean with the range over the seeds. No legend when legend_loc is None."""
    v_all = np.array(values, dtype=float)
    if v_all.ndim == 2:
        v_all = v_all[None]
    mean, low, high = v_all.mean(0), v_all.min(0), v_all.max(0)
    n = mean.shape[0]
    width = 0.8 / n
    for q, (hatch, fill) in enumerate(styles):
        for j, lab in enumerate(OCCLUDED):
            color = STYLE[lab][1]
            xb = j - 0.4 + width * (q + 0.5)
            ax.bar(xb, mean[q, j], width, color=color if fill else 'white', edgecolor=color, hatch=hatch, lw=0.8)
            if len(v_all) > 1:
                ax.errorbar(xb, mean[q, j], yerr=[[mean[q, j] - low[q, j]], [high[q, j] - mean[q, j]]], color='k',
                            lw=0.6, capsize=1.5)
    ax.set_xticks(range(len(OCCLUDED)))
    ax.set_xticklabels([STYLE[lab][0] for lab in OCCLUDED], rotation=30, ha='right')
    handles = [matplotlib.patches.Patch(facecolor='0.5' if fill else 'white', edgecolor='0.3', hatch=hatch)
               for hatch, fill in styles]
    if legend_loc:
        ax.legend(handles, labels, loc=legend_loc, frameon=False)


def fig_alignment(models):
    """Data space and model space at the initial model and along the inversion: share of the misfit on
    the early arrivals (integrated gradients and occlusion), alignment of the descent direction with the
    direction to the true model along the inversion, and its change when the simulated early or later
    arrivals are replaced by the observed ones (epochs 0 and 10); one figure of 2 x 2 panels per model."""
    replaced = [('', True), ('////', False), ('....', False)]
    replaced_labels = ['Full gather', 'Early arrivals replaced', 'Later arrivals replaced']
    for model in models:
        adjs = npz_seeds('adjoint', model, 'adjoint_seed%d.npz')
        occs = npz_seeds('occlusion', model, 'occlusion_seed%d.npz')
        if not adjs or not occs:
            continue
        adj = adjs[0]
        fig, ax = plt.subplots(2, 2, figsize=(WIDTH, 5.6), constrained_layout=True)
        a_share, a_occ0, a_cos, a_occ10 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
        shares = [[[100 * d['%s@0/ig_win' % lab][0] / d['%s@0/ig_win' % lab].sum() for lab in OCCLUDED],
                   [100 * o['%s@0/removed' % lab][0] for lab in OCCLUDED]] for d, o in zip(adjs, occs)]
        grouped_bars(a_share, shares, [('', True), ('////', False)], ['Integrated gradients', 'Occlusion'])
        a_share.set_ylabel('Share on early arrivals (%)')
        title(a_share, 0, 'Attribution to the early arrivals')
        for a, e, k, loc in ((a_occ0, 0, 1, None), (a_occ10, 10, 3, 'upper left')):
            v = [[[o['%s@%d/cos_full' % (lab, e)] for lab in OCCLUDED]]
                 + [[o['%s@%d/cos_occluded' % (lab, e)][r] for lab in OCCLUDED] for r in (0, 1)] for o in occs]
            grouped_bars(a, v, replaced, replaced_labels, loc)
            if loc:  # leave room above the bars for the legend
                low, high = np.min(v), np.max(v)
                a.set_ylim(min(0, low) - 0.1 * (high - low), high + 0.6 * (high - min(0, low)))
            a.axhline(0, color='0.3', lw=0.8)
            a.set_ylabel('Alignment $c$')
            title(a, k, 'Alignment at epoch %d' % e)
        for _, lab, legend, color, ls in LANDSCAPE:
            eps = sorted({int(x.split('@')[1].split('/')[0]) for x in adj.files if x.startswith(lab + '@')})
            band_plot(a_cos, eps, [[d['%s@%d/cos_true' % (lab, e)] for e in eps] for d in adjs], color, ls, legend, 'o')
        a_cos.axhline(0, color='0.3', lw=0.8)
        a_cos.set_xlabel('Epoch')
        a_cos.set_ylabel('Alignment $c$')
        title(a_cos, 2, 'Alignment along the inversion')
        fig.legend(*a_cos.get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
        save(fig, 'fig_alignment_%s' % model)


def fig_adjoint_initial(models):
    """Adjoint sources of the middle shot at the initial model; one figure of 2 x 4 panels per model."""
    panels = [('l2', None), ('euclidean', None), ('siamesefwi', None),
              ('maxpool', None), ('mpbae', None), ('siamesefit', 'SiameseFit (random)')]
    for model in models:
        adj = npz('adjoint', case(model), 'adjoint_seed0.npz')
        if adj is None:
            continue
        fig, ax = plt.subplots(2, 4, figsize=(WIDTH, 4.6), sharex=True, sharey=True, constrained_layout=True)
        items = [('Observed data', observed(model)['d_obs'][:, SHOT].numpy())]
        items += [(name or STYLE[lab][0], adj['%s@0/adjoint_mid' % lab]) for lab, name in panels]
        for k, (name, g) in enumerate(items):
            a = ax[k // 4, k % 4]
            show_gather(a, g)
            title(a, k, name)
        for r in (0, 1):
            ax[r, 0].set_ylabel('Time (s)')
        for k in range(len(items), 8):
            ax[k // 4, k % 4].axis('off')
            above = ax[k // 4 - 1, k % 4]  # panel left without a neighbour below
            above.tick_params(labelbottom=True)
            above.set_xlabel('Receiver (km)')
        for a in ax[-1, :]:
            a.set_xlabel('Receiver (km)')
        save(fig, 'fig_adjoint_initial_%s' % model)


def plot_cka(ax, d):
    """Linear CKA between the SiameseFit representations and fixed 16 x 16 poolings of the input, along the
    inversion: mean and range over the XAI seeds."""
    epochs = sorted(int(k.split('@')[1].split('/')[0]) for k in d.files if k.startswith('seed0@') and k.endswith('/output4'))
    seeds = [sd for sd in XAI_SEEDS if all('seed%d@%d/diff_output4' % (sd, e) in d.files for e in epochs)]
    for layer, ref, legend, color, ls in (
            ('output4', 'maxpool16', 'Output vs max-pool 16/16', '#D55E00', '-'),
            ('output4', 'avgpool16', 'Output vs average pool 16/16', '#D55E00', '--'),
            ('deep4', 'maxpool16', 'Deep path vs max-pool 16/16', '#0072B2', '-'),
            ('skip4', 'maxpool16', 'Skip branch vs max-pool 16/16', '#999999', '-')):
        band_plot(ax, epochs, [[linear_cka(d['seed%d@%d/diff_%s' % (sd, e, layer)], d['refdiff/' + ref]) for e in epochs]
                               for sd in seeds], color, ls, legend)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Linear CKA')
    ax.set_ylim(0.4, 1.02)


def gather_grid(fig, spec, rows, show, k):
    """Grid of gathers in a gridspec cell: one row per (name, getter, title format), one column per
    epoch. Returns the index of the next panel label."""
    sub = spec.subgridspec(len(rows), len(show))
    for r, (name, get, fmt_title) in enumerate(rows):
        for c, e in enumerate(show):
            a = fig.add_subplot(sub[r, c])
            show_gather(a, get(e))
            title(a, k, fmt_title % e)
            k += 1
            if r < len(rows) - 1:
                a.set_xticklabels([])
            else:
                a.set_xlabel('Receiver (km)')
            if c == 0:
                a.set_ylabel('%s\nTime (s)' % name)
            else:
                a.set_yticklabels([])
    return k


def fig_training_fixed(models):
    """Effect of the training on fixed data: adjoint sources of the Siamese networks of several
    epochs, and misfit and adjoint-source statistics along the training."""
    nets = ['siamesefwi', 'siamesefit']
    for model in models:
        ds = npz_seeds('fixed_data', model, 'fixed_data_seed%d.npz')
        if not ds:
            continue
        d = ds[0]  # images: seed 0
        last = int(d['siamesefit/epochs'][-1])
        show = [e for e in d['epochs_show'] if e < last] + [last]  # ends at the last epoch (200 or 300)
        fig = plt.figure(figsize=(WIDTH, 6.0), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[2, 1.1])
        rows = [(STYLE[net][0], lambda e, net=net: d['%s@%d' % (net, e)], 'network %d') for net in nets]
        k = gather_grid(fig, outer[0], rows, show, 0)
        bottom = outer[1].subgridspec(1, 3)
        quantities = [('loss', 'Misfit / misfit at epoch 0', True, 'Misfit'),
                      ('cos0', 'Cosine with epoch 0', False, 'Direction of the adjoint'),
                      ('norm', 'Norm / norm at epoch 0', True, 'Size of the adjoint')]
        for q, (key, ylabel, log, name) in enumerate(quantities):
            a = fig.add_subplot(bottom[0, q])
            for net in nets:
                label, color, ls = STYLE[net]
                curves = [x['%s/%s' % (net, key)] / (x['%s/%s' % (net, key)][0] if key == 'loss' else 1) for x in ds]
                band_plot(a, d['%s/epochs' % net], curves, color, ls, label)
            if log:
                a.set_yscale('log')
            a.set_xlabel('Epoch of the network')
            a.set_ylabel(ylabel)
            title(a, k, name)
            k += 1
        fig.legend(*a.get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0))
        save(fig, 'fig_training_fixed_%s' % model)


def fig_sfit_epochs(models):
    """SiameseFit along the inversion: adjoint sources of the trained and frozen networks and of
    max-pool 16/16 at several epochs, CKA of the encoder with fixed poolings and the misfit along a
    time shift for the networks of several epochs (the trained networks on fixed data are in
    fig_training_fixed)."""
    for model in models:
        adj = npz('adjoint', case(model), 'adjoint_seed0.npz')
        cka = npz('cka', case(model), 'cka_grams.npz')
        convs = npz_seeds('landscape', model, 'landscape_seed%d.npz')
        conv = convs[0] if convs else None
        if adj is None or cka is None:
            continue
        show = sorted({int(x.split('@')[1].split('/')[0]) for x in adj.files if x.startswith('siamesefit@')})
        rows = [('SiameseFit', lambda e: adj['siamesefit@%d/adjoint_mid' % e], 'epoch %d'),
                ('Frozen', lambda e: adj['frozen@%d/adjoint_mid' % e], 'epoch %d'),
                ('Max-pool 16/16', lambda e: adj['maxpool@%d/adjoint_mid' % e], 'epoch %d')]
        fig = plt.figure(figsize=(WIDTH, 7.8), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[3, 2.0])
        k = gather_grid(fig, outer[0], rows, show, 0)
        bottom = outer[1].subgridspec(1, 2)
        a = fig.add_subplot(bottom[0, 0])
        plot_cka(a, cka)
        a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False)
        title(a, k, 'CKA of the SiameseFit encoder')
        k += 1
        a = fig.add_subplot(bottom[0, 1])
        if conv is not None:
            labels = list(conv['labels'])
            eps = sorted(int(l.split('@')[1]) for l in labels if l.startswith('siamesefit@'))
            colors = plt.get_cmap('Oranges')(np.linspace(0.45, 0.95, len(eps)))
            for e, color in zip(eps, colors):  # mean over the seeds
                js = [x['j_shift'][list(x['labels']).index('siamesefit@%d' % e)] for x in convs]
                a.plot(conv['shifts'], np.mean([j / j.max() for j in js], 0), color=color, label='SiameseFit, epoch %d' % e)
            j = conv['j_shift'][labels.index('maxpool')]
            a.plot(conv['shifts'], j / j.max(), color=STYLE['maxpool'][1], ls='--', label='Max-pool 16/16')
            a.set_xlabel('Time shift (s)')
            a.set_ylabel('Normalized misfit')
            a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
            title(a, k, 'Misfit along a time shift')
        save(fig, 'fig_sfit_epochs_%s' % model)


# ----------------------------------------------------------------------------- schematic of the method

INK, BACK, XAIC = '0.15', '#C0392B', '#5E3C99'  # forward path, backward path, XAI marks
CONVC, MAXC = '0.78', STYLE['siamesefit'][1]    # convolution + LeakyReLU, max-pooling


class Canvas:
    """Figure whose drawing coordinates are inches; thumbnails are axes placed in inches."""

    def __init__(self, width, height):
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis('off')
        self.ax.set_zorder(10)  # arrows and marks above the thumbnails

    def image(self, x, y, w, h, img, cmap, signed=False):
        """Thumbnail in the box (x, y, w, h); signed fields get a symmetric scale clipped at the 99th
        percentile, the others the range between the 1st and 99th percentiles."""
        a = self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])
        a.set_xticks([])
        a.set_yticks([])
        clip = np.percentile(np.abs(img), 99)
        lim = (-clip, clip) if signed else tuple(np.percentile(img, [1, 99]))
        a.imshow(img, cmap=cmap, aspect='auto', interpolation='bilinear', vmin=lim[0], vmax=lim[1])

    def stack(self, x, y, w, h, img, cmap, n=2, off=0.035):
        """Thumbnail with n empty frames behind it (a stack of feature maps)."""
        for k in range(n, 0, -1):
            a = self.fig.add_axes([(x + off * k) / self.W, (y + off * k) / self.H, w / self.W, h / self.H])
            a.set_xticks([])
            a.set_yticks([])
        self.image(x, y, w, h, img, cmap)

    def box(self, x, y, w, h, text='', fc='white', ec=INK, fontsize=8, lw=1.0):
        """Rounded box with centred text."""
        self.ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=fc, ec=ec, lw=lw))
        if text:
            self.ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)

    def rect(self, x, y, w, h, fc, ec=INK, lw=0.6):
        """Plain rectangle (network layers)."""
        self.ax.add_patch(matplotlib.patches.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))

    def arrow(self, p0, p1, color=INK, dashed=False, head=True, lw=1.2):
        """Arrow (or segment without head) from p0 to p1."""
        self.ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(
            arrowstyle='-|>' if head else '-', color=color, lw=lw, linestyle=(0, (4, 2)) if dashed else '-',
            shrinkA=0, shrinkB=0, mutation_scale=9))

    def text(self, x, y, s, **kw):
        """Text in inches (fontsize 8 unless given)."""
        kw.setdefault('fontsize', 8)
        self.ax.text(x, y, s, **kw)

    def mark(self, x, y, n):
        """Numbered circle that marks where an XAI analysis looks."""
        self.ax.add_patch(matplotlib.patches.Circle((x, y), 0.08, fc=XAIC, ec='none', zorder=5))
        self.ax.text(x, y, str(n), ha='center', va='center', fontsize=6.5, color='white', fontweight='bold', zorder=6)


def workflow_data(model):
    """Thumbnails of the loop for the middle shot at the linear initial model: velocity model,
    observed and simulated data, one latent map of the SiameseFit network at initialization (seed 0),
    and the adjoint source and velocity gradient of SiameseFit at epoch 0."""
    s = observed(model)
    m = initial_model(model)
    d_obs = s['d_obs'][:, SHOT:SHOT + 1]
    with torch.no_grad():
        prop = deepwave.scalar.Propagator({'vp': m}, protocol.DX, protocol.PML_WIDTH, protocol.ORDER, None)
        d_sim = prop(s['source'], s['x_s'][SHOT:SHOT + 1], s['x_r'][SHOT:SHOT + 1], protocol.DT)
        torch.manual_seed(0)
        net = Misfit('siamesefit', torch.device('cpu'), KEY[model]).net
        latent = [net(to_image(d) / torch.norm(d))[0] for d in (d_obs, d_sim)]
    c = int(torch.argmax(latent[0].std(dim=(1, 2))))  # most variable feature map
    adj = npz('adjoint', case(model), 'adjoint_seed0.npz')
    return {'m': m.numpy(), 'd_obs': d_obs[:, 0].numpy(), 'd_sim': d_sim[:, 0].numpy(),
            'z_obs': latent[0][c].numpy(), 'z_sim': latent[1][c].numpy(),
            's_adj': adj['siamesefit@0/adjoint_mid'], 'grad': adj['siamesefit@0/grad']}


def draw_loop(cv, t, y0):
    """Panel (a): one iteration of FWI with a Siamese misfit. Forward path (solid): the velocity model is simulated,
    the simulated and observed gathers go through the same network and the distance compares the outputs. Backward
    path (dashed): the distance updates the network (top) and, through the adjoint source and the adjoint wave
    equation, the velocity model (bottom). Numbered marks: where the XAI analyses look. All routes are orthogonal
    and no arrow crosses a box or a mark."""
    r1, r3, r4 = y0 + 2.45, y0 + 1.45, y0 + 0.45     # centres of the observed, simulated and backward rows
    gw, gh, mw, mh = 0.55, 0.72, 1.25, 0.42           # gather and model thumbnails
    xm, xb, xg, xn, xz, xd = 0.12, 1.62, 2.78, 3.62, 4.55, 5.3  # columns: model, solver, gathers, net, latent, distance
    bw, bh, nw, zw, zh, dw = 0.92, 0.46, 0.62, 0.24, 0.36, 0.86
    top = y0 + 3.02                                   # route of the network update, above the observed row
    cv.text(0.0, y0 + 3.15, '(a) One iteration of FWI with a Siamese misfit', fontsize=9)

    # forward path
    cv.image(xm, r3 - mh / 2, mw, mh, t['m'], 'turbo')
    cv.text(xm + mw / 2, r3 + mh / 2 + 0.06, r'Velocity model $\mathbf{m}$', ha='center', va='bottom')
    cv.box(xb, r3 - bh / 2, bw, bh, 'Wave equation')
    cv.arrow((xm + mw, r3), (xb, r3))
    cv.arrow((xb + bw, r3), (xg, r3))
    cv.image(xg, r3 - gh / 2, gw, gh, t['d_sim'], 'seismic', signed=True)
    cv.text(xg + gw / 2, r3 - gh / 2 - 0.05, r'Simulated $\mathbf{d}^{\mathrm{sim}}$', ha='center', va='top')
    cv.image(xg, r1 - gh / 2, gw, gh, t['d_obs'], 'seismic', signed=True)
    cv.text(xg + gw / 2, r1 + gh / 2 + 0.05, r'Observed $\mathbf{d}^{\mathrm{obs}}$', ha='center', va='bottom')
    # one network box for both gathers (shared weights)
    ny0, ny1 = r3 - bh / 2, r1 + bh / 2
    cv.box(xn, ny0, nw, ny1 - ny0, fc='#F6E3D6')
    cv.text(xn + nw / 2, (r1 + r3) / 2 + 0.1, r'$\Phi_\theta$', ha='center', va='center', fontsize=11)
    cv.text(xn + nw / 2, (r1 + r3) / 2 - 0.13, 'shared\nweights', ha='center', va='center', fontsize=6.5)
    # distance box spanning both rows
    dy0, dy1 = r3 - 0.25, r1 + 0.25
    cv.box(xd, dy0, dw, dy1 - dy0, 'Distance\n' + r'$\mathcal{D}_\theta$', fc='#EEEEEE', fontsize=8.5)
    for r, z in ((r1, t['z_obs']), (r3, t['z_sim'])):
        cv.arrow((xg + gw, r), (xn, r))
        cv.stack(xz, r - zh / 2, zw, zh, z, 'viridis')
        cv.arrow((xn + nw, r), (xz, r))
        cv.arrow((xz + zw + 0.08, r), (xd, r))
    cv.text(xz + zw / 2 + 0.035, r1 + zh / 2 + 0.09, r'$\Phi_\theta(\mathbf{d}^{\mathrm{obs}})$', ha='center', va='bottom')
    cv.text(xz + zw / 2 + 0.035, r3 - zh / 2 - 0.05, r'$\Phi_\theta(\mathbf{d}^{\mathrm{sim}})$', ha='center', va='top')

    # backward path 1: network update, routed above the observed row
    xc, xnc = xd + dw / 2, xn + nw / 2
    cv.arrow((xc, dy1), (xc, top), color=BACK, dashed=True, head=False)
    cv.arrow((xc, top), (xnc, top), color=BACK, dashed=True, head=False)
    cv.arrow((xnc, top), (xnc, ny1), color=BACK, dashed=True)
    cv.text((xc + xnc) / 2, top + 0.04, r'network update: $\partial\mathcal{D}/\partial\theta$, Adam', color=BACK,
            ha='center', va='bottom', fontsize=7.5)
    # backward path 2: adjoint source, adjoint wave equation and velocity update, routed below the simulated row
    cv.arrow((xc, dy0), (xc, r4), color=BACK, dashed=True, head=False)
    cv.arrow((xc, r4), (xg + gw, r4), color=BACK, dashed=True)
    cv.text((xc + xg + gw) / 2, r4 + 0.05, r'back-propagation through $\Phi_\theta$', color=BACK, ha='center',
            va='bottom', fontsize=7.5)
    cv.image(xg, r4 - gh / 2, gw, gh, t['s_adj'], 'seismic', signed=True)
    cv.text(xg + gw / 2, r4 - gh / 2 - 0.05, r'Adjoint source $\mathbf{s}^{\mathrm{adj}}$', ha='center', va='top')
    cv.box(xb, r4 - bh / 2, bw, bh, 'Adjoint\nwave equation', ec=BACK)
    cv.arrow((xg, r4), (xb + bw, r4), color=BACK, dashed=True)
    cv.arrow((xb, r4), (xm + mw, r4), color=BACK, dashed=True)
    cv.image(xm, r4 - mh / 2, mw, mh, t['grad'], 'seismic', signed=True)
    cv.text(xm + mw / 2, r4 - mh / 2 - 0.05, r'Gradient $\partial\mathcal{D}/\partial\mathbf{m}$', ha='center', va='top')
    cv.arrow((xm + mw / 2, r4 + mh / 2), (xm + mw / 2, r3 - mh / 2), color=BACK, dashed=True)
    cv.text(xm + mw / 2 + 0.05, (r3 + r4) / 2, 'model update: Adam', color=BACK, ha='left', va='center', fontsize=7)

    # XAI analyses, numbered in the order of the Methods section, on corners free of arrows
    cv.mark(xd + dw, dy1, 1)
    cv.mark(xg + gw, r4 + gh / 2, 2)
    cv.mark(xm + mw, r4 + mh / 2, 3)
    cv.mark(xz - 0.02, r3 + zh / 2 + 0.06, 4)
    cv.mark(xn + nw, ny1, 5)
    items = ['Misfit landscape', 'Adjoint source, IG, occlusion', 'Alignment of the update',
             'CKA of the representations', 'Networks on fixed data']
    cv.text(xm, r1 + 0.42, 'XAI analyses', fontsize=8, fontweight='bold', va='center')
    for k, item in enumerate(items):
        y = r1 + 0.22 - 0.17 * k
        cv.mark(xm + 0.08, y, k + 1)
        cv.text(xm + 0.22, y, item, va='center', fontsize=7.5)


def draw_networks(cv, y0):
    """Panels (b) SiameseFWI network and (c) SiameseFit encoder: layer height follows the size of the
    feature maps, layer width the number of channels; max-pooling in the SiameseFit colour."""
    H0, yc = 0.9, y0 + 0.92  # height of the input gather and centre line of the layers
    cv.text(0.0, y0 + 1.72, '(b) SiameseFWI network', fontsize=9)
    cv.text(3.3, y0 + 1.72, '(c) SiameseFit encoder', fontsize=9)

    # SiameseFWI: eight 3x3 convolutions without pooling, a 3x3 convolution of the input added to every
    # layer and the input added to the output
    x = 0.25
    cv.rect(x, yc - H0 / 2, 0.1, H0, fc='white')
    cv.text(x - 0.05, yc, r'$\mathbf{d}$', ha='right', va='center')
    x_in, x = x + 0.05, x + 0.22
    tops = []
    for ch in (1, 2, 2, 4, 4, 2, 1, 1):
        w = 0.05 + 0.025 * ch
        cv.rect(x, yc - H0 / 2, w, H0, fc=CONVC)
        tops.append(x + w / 2)
        cv.arrow((x - 0.1, yc), (x, yc), lw=0.8)
        x += w + 0.1
    xo = x + 0.12
    cv.ax.add_patch(matplotlib.patches.Circle((x + 0.02, yc), 0.05, fc='white', ec=INK, lw=0.8))
    cv.text(x + 0.02, yc, '+', ha='center', va='center', fontsize=8)
    cv.arrow((x - 0.1, yc), (x - 0.03, yc), lw=0.8)
    cv.arrow((x + 0.07, yc), (xo, yc), lw=0.8)
    cv.rect(xo, yc - H0 / 2, 0.1, H0, fc='white')
    cv.text(xo + 0.05, yc - H0 / 2 - 0.05, r'$\mathbf{d}+R_\theta(\mathbf{d})$', ha='center', va='top')
    yb = yc - H0 / 2 - 0.12  # skip branches below the layers
    cv.arrow((x_in, yc - H0 / 2), (x_in, yb), head=False, lw=0.8)
    cv.arrow((x_in, yb), (tops[-2], yb), head=False, lw=0.8)
    for xt in tops[:-1]:
        cv.arrow((xt, yb), (xt, yc - H0 / 2), lw=0.6)
    ya = yc + H0 / 2 + 0.1  # input connection above
    cv.arrow((x_in, yc + H0 / 2), (x_in, ya), head=False, lw=0.8)
    cv.arrow((x_in, ya), (x + 0.02, ya), head=False, lw=0.8)
    cv.arrow((x + 0.02, ya), (x + 0.02, yc + 0.05), lw=0.8)
    cv.text((x_in + x) / 2, ya + 0.03, 'input connection', ha='center', va='bottom', fontsize=7)
    cv.text((x_in + tops[-2]) / 2, yb - 0.04, 'convolution of the input added to each layer', ha='center',
            va='top', fontsize=7)

    # SiameseFit: four blocks of 3x3 convolution, LeakyReLU and 2x2 max-pooling; each adds a convolution of
    # the input max-pooled to its own output size
    x = 3.85
    cv.rect(x, yc - H0 / 2, 0.1, H0, fc='white')
    cv.text(x - 0.04, yc, r'$\mathbf{d}/\|\mathbf{d}\|$', ha='right', va='center')
    x_in, x = x + 0.05, x + 0.22
    h, outs = H0, []
    for k, ch in enumerate((16, 32, 64, 128)):
        w = 0.06 + 0.0016 * ch
        cv.arrow((x - 0.1, yc), (x, yc), lw=0.8)
        cv.rect(x, yc - h / 2, w, h, fc=CONVC)
        cv.rect(x + w, yc - h / 2, 0.05, h, fc=MAXC)
        cv.text(x + w / 2 + 0.025, yc + h / 2 + 0.03, str(ch), ha='center', va='bottom', fontsize=6.5)
        outs.append(x + w + 0.025)
        x += w + 0.05 + 0.13
        h /= 2
    xo = x + 0.02
    for k in (2, 1, 0):
        cv.rect(xo + 0.03 * k, yc - h / 2 + 0.03 * k, 0.1, max(h, 0.04), fc='white', lw=0.5)
    cv.arrow((x - 0.13, yc), (xo, yc), lw=0.8)
    cv.text(xo + 0.08, yc + 0.15, r'$\phi_\theta(\mathbf{d})$', ha='left', va='bottom', fontsize=8)
    cv.text(xo + 0.08, yc - 0.12, '128 maps,\n16 times\nsmaller', ha='left', va='top', fontsize=7)
    yb = yc - H0 / 2 - 0.27  # input line below the skip boxes, which sit on short branches (no line crosses a box)
    cv.arrow((x_in, yc - H0 / 2), (x_in, yb), head=False, lw=0.8)
    cv.arrow((x_in, yb), (outs[-1], yb), head=False, lw=0.8)
    for k, xt in enumerate(outs):  # skip branch k: convolution of the input, 2^k x 2^k max-pooling, added after block k
        cv.arrow((xt, yb), (xt, yb + 0.06), head=False, lw=0.6)
        cv.rect(xt - 0.08, yb + 0.06, 0.16, 0.11, fc=MAXC, lw=0.5)
        cv.text(xt, yb + 0.115, '%d' % 2 ** (k + 1), ha='center', va='center', fontsize=6, color='white')
        cv.arrow((xt, yb + 0.17), (xt, yc - H0 / 2 ** (k + 1)), lw=0.6)
    cv.text((x_in + outs[-1]) / 2, yb - 0.04, r'skip branches: convolution of the input, $2^k\times2^k$ max-pooling',
            ha='center', va='top', fontsize=7)

    # legend of the layers
    for k, (fc, label) in enumerate(((CONVC, r'$3\times3$ convolution + LeakyReLU'), (MAXC, 'max-pooling'))):
        x = 1.3 + 2.4 * k
        cv.rect(x, y0 - 0.2, 0.16, 0.1, fc=fc)  # below the labels of the skip branches
        cv.text(x + 0.22, y0 - 0.15, label, va='center', fontsize=7.5)


def fig_workflow(models):
    """Siamese misfit inside the FWI loop (Overthrust thumbnails) and the two architectures."""
    model = 'overthrust'
    cv = Canvas(WIDTH, 5.55)
    draw_loop(cv, workflow_data(model), 2.2)
    draw_networks(cv, 0.2)
    save(cv.fig, 'fig_workflow')


# ----------------------------------------------------------------------------- tables

def std(v):
    """Sample standard deviation (0 for a single value)."""
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def fmt(v, digits=2):
    """Mean and sample standard deviation as 'm $\\pm$ s'."""
    return '%.*f $\\pm$ %.*f' % (digits, np.mean(v), digits, std(v))


def summary(paths, model, init='linear'):
    """Final SNR, SSIM and relative data residual of each seed, and the escape epochs (None if a seed
    ends below the initial model, so that no escape epoch is reported)."""
    snr0 = initial_metrics(model, init)[0]
    res0 = initial_residual(model, init)
    out = {'snr': [], 'ssim': [], 'res': [], 'esc': []}
    for p in paths:
        d = metrics(p)
        out['snr'].append(d[-1, 3])
        out['ssim'].append(d[-1, 4])
        out['res'].append(d[-1, 2] / res0)
        out['esc'].append(int(d[np.argmax(d[:, 3] > snr0), 0]) if d[-1, 3] > snr0 else None)
    if None in out['esc']:
        out['esc'] = None
    return out


def cells_of(sm, best):
    """Table cells of one method and model; the values flagged in best are set in bold."""
    bold = lambda text, key: '\\textbf{\\boldmath %s}' % text if best.get(key) else text
    esc = sm['esc']
    escape = '--' if esc is None else ('%d' % esc[0] if min(esc) == max(esc) else '%d--%d' % (min(esc), max(esc)))
    return [bold(fmt(sm['snr']), 'snr'), bold(fmt(sm['ssim'], 3), 'ssim'), bold('%.3f' % np.mean(sm['res']), 'res'),
            bold(escape, 'esc')]


def best_flags(summaries):
    """For each method, which columns hold the best value among the methods (ties at the printed precision):
    highest SNR and SSIM, lowest residual, lowest mean escape epoch."""
    keys = {'snr': (lambda s: round(np.mean(s['snr']), 2), max), 'ssim': (lambda s: round(np.mean(s['ssim']), 3), max),
            'res': (lambda s: round(np.mean(s['res']), 3), min),
            'esc': (lambda s: np.mean(s['esc']) if s['esc'] is not None else None, min)}
    flags = {m: {} for m in summaries}
    for key, (value, pick) in keys.items():
        vals = {m: value(s) for m, s in summaries.items() if value(s) is not None}
        if vals:
            target = pick(vals.values())
            for m, v in vals.items():
                flags[m][key] = v == target
    return flags


def table_results(models, regimes=(('A2', 'linear', None),), name='tab_results'):
    """Final metrics of every method in each regime (results subdirectory, initial model, title row or
    None); best values in bold. The main regime goes to tab_results, the tests to tab_tests."""
    lines = []
    head = ' & '.join(['Misfit'] + ['SNR (dB) & SSIM & Residual & Escape'] * len(models)) + ' \\\\'
    for sub, init, regime in regimes:
        have = [m for m in models if runs(sub, m, 'l2', init)]
        if not have:
            continue
        if regime:
            lines.append('\\multicolumn{%d}{l}{\\textit{%s}} \\\\' % (1 + 4 * len(models), regime))
        cells = ['Initial model']
        for m in models:
            if m in have:
                snr0, ssim0 = initial_metrics(m, init)
                cells += ['%.2f' % snr0, '%.3f' % ssim0, '1', '--']
            else:
                cells += ['', '', '', '']
        lines.append(' & '.join(cells) + ' \\\\')
        table = {m: {k: summary(runs(sub, m, k, init), m, init) for k in MAIN if runs(sub, m, k, init)} for m in have}
        flags = {m: best_flags(table[m]) for m in have}
        for method in MAIN:
            cells = [STYLE[method][0]]
            for m in models:
                cells += cells_of(table[m][method], flags[m][method]) if m in have and method in table[m] \
                    else ['', '', '', '']
            lines.append(' & '.join(cells) + ' \\\\')
    cols = 'l' + 'rrrr' * len(models)  # GJI: numbers flush right
    groups = ' & '.join([''] + ['\\multicolumn{4}{c}{%s}' % NAMES[m] for m in models]) + ' \\\\'
    tex = '\n'.join(['\\begin{tabular}{%s}' % cols, '\\hline', groups, head, '\\hline'] + lines + ['\\hline', '\\end{tabular}'])
    os.makedirs(TAB_DIR, exist_ok=True)
    open(os.path.join(TAB_DIR, name + '.tex'), 'w').write(tex + '\n')
    print('saved', name)


CROSS_ROWS = [('initial', 'Initial model'), ('maxpool', r'Max-pool 16/16 ($\alpha=50$)'),
              ('maxpool_lr20', r'Max-pool 16/16 ($\alpha=20$)'), ('frozen', 'SiameseFit (frozen)'),
              ('siamesefit', 'SiameseFit (trained)'), ('mpbae', 'MPBAE'), ('euclidean', 'Euclidean')]


def table_cross(models):
    """Misfits evaluated at the final models of other inversions (seed 0), relative to the initial model, and
    alignment of the gradient of each run at epoch 10 per depth band and for its sign
    (xai.cross_misfit); one table per model with results."""
    for model in models:
        ds = npz_seeds('cross_misfit', model, 'cross_misfit_seed%d.npz')
        if not ds:
            continue
        d = ds[0]
        names = list(d['models'])
        vs = np.array([x['values'] / x['values'][0] for x in ds])  # seed x model x misfit
        aligns = np.array([x['align'] if 'align' in x.files else np.full((len(names), 4), np.nan) for x in ds])
        snrs = np.array([x['snr'] for x in ds])
        cell = lambda a: '--' if np.all(np.isnan(a)) else (fmt(a) if len(a) > 1 else '%.2f' % a[0])
        lines = [r'\begin{tabular}{lrrrrrrrrr}', r'\hline',
                 r' & & \multicolumn{4}{c}{Misfit relative to the initial model} & '
                 r'\multicolumn{4}{c}{Alignment at epoch %d} \\' % int(d['align_epoch'] if 'align_epoch' in d.files else 10),
                 r'Final model of & SNR (dB) & Max-pool & Frozen & MPBAE & Residual & 0--1 km & 1--2 km & $>2$ km & Sign \\',
                 r'\hline']
        for key, label in CROSS_ROWS:
            if key not in names:
                continue
            i = names.index(key)
            vals = ' & '.join('1' if key == 'initial' else cell(vs[:, i, q]) for q in range(vs.shape[2]))
            cols = ' & '.join(cell(aligns[:, i, q]) for q in range(aligns.shape[2]))
            snr = '%.2f' % snrs[0, i] if key == 'initial' else cell(snrs[:, i])
            lines.append(r'%s & %s & %s & %s \\' % (label, snr, vals, cols))
        lines += [r'\hline', r'\end{tabular}']
        with open(os.path.join(TAB_DIR, 'tab_cross_%s.tex' % model), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('saved tab_cross_%s' % model)


HYBRID_ROWS = [  # (label above 2 km, label below 2 km, folder of the seeds relative to results/)
    ('Linear', 'Linear', 'marm_linear/siamesefit'),
    ('Smooth', 'Linear', 'marm_smooth_top/siamesefit'),
    ('Linear', 'Smooth', 'marm_linear_top/siamesefit'),
    ('Smooth', 'Smooth', 'marm_smooth/siamesefit'),
]


def table_hybrid(models):
    """SiameseFit on the Marmousi2 model (alpha = 50) from the linear, smooth and hybrid initial models: SNR per depth
    band and of the whole model at the initial model and at the end (mean and standard deviation over the seeds)."""
    if 'marmousi2' not in models:
        return
    true = true_model('marmousi2').astype(np.float64)
    masks = band_masks('marmousi2')
    whole = np.ones_like(masks[0])
    lines = [r'\begin{tabular}{llrrrr}', r'\hline',
             r'\multicolumn{2}{c}{Initial model} & \multicolumn{3}{c}{SNR per depth band (dB)} & SNR (dB) \\',
             r'Above 2 km & Below 2 km & 0--1 km & 1--2 km & $>2$ km & \\', r'\hline']
    for up, down, rel in HYBRID_ROWS:
        seeds = sorted(glob.glob(os.path.join(ROOT, 'results', rel, 'seed*')))
        seeds = [d for d in seeds if os.path.exists(os.path.join(d, 'model_final.npy'))]
        if not seeds:
            continue
        m0 = torch.load(os.path.join(seeds[0], 'ckpt_epoch0000.pt'), map_location='cpu')['model'].numpy().astype(np.float64)
        finals = [np.load(os.path.join(d, 'model_final.npy')).astype(np.float64) for d in seeds]
        cells = []
        for k in masks + [whole]:
            v = [band_snr(m1, true, k) for m1 in finals]
            cells.append(r'%.1f $\rightarrow$ %.1f $\pm$ %.1f' % (band_snr(m0, true, k), np.mean(v), std(v)))
        lines.append(r'%s & %s & %s \\' % (up, down, ' & '.join(cells)))
    lines += [r'\hline', r'\end{tabular}']
    with open(os.path.join(TAB_DIR, 'tab_hybrid_marmousi2.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved tab_hybrid_marmousi2')


def runaway(m, true):
    """Cells outside [0.8, 1.2] times the velocity range of the true model."""
    return (m < 0.8 * true.min()) | (m > 1.2 * true.max())


def table_runaway(models):
    """Final SNR with and without the run-away cells, and their share of the model (mean over seeds)."""
    lines = [r'\begin{tabular}{l' + 'rrr' * len(models) + '}', r'\hline',
             ' & ' + ' & '.join(r'\multicolumn{3}{c}{%s}' % NAMES[m] for m in models) + r' \\',
             'Misfit' + ' & SNR (dB) & Without & Cells (\\%)' * len(models) + r' \\', r'\hline']
    for method in MAIN + ['maxpool', 'frozen', BRANCH]:
        cells = []
        for model in models:
            true = true_model(model).astype(np.float64)
            vals = []
            for p in runs(WHERE.get(method, 'ablations'), model, method):
                m = np.load(os.path.join(p, 'model_final.npy')).astype(np.float64)
                bad = runaway(m, true)
                vals.append((band_snr(m, true, np.ones_like(bad)), band_snr(m[~bad], true[~bad], np.ones((~bad).sum(), bool)),
                             100 * bad.mean()))
            cells.append(' & '.join('%.2f' % x for x in np.mean(vals, 0)) if vals else '-- & -- & --')
        lines.append('%s & %s \\\\' % (STYLE[method][0], ' & '.join(cells)))
    lines += [r'\hline', r'\end{tabular}']
    with open(os.path.join(TAB_DIR, 'tab_runaway.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved tab_runaway')


MECH_ROWS = [('overthrust', 'maxpool', 'Max-pool 16/16'), ('overthrust', 'skip128', 'Skip branch, 128 filters'),
             ('overthrust', 'frozen', 'SiameseFit, frozen'),
             ('marmousi2', 'maxpool', 'Max-pool 16/16'), ('marmousi2', 'maxpool_shift', 'Max-pool, shifted grid'),
             ('marmousi2', 'maxpool_top16', 'Mean of the 16 largest'), ('marmousi2', 'skip128', 'Skip branch, 128 filters'),
             ('marmousi2', 'frozen', 'SiameseFit, frozen'), ('marmousi2', 'siamesefit', 'SiameseFit, trained')]


def table_mechanism(models):
    """Supplementary Table S6: alignment of the actual updates between checkpoints against the threshold above which the error
    decreases, and alignment of the single-shot gradients in the shallowest kilometre and of their sign
    (xai.updates)."""
    lines = [r'\begin{tabular}{llrrrrrrr}', r'\hline',
             r' & & \multicolumn{4}{c}{Update between checkpoints} & \multicolumn{3}{c}{Single-shot gradient} \\',
             r' & & \multicolumn{2}{c}{Epochs 0--10} & \multicolumn{2}{c}{Epochs 20--30} & '
             r'\multicolumn{2}{c}{0--1 km} & Sign \\',
             r'Model & Misfit & $c$ & $t$ & $c$ & $t$ & Ep. 0 & Ep. 50 & Ep. 50 \\', r'\hline']
    for model, key, label in MECH_ROWS:
        ds = [x for x in npz_seeds('updates', model, 'updates_seed%d.npz') if 'A2/%s/c' % key in x.files]
        d4s = npz_seeds('updates', model, 'updates_seed%d.npz')
        if not ds or not d4s:
            continue
        cell = lambda v, f='%.2f': (f + r' $\pm$ ' + f) % (np.mean(v), std(v)) if len(v) > 1 else f % v[0]
        c = np.array([x['A2/%s/c' % key][[0, 2]] for x in ds])
        t = np.array([x['A2/%s/threshold' % key][[0, 2]] for x in ds])
        a4 = lambda e, q: cell([np.mean(x['A4/%s@%d/%s' % (key, e, q)]) for x in d4s if 'A4/%s@%d/%s' % (key, e, q) in x.files],
                               '%.3f') if any('A4/%s@%d/%s' % (key, e, q) in x.files for x in d4s) else '--'
        lines.append('%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\'
                     % (NAMES[model], label, cell(c[:, 0]), cell(t[:, 0]), cell(c[:, 1]), cell(t[:, 1]),
                        a4(0, 'c_shallow'), a4(50, 'c_shallow'), a4(50, 'c_sign')))
    lines += [r'\hline', r'\end{tabular}']
    with open(os.path.join(TAB_DIR, 'tab_mechanism.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved tab_mechanism')


ROBUST_MISFITS = [  # (style key, clean runs: results subdirectory and method directory)
    ('siamesefit', 'A2', 'siamesefit'), ('frozen', 'ablations', 'frozen'), ('skip128', 'ablations', 'skip128')]
ROBUST_TESTS = [('noise', 'Noise, 10 dB per shot'), ('lowcut', 'Low cut, 3 Hz')]


def robust_runs(model, test, method):
    """Finished runs of one misfit in one robustness test (utils.fwi.run_dir with test 'noise' or 'lowcut')."""
    paths = sorted(glob.glob(os.path.join(ROOT, 'results', '%s_linear_%s' % (KEY[model], test), method, 'seed*')))
    return [q for q in paths if os.path.exists(os.path.join(q, 'model_final.npy'))]


def table_robustness(models):
    """Final SNR, SSIM and escape epochs of the misfits in the robustness tests, with the clean data as reference
    (the data residual is left out: with noise it is measured against the noisy data)."""
    lines = [r'\begin{tabular}{l' + 'rrr' * len(models) + '}', r'\hline',
             ' & ' + ' & '.join(r'\multicolumn{3}{c}{%s}' % NAMES[m] for m in models) + r' \\',
             'Misfit' + ' & SNR (dB) & SSIM & Escape' * len(models) + r' \\', r'\hline']
    for test, title in [(None, 'Clean data')] + ROBUST_TESTS:
        rows = []
        for key, sub, method in ROBUST_MISFITS:
            cells, any_run = [], False
            for model in models:
                paths = runs(sub, model, method) if test is None else robust_runs(model, test, method)
                if test is None and key in ('frozen', 'skip128') and not robust_runs(model, 'noise', method):
                    paths = []  # clean reference only where the noise test was run
                if paths:
                    any_run = True
                    sm = summary(paths, model)
                    esc = sm['esc']
                    escape = '--' if esc is None else ('%d' % esc[0] if min(esc) == max(esc) else '%d--%d' % (min(esc), max(esc)))
                    cells += ['%s (%d)' % (fmt(sm['snr']), len(paths)), fmt(sm['ssim'], 3), escape]
                else:
                    cells += ['', '', '']
            if any_run:
                rows.append('%s & %s \\\\' % (STYLE[key][0], ' & '.join(cells)))
        if rows:
            lines.append(r'\multicolumn{%d}{l}{\textit{%s}} \\' % (1 + 3 * len(models), title))
            lines += rows
    lines += [r'\hline', r'\end{tabular}']
    with open(os.path.join(TAB_DIR, 'tab_robustness.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved tab_robustness')


def fig_robustness(models):
    """SNR along the inversion in the robustness tests (rows) for each model (columns): mean of the seeds (line) and
    range (shaded); thin dotted lines: the same misfits on the clean data."""
    fig, ax = plt.subplots(len(ROBUST_TESTS), len(models), figsize=(WIDTH, 2.4 * len(ROBUST_TESTS)), squeeze=False,
                           constrained_layout=True)
    k = 0
    for i, (test, title_text) in enumerate(ROBUST_TESTS):
        for j, model in enumerate(models):
            a = ax[i, j]
            for key, sub, method in ROBUST_MISFITS:
                paths = robust_runs(model, test, method)
                if not paths:
                    continue
                label, color, ls = STYLE[key]
                snr_band(a, paths, color, ls, label)
                clean = runs(sub, model, method)
                if clean:
                    a.plot(np.mean([metrics(q)[:, 3] for q in clean], 0), color=color, ls=':', lw=0.9)
            a.axhline(initial_metrics(model)[0], color='0.4', ls=':', lw=1.0, label='Initial model')
            a.set_xlabel('Epoch')
            a.set_ylabel('SNR (dB)')
            title(a, k, '%s, %s' % (NAMES[model], title_text[0].lower() + title_text[1:]))
            k += 1
    handles, labels = {}, {}
    for a in ax.ravel():
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, 'fig_robustness')


def table_training(models):
    """Supplementary Table S5: departure of SiameseFWI from the identity and training statistics of SiameseFit (identity.py,
    xai.identity, xai.fixed_data, xai.cka, xai.training_stats and the calibration runs with eps = 1e-8)."""
    rng = lambda v, f='%.2f': (f + (' to ' if np.min(v) < 0 else '--') + f) % (np.min(v), np.max(v))
    col = {}
    for model in models:
        c = case(model)
        idn, fxs = npz('identity', c, 'identity.npz'), npz_seeds('fixed_data', model, 'fixed_data_seed%d.npz')
        fx = fxs[0] if fxs else None
        tr, g = npz('training', c, 'training_stats.npz'), npz('cka', c, 'cka_grams.npz')
        if idn is None or fx is None or tr is None or g is None:
            continue
        last = int(fx['siamesefit/epochs'][-1])
        seeds = sorted({int(k.split('/')[1][4:].split('@')[0]) for k in idn.files if k.startswith('siamesefwi/seed')})
        q = lambda e, key: [float(idn['siamesefwi/seed%d@%d/%s' % (sd, e, key)]) for sd in seeds]
        fe = list(fx['siamesefwi/epochs'])
        tf = list(fx['siamesefit/epochs'])
        gseeds = sorted({int(k[4:].split('@')[0]) for k in g.files if k.startswith('seed')})
        cross = [linear_cka(g['seed%d@%d/diff_output4' % (a, last)], g['seed%d@%d/diff_output4' % (b, last)])
                 for a in gseeds for b in gseeds if a < b]
        own = [linear_cka(g['seed%d@%d/diff_output4' % (a, last)], g['seed%d@0/diff_output4' % a]) for a in gseeds]
        wd = [float(g[k]) for k in g.files if k.startswith('seed') and '@%d/wdist' % last in k]
        cal = sorted(glob.glob(os.path.join(ROOT, 'results', 'calibration', c, 'siamesefit', 'lr*_eps1e-8', 'seed0')))
        cal = [metrics(r)[-1, 3] for r in cal if os.path.exists(os.path.join(r, 'metrics.csv'))]
        col[model] = [
            None,
            rng(q(0, 'r_sim')), rng(q(0, 'cos_adjoint')),
            rng(100 * np.array(q(last, 'loss_ratio')), '%.1f'), rng(q(last, 'cos_adjoint')),
            rng([100 * x['siamesefwi/loss'][-1] / x['siamesefwi/loss'][0] for x in fxs], '%.1f'),
            rng([x['siamesefwi/cos0'][-1] for x in fxs], '%.3f'),
            None,
            rng([100 * x['siamesefit/loss'][-1] / x['siamesefit/loss'][0] for x in fxs]),
            '1/%.0f--1/%.0f' % (1 / max(x['siamesefit/norm'][-1] for x in fxs), 1 / min(x['siamesefit/norm'][-1] for x in fxs)),
            '%s; %s' % (rng([x['siamesefit/cos0'][tf.index(10)] for x in fxs]), rng([x['siamesefit/cos0'][-1] for x in fxs])),
            rng(tr['zero_ratio'], '%.4f'), rng(tr['growth'], '%.0f'), rng(100 * np.array(wd), '%.0f'),
            rng(cross, '%.3f'), rng(own, '%.3f'),
            rng(cal) if cal else '--']
    labels = [r'\textit{SiameseFWI}',
              r'$\|R_\theta(\mathbf{x})\|/\|\mathbf{x}\|$, initial network',
              'Cosine with the Euclidean adjoint source, initial network',
              r'Loss relative to the Euclidean loss, last epoch (\%)',
              'Cosine with the Euclidean adjoint source, last epoch',
              r'Fixed data: misfit at the last epoch relative to epoch 0 (\%)',
              'Fixed data: cosine with the initial adjoint source, last epoch',
              r'\textit{SiameseFit}',
              r'Fixed data: misfit at the last epoch relative to epoch 0 (\%)',
              'Fixed data: norm of the adjoint source relative to epoch 0',
              'Fixed data: cosine with the initial adjoint source, epoch 10 and last epoch',
              r'$\|\phi_\theta(\mathbf{0})\|/\|\phi_\theta(\mathbf{x})\|$, last epoch',
              'Output norm, last relative to initial network',
              r'Change of the weights relative to their initial norm (\%)',
              'CKA between seeds, last epoch',
              'CKA with the initial network of the same seed, last epoch',
              r'Final SNR with $\epsilon=10^{-8}$, $\alpha\in\{20,40,80\}$ (dB)']
    have = [m for m in models if m in col]
    lines = [r'\begin{tabular}{l' + 'r' * len(have) + '}', r'\hline',
             'Quantity & ' + ' & '.join(NAMES[m] for m in have) + r' \\', r'\hline']
    for i, label in enumerate(labels):
        if col[have[0]][i] is None:
            lines.append(r'\multicolumn{%d}{l}{%s} \\' % (1 + len(have), label))
        else:
            lines.append('%s & %s \\\\' % (label, ' & '.join(col[m][i] for m in have)))
    lines += [r'\hline', r'\end{tabular}']
    with open(os.path.join(TAB_DIR, 'tab_training.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('saved tab_training')


FIGURES = {'workflow': fig_workflow, 'models': fig_models, 'performance': fig_performance,
           'ablations': fig_ablations, 'landscape': fig_landscape, 'adjoint_initial': fig_adjoint_initial,
           'alignment': fig_alignment, 'training_fixed': fig_training_fixed, 'sfit_epochs': fig_sfit_epochs,
           'step_schedule': fig_step_schedule, 'calibration': fig_calibration,
           'results': lambda models: table_results(models, name='tab_results'),  # Table 3, both models
           'tab_cross': table_cross, 'tab_hybrid': table_hybrid, 'tab_runaway': table_runaway,
           'tab_mechanism': table_mechanism, 'robustness': fig_robustness, 'tab_robustness': table_robustness,
           'tab_training': table_training}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=['overthrust', 'marmousi2'], choices=list(KEY))
    p.add_argument('--only', nargs='+', default=list(FIGURES), choices=list(FIGURES))
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    torch.set_num_threads(args.threads)
    set_style()
    models = available(args.models)
    print('models with complete main-regime runs:', models)
    for name in args.only:
        FIGURES[name](list(args.models) if name == 'models' else models)


if __name__ == '__main__':
    main()
