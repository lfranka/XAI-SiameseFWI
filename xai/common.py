"""Helpers shared by the explainability analyses: runs, checkpoints and misfits of the public layout.

The analyses read the inversions written by utils.fwi.run (results/<model>_<init>/<method>/seed<k>) and write their
outputs to results/xai/<analysis>/<model>_<init>/.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import protocol  # noqa: E402
from utils.fwi import run_dir  # noqa: E402
from utils.misfits import ABLATIONS, MAXPOOL_TESTS, METHODS, Misfit  # noqa: E402

XAI_ROOT = os.path.join(protocol.ROOT, 'results', 'xai')
SHOTS = (0, 5, 10, 15, 20, 25)  # subset of shots of the analyses
MIDDLE_SHOT = 15


def device(name=None):
    return torch.device(name or ('cuda' if torch.cuda.is_available() else 'cpu'))


def out_dir(analysis, model, init='linear'):
    """Output folder of one analysis and model (created if needed)."""
    d = os.path.join(XAI_ROOT, analysis, '%s_%s' % (model, init))
    os.makedirs(d, exist_ok=True)
    return d


def run_path(model, method, seed=0, init='linear', test=None):
    return run_dir(model, init, method, seed, test)


def finished(model, method, init='linear', test=None):
    """Seed folders of the finished runs of one misfit."""
    paths = sorted(glob.glob(os.path.join(os.path.dirname(run_path(model, method, 0, init, test)), 'seed*')))
    return [p for p in paths if os.path.exists(os.path.join(p, 'model_final.npy'))]


def config(run):
    return json.load(open(os.path.join(run, 'config.json')))


def checkpoints(run):
    """Sorted list of (epoch, path) of the stored checkpoints of a run."""
    paths = glob.glob(os.path.join(run, 'ckpt_epoch*.pt'))
    return sorted((int(os.path.basename(p)[10:14]), p) for p in paths)


def checkpoint(run, epoch, dev):
    return torch.load(os.path.join(run, 'ckpt_epoch%04d.pt' % epoch), map_location=dev)


def method_of(run):
    """Misfit of a run: the name in its config.json, or, if that is not a misfit of utils.misfits (runs written by an
    older layout), the name of its folder without a suffix such as '_lr20'."""
    names = METHODS + ABLATIONS + MAXPOOL_TESTS
    name = config(run)['method']
    if name in names:
        return name
    folder = os.path.basename(os.path.dirname(os.path.abspath(run)))
    return folder if folder in names else next(n for n in sorted(names, key=len, reverse=True) if folder.startswith(n))


def misfit_of(run, model, dev, state=None):
    """Misfit of a run (its network, if any, loaded from the checkpoint state when given). The random seed of the run is
    set first, as in utils.fwi.run, so that fixed random operators that are not stored in the checkpoints (the filters
    of the skip branches, the random projection) are rebuilt as in the inversion."""
    torch.manual_seed(config(run)['seed'])
    f = Misfit(method_of(run), dev, model)
    if f.net is not None and state is not None:
        f.net.load_state_dict(state['net'])
    return f


def cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def alignment(g, m, m_true, water_rows=0):
    """Cosine between the descent direction -g and the direction to the true model, over the cells updated by the
    inversion (the water layer is reset after each update and is left out)."""
    g = g[water_rows:].flatten()
    direction = (m_true - m)[water_rows:].flatten()
    return float(torch.dot(-g, direction) / (g.norm() * direction.norm()))


def simulate(m, s, shots):
    """Gathers [nt, len(shots), nrec] simulated in model m (with gradients if m requires them)."""
    import deepwave
    prop = deepwave.scalar.Propagator({'vp': m}, protocol.DX, protocol.PML_WIDTH, protocol.ORDER, None)
    return prop(s['source'].repeat(1, len(shots), 1), s['x_s'][list(shots)], s['x_r'][list(shots)], protocol.DT)
