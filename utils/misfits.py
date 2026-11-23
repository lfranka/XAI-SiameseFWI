"""Misfit functions: SiameseFWI, SiameseFit, MPBAE, the classical baselines and the ablations of SiameseFit.

Every misfit receives one observed and one simulated shot gather, both [nt, 1, num_receivers], and
returns a scalar loss. Networks and their optimizers follow the official notebooks of each method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim

from Models import Module22, Module22xP_official
from utils.protocol import DT, MODELS

# Siamese networks: module, width, Adam settings and gain of the Glorot initialization (official notebooks)
NETWORKS = {
    'siamesefwi': {'module': Module22, 'width': 1, 'lr': 1e-4, 'betas': (0.5, 0.99), 'eps': 1e-6, 'gain': 1.0},
    'siamesefit': {'module': Module22xP_official, 'width': 16, 'lr': 1e-6, 'betas': (0.5, 0.9), 'eps': 1e-8,
                   'gain': 1.0},
}

# misfits compared in the article
METHODS = ('l2', 'euclidean', 'siamesefwi', 'siamesefit', 'mpbae')

# ablations of SiameseFit: the trained encoder is replaced by a fixed operator, while the normalization of
# the data, the form of the loss and the optimizer of the velocity model are kept
#   identity: no encoder; avgpool / maxpool: 16 x 16 pooling with stride 16; randproj: fixed random linear
#   projection with the same downsampling and 128 channels; frozen: SiameseFit network kept at its random
#   initialization; maxpool_l2: 16 x 16 max-pooling with the L2 loss of MPBAE instead of the SiameseFit loss;
#   skip128: the last skip branch of the frozen SiameseFit network alone (its 128 random 3 x 3 convolutions of the
#   input, LeakyReLU and 16 x 16 max-pooling)
ABLATIONS = ('identity', 'avgpool', 'randproj', 'maxpool', 'skip128', 'frozen', 'maxpool_l2')
# tests of the max-pooling on the Marmousi2 model (Section IV-B2), with the SiameseFit loss:
#   maxpool_shift: grid of the blocks shifted by random offsets (0 to 15 samples in time and receivers) at each call
#   maxpool_top16: mean of the 16 largest samples of each block instead of the maximum
#   maxpool_abs:   max-pooling of the absolute value of the data
#   maxpool_mute:  early arrivals (up to EARLY_MUTE s after the first break of the observed data) muted in both gathers
MAXPOOL_TESTS = ('maxpool_shift', 'maxpool_top16', 'maxpool_abs', 'maxpool_mute')
POOL = 16        # total downsampling of the SiameseFit encoder along each axis
TOPK = 16        # samples averaged per block by maxpool_top16
EARLY_MUTE = 0.4  # s


def to_image(d):
    """[nt, 1, nrec] -> [1, 1, nt, nrec], the layout expected by the 2D CNNs."""
    return d.permute(1, 0, 2).unsqueeze(1)


def euclidean(a, b):
    """Mean over time of the L2 norm along the receiver axis (loss of the notebooks)."""
    return F.pairwise_distance(a, b, keepdim=True).mean()


def normalized_latent_loss(encoder, d_obs, d_sim):
    """SiameseFit loss: gathers normalized by their norm, encoded, compared with the Euclidean
    distance and normalized by the norm of the simulated features."""
    real, fake = to_image(d_obs), to_image(d_sim)
    o1 = encoder(real / torch.norm(real, p=2))
    o2 = encoder(fake / torch.norm(fake, p=2))
    return euclidean(o1, o2) / torch.norm(o2, p=2)


def shifted_latent_loss(pool, d_obs, d_sim):
    """SiameseFit loss on block maxima with the grid of the blocks shifted by a random offset, drawn once per call
    and applied to both gathers after their normalization (the first offset samples and receivers are left out)."""
    real, fake = to_image(d_obs), to_image(d_sim)
    real, fake = real / torch.norm(real, p=2), fake / torch.norm(fake, p=2)
    t0, r0 = (int(v) for v in torch.randint(0, POOL, (2,)))
    o1, o2 = pool(real[..., t0:, r0:]), pool(fake[..., t0:, r0:])
    return euclidean(o1, o2) / torch.norm(o2, p=2)


def mute_early(d_obs, d_sim, threshold=0.02):
    """Zero both gathers [nt, 1, nrec] before EARLY_MUTE s after the first break of each observed trace (first
    sample above threshold x trace maximum)."""
    a = d_obs[:, 0].abs()
    first = (a > threshold * a.max(0, keepdim=True).values).float().argmax(0)
    t = torch.arange(d_obs.shape[0], device=d_obs.device).view(-1, 1)
    keep = (t >= first.view(1, -1) + int(round(EARLY_MUTE / DT))).float().unsqueeze(1)
    return d_obs * keep, d_sim * keep


class Abs(nn.Module):
    def forward(self, x):
        return x.abs()


class TopKPool(nn.Module):
    """Mean of the k largest samples of each non-overlapping size x size block (remainders left out)."""

    def __init__(self, size, k):
        super().__init__()
        self.size, self.k = size, k

    def forward(self, x):
        b = x.unfold(2, self.size, self.size).unfold(3, self.size, self.size)  # [n, c, rows, cols, size, size]
        return b.reshape(*b.shape[:4], -1).topk(self.k, dim=-1).values.mean(-1)


def skip_branch(n_filters, device):
    """Last skip branch of the SiameseFit network with its first n_filters filters: the whole network is built as for
    SiameseFit (same random stream as the frozen network of the same seed) and only that branch is kept."""
    cfg = NETWORKS['siamesefit']
    net = cfg['module'].SiameseNetwork(cfg['width']).apply(lambda m: cfg['module'].weights_init(m, cfg['gain']))
    conv = nn.Conv2d(1, n_filters, kernel_size=3, stride=1, padding=1)
    with torch.no_grad():
        conv.weight.copy_(net.cnnXX4.weight[:n_filters])
        conv.bias.copy_(net.cnnXX4.bias[:n_filters])
    return nn.Sequential(conv, nn.LeakyReLU(0.5), nn.MaxPool2d(POOL, stride=POOL)).requires_grad_(False).to(device)


def global_latent_loss(encoder, d_obs, d_sim):
    """MPBAE loss: gathers normalized by their norm, encoded and compared with the L2 norm of the difference."""
    real, fake = to_image(d_obs), to_image(d_sim)
    return torch.norm(encoder(fake / torch.norm(fake, p=2)) - encoder(real / torch.norm(real, p=2)), p=2)


class Misfit:
    """Misfit function (a name of METHODS, ABLATIONS or MAXPOOL_TESTS) with an optional Siamese network.

    MPBAE (Mu et al., 2026) uses q successive 2 x 2 max-poolings with stride 1, which are identical to a single
    (q + 1) x (q + 1) max-pooling with stride 1; q is 10 for the Overthrust and 18 for the Marmousi2 model.
    """

    def __init__(self, name, device, model='over'):
        self.name = name
        self.net, self.net_optimizer, self.encoder = None, None, None

        if name in ('siamesefwi', 'siamesefit', 'frozen'):
            cfg = NETWORKS['siamesefit' if name == 'frozen' else name]
            module = cfg['module']
            net = module.SiameseNetwork(cfg['width'])
            self.net = net.apply(lambda m: module.weights_init(m, cfg['gain'])).to(device)
            if name == 'frozen':
                self.net.requires_grad_(False)
            else:
                self.net_optimizer = optim.Adam(self.net.parameters(), lr=cfg['lr'], betas=cfg['betas'],
                                                eps=cfg['eps'])
        elif name == 'identity':
            self.encoder = nn.Identity()
        elif name == 'avgpool':
            self.encoder = nn.AvgPool2d(POOL, stride=POOL)
        elif name == 'randproj':
            proj = nn.Conv2d(1, 128, kernel_size=POOL, stride=POOL, bias=False)
            init.xavier_uniform_(proj.weight, 1.0)
            self.encoder = proj.requires_grad_(False).to(device)
        elif name in ('maxpool', 'maxpool_l2', 'maxpool_shift', 'maxpool_mute'):
            self.encoder = nn.MaxPool2d(POOL, stride=POOL)
        elif name == 'skip128':
            self.encoder = skip_branch(128, device)
        elif name == 'maxpool_top16':
            self.encoder = TopKPool(POOL, TOPK)
        elif name == 'maxpool_abs':
            self.encoder = nn.Sequential(Abs(), nn.MaxPool2d(POOL, stride=POOL))
        elif name == 'mpbae':
            self.encoder = nn.MaxPool2d(MODELS[model]['mpbae_window'], stride=1)
        elif name not in ('l2', 'euclidean'):
            raise ValueError('unknown misfit: %s' % name)

    def __call__(self, d_obs, d_sim):
        """Loss between one observed and one simulated shot gather."""
        if self.name == 'l2':
            return F.mse_loss(d_sim, d_obs)
        if self.name == 'euclidean':
            return euclidean(to_image(d_sim), to_image(d_obs))
        if self.name == 'siamesefwi':
            _, _, o1, o2 = self.net(to_image(d_sim), to_image(d_obs))
            return euclidean(o1, o2)
        if self.name in ('siamesefit', 'frozen'):
            return normalized_latent_loss(self.net, d_obs, d_sim)
        if self.name in ('mpbae', 'maxpool_l2'):
            return global_latent_loss(self.encoder, d_obs, d_sim)
        if self.name == 'maxpool_shift':
            return shifted_latent_loss(self.encoder, d_obs, d_sim)
        if self.name == 'maxpool_mute':
            return normalized_latent_loss(self.encoder, *mute_early(d_obs, d_sim))
        return normalized_latent_loss(self.encoder, d_obs, d_sim)
