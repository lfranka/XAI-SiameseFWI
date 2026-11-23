# XAI-SiameseFWI

Reproducible material for **"Why do Siamese misfit functions escape cycle skipping in full-waveform inversion? An
explainable artificial intelligence analysis"** (Queiroz, Silveira, Macedo and de Figueiredo, submitted to
*IEEE Access*).

The repository compares SiameseFWI, SiameseFit, MPBAE and two classical misfits (squared $L_2$ norm and Euclidean
distance) under a single protocol on the Overthrust and Marmousi2 models, dissects SiameseFit with ablations, explains
the behavior of each misfit with explainability analyses computed from checkpoints stored during the inversions.

## Project structure

* :open_file_folder: **data**: true and initial velocity models of the Overthrust and Marmousi2 models and source
  wavelets, from the [SiameseFWI](https://github.com/DeepWave-KAUST/Siamese_FWI-pub) and
  [SiameseFit](https://github.com/DeepWave-KAUST/SiameseFit-pub) repositories;
* :open_file_folder: **Models**: networks of SiameseFWI (`Module22.py`) and SiameseFit (`Module22xP_official.py`, a
  reconstruction of the network behind the official SiameseFit results; see Section S1 of the supplementary material);
* :open_file_folder: **utils**: modelling protocol (`protocol.py`), misfit functions (`misfits.py`), inversion
  (`fwi.py`), figures and tables of the article (`figures.py`), plots of single runs (`plots.py`) and the utilities of
  the official repositories (`utils.py`);
* :open_file_folder: **xai**: explainability analyses (one module per analysis, run as `python -m xai.<analysis>`);
* :open_file_folder: **deepwave-old**: Deepwave 0.0.8, the wave-equation solver used by the official repositories;
* :page_facing_up: **run_all.sh**: every inversion, analysis, figure and table of the article;
* :open_file_folder: **results**: created by the notebooks and scripts (not versioned).

## Notebooks

| Notebook | Content | Keys |
|---|---|---|
| `FWI.ipynb` | one inversion with a misfit of the article, also in the robustness tests | `model` (`'over'`, `'marm'`), `init` (`'linear'`; `'smooth'`, `'smooth_top'`, `'linear_top'` for the deep-band test), `method` (`'l2'`, `'euclidean'`, `'siamesefwi'`, `'siamesefit'`, `'mpbae'`), `seed`, `test` (`None`, `'noise'`, `'lowcut'`) |
| `Ablations.ipynb` | SiameseFit with its encoder replaced by fixed operators, and the tests of the max-pooling | `model`, `variant` (`'identity'`, `'avgpool'`, `'randproj'`, `'maxpool'`, `'skip128'`, `'frozen'`, `'maxpool_l2'`, `'maxpool_shift'`, `'maxpool_top16'`, `'maxpool_abs'`, `'maxpool_mute'`), `seed` |
| `XAI.ipynb` | one explainability analysis and its figure | `model`, `analysis` (`'landscape'`, `'adjoint'`, `'occlusion'`, `'identity'`, `'fixed_data'`, `'cka'`, `'depth'`, `'training_stats'`, `'cross_misfit'`, `'updates'`) |
| `Figures.ipynb` | every figure and table of the article from the saved results | -- |

The same steps run from the command line, from the root of the repository:

```
python -m utils.fwi --model marm --init linear --method siamesefit --seed 0
python -m xai.adjoint --model marm
python -m utils.figures --only ablations
```

Inversions are written to `results/<model>_<init>[_<test>]/<method>/seed<k>/` (configuration, metrics per epoch, final
model and checkpoints of the velocity model and network every 10 epochs), analyses to
`results/xai/<analysis>/<model>_linear/`, figures to `results/figures/` and tables to `results/tables/`.

## Reproducing the article

`./run_all.sh` runs everything in order; blocks can be run separately, e.g. `./run_all.sh main ablations`:

| Block | Content | GPU time (RTX A4000) |
|---|---|---|
| `calibration` | grid of steps and Adam eps of the velocity model (supplementary Section S2) | about 1 day |
| `main` | five misfits, five seeds, both models (Section IV) | about 1 day |
| `ablations` | ablation ladder and tests of the max-pooling, three seeds (Sections IV-A2 and IV-B2) | about 1 day |
| `step` | Euclidean misfit with a constant step (supplementary Section S2) | 3 h |
| `hybrids` | SiameseFit from hybrid and smooth initial models, three seeds (Section IV-B1) | 8 h |
| `robustness` | noise and missing low frequencies (Section V-B) | 10 h |
| `reproduction` | settings of the official SiameseFit notebook and of the MPBAE publication (supplementary Section S1) | 4 h |
| `analyses` | explainability analyses, seeds 0 to 2 (Sections IV-A3 and IV-B3, supplementary Sections S3 and S4) | 5 h |
| `figures` | figures and tables (see `Figures.ipynb` for the map to the article) | 10 min |

Runs with identical settings are not bit-for-bit reproducible, because the GPU kernels of Deepwave accumulate the
velocity gradient with atomic additions; repeated runs differ by up to 0.3 dB in the final SNR, and by more at a fixed
epoch when they leave the initial basin at different times (supplementary Section S2).

## Installation

The environment has the packages of the `FWIGAN` environment of the SiameseFWI and SiameseFit repositories (Python 3.8,
PyTorch 1.7.1 with CUDA 11.0 and Deepwave 0.0.8). With conda installed, run

```
./install_env.sh
```

which creates the `xai-fwi` environment, compiles Deepwave 0.0.8 from `./deepwave-old` with a CUDA 11.x toolkit
(default `/usr/local/cuda-11.8`, set `CUDA_HOME` to change it) and tests PyTorch and Deepwave on the GPU.
Then activate it with `conda activate xai-fwi`. The experiments of the article were run on an NVIDIA RTX A4000
(16 GB).

## Licence

The code of this study is distributed under the MIT licence (`LICENSE`). The data, the network modules,
`utils/utils.py` and `deepwave-old` come from the repositories of DeepWave-KAUST and Alan Richardson and keep
their MIT licences (`LICENSE_DeepWave-KAUST`, `deepwave-old/LICENSE`).

## Citation

To be added after publication.
