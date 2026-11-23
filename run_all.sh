#!/bin/bash
# All inversions, analyses, figures and tables of the article, in blocks that can be run separately:
#   ./run_all.sh                     every block, in order
#   ./run_all.sh main ablations      only these blocks
# Blocks: calibration main ablations step hybrids robustness reproduction analyses figures
# Approximate GPU time on an NVIDIA RTX A4000 (one run at a time): calibration 1 day, main 1 day, ablations 1 day,
# step 3 h, hybrids 8 h, robustness 10 h, reproduction 4 h, analyses 5 h, figures 10 min.
# Each run writes results/<model>_<init>[_<test>]/<misfit>/seed<k> (see utils/fwi.py); finished runs are skipped.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
BLOCKS=${@:-calibration main ablations step hybrids robustness reproduction analyses figures}

fwi() {  # out, then arguments of utils.fwi; skipped if the run is complete
  local out=$1; shift
  if [ -f $out/model_final.npy ]; then echo "SKIP $out"; return; fi
  mkdir -p $out
  echo "START $out $(date +%H:%M:%S)"
  $PY -W ignore -m utils.fwi "$@" --out $out > $out/run.log 2>&1
  echo "END   $out $(date +%H:%M:%S) $(grep -o 'complete.*\|Error.*' $out/run.log | tail -1)"
}

seeds() { seq 0 $(($1 - 1)); }

for block in $BLOCKS; do case $block in

calibration)  # supplementary Section S2 (Table S3, Fig. S1): seed 0, step halved every 100 epochs
  for m in l2 euclidean siamesefwi siamesefit mpbae; do
    for lr in 20 40 80; do for eps in 1e-22 1e-8; do
      fwi results/calibration/over_linear/$m/lr${lr}_eps${eps}/seed0 --model over --method $m --lr $lr --eps $eps
    done; done
  done
  # Marmousi2: the eps selected on the Overthrust model, with the step of SiameseFit for this model (50)
  for m in l2 euclidean siamesefwi siamesefit mpbae; do
    eps=1e-8; [ $m = siamesefit ] && eps=1e-22
    for lr in 20 40 50 80; do
      fwi results/calibration/marm_linear/$m/lr${lr}_eps${eps}/seed0 --model marm --method $m --lr $lr --eps $eps
    done
  done
  # misfits whose best step was the smallest of the grid
  for m in l2 euclidean siamesefwi mpbae; do fwi results/calibration/over_linear/$m/lr10_eps1e-8/seed0 --model over --method $m --lr 10 --eps 1e-8; done
  for m in l2 euclidean siamesefwi; do fwi results/calibration/marm_linear/$m/lr10_eps1e-8/seed0 --model marm --method $m --lr 10 --eps 1e-8; done
  ;;

main)  # Section IV, Table 3: five seeds, calibrated settings (utils.fwi.CALIBRATED)
  for model in over marm; do for m in l2 euclidean siamesefwi siamesefit mpbae; do for s in $(seeds 5); do
    fwi results/${model}_linear/$m/seed$s --model $model --method $m --seed $s
  done; done; done
  ;;

ablations)  # Sections IV-A2 and IV-B2, Figs 4 and 11 (settings of SiameseFit), three seeds
  for spec in identity:3 avgpool:3 randproj:3 maxpool:3 skip128:3 frozen:3 maxpool_l2:3; do
    m=${spec%:*}; for s in $(seeds ${spec#*:}); do fwi results/over_linear/$m/seed$s --model over --method $m --seed $s; done
  done
  for spec in maxpool:3 skip128:3 frozen:3 maxpool_shift:3 maxpool_top16:3 maxpool_abs:3 maxpool_mute:3; do
    m=${spec%:*}; for s in $(seeds ${spec#*:}); do fwi results/marm_linear/$m/seed$s --model marm --method $m --seed $s; done
  done
  for s in $(seeds 3); do fwi results/marm_linear/maxpool_lr20/seed$s --model marm --method maxpool --lr 20 --seed $s; done
  ;;

step)  # supplementary Fig. S2: Euclidean misfit with the constant step of the SiameseFWI implementation, 400 epochs
  for s in $(seeds 3); do
    fwi results/over_linear/euclidean_lr80_nodecay/seed$s --model over --method euclidean --lr 80 --decay-every 0 --epochs 400 --seed $s
  done
  ;;

hybrids)  # Section IV-B1, Table 4: SiameseFit on the Marmousi2 model from hybrid and smooth initial models, 3 seeds
  for init in smooth_top linear_top smooth; do for s in $(seeds 3); do
    fwi results/marm_${init}/siamesefit/seed$s --model marm --init $init --method siamesefit --seed $s
  done; done
  ;;

robustness)  # Section V-B, Table 6 and supplementary Fig. S8: noise of 10 dB per shot and low cut at 3 Hz for
             # SiameseFit, noise for the frozen network and its skip branch, three seeds, cuDNN autotuner off
  for model in over marm; do for test in noise lowcut; do for s in $(seeds 3); do
    opt="--noise-snr 10"; [ $test = lowcut ] && opt="--lowcut 3"
    fwi results/${model}_linear_${test}/siamesefit/seed$s --model $model --method siamesefit --seed $s $opt --no-cudnn-benchmark
  done; done; done
  for m in frozen skip128; do for s in $(seeds 3); do
    fwi results/over_linear_noise/$m/seed$s --model over --method $m --seed $s --noise-snr 10 --no-cudnn-benchmark
  done; done
  ;;

reproduction)  # supplementary Section S1, Table S1: water layer reset after each update (official SiameseFit
               # Marmousi2 notebook) and the settings of the MPBAE publication (constant step of 40 m/s, 200 epochs, eps 1e-8, with and without the 3 Hz low cut)
  for s in $(seeds 3); do fwi results/marm_linear/siamesefit_water_after/seed$s --model marm --method siamesefit --water-after-update --seed $s; done
  for m in mpbae euclidean; do
    fwi results/marm_linear/${m}_paper/seed0 --model marm --method $m --lr 40 --eps 1e-8 --decay-every 0 --epochs 200
    fwi results/marm_linear_lowcut/${m}_paper/seed0 --model marm --method $m --lr 40 --eps 1e-8 --decay-every 0 --epochs 200 --lowcut 3
  done
  ;;

analyses)  # Sections IV-A3 and IV-B3, supplementary Sections S3 and S4; seeds 0 to 2 (images of the article: seed 0)
  for model in over marm; do
    for an in identity cka depth training_stats; do  # these use all seeds themselves
      echo "$(date +%H:%M:%S) $an $model"; $PY -W ignore -m xai.$an --model $model
    done
    for s in 0 1 2; do
      for an in landscape adjoint occlusion fixed_data; do
        echo "$(date +%H:%M:%S) $an $model seed $s"; $PY -W ignore -m xai.$an --model $model --seed $s
      done
    done
  done
  for s in 0 1 2; do
    $PY -W ignore -m xai.cross_misfit --model marm --seed $s
    $PY -W ignore -m xai.updates --model marm --seed $s
    $PY -W ignore -m xai.updates --model over --parts A2 A4 --seed $s
  done
  ;;

figures)  # all figures and tables of the article, in results/figures and results/tables
  $PY -W ignore -m utils.figures
  ;;

*) echo "unknown block: $block"; exit 1 ;;
esac; done
echo "DONE $(date +%H:%M:%S)"
