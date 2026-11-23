#!/bin/bash
#
# Installer of the environment of this repository, with the packages of the FWIGAN environment of the
# SiameseFWI and SiameseFit repositories (PyTorch 1.7.1 + CUDA 11.0 + Deepwave 0.0.8 built from ./deepwave-old)
#
# Run: ./install_env.sh
#
# Optional variables:
#   CUDA_HOME : CUDA 11.x toolkit used to compile Deepwave (default: /usr/local/cuda-11.8)
#

set -e
cd "$(dirname "$0")"

echo 'Creating Package environment'

# create conda env
conda env create -f xai-fwi.yml
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate xai-fwi
echo 'Created and activated environment:' $(which python)

# Check torch works as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

# Install the old Deepwave version (0.0.8), as in the official repositories.
# The CUDA extension must be compiled with a CUDA 11.x toolkit to match torch.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
# nvcc must use the gcc 11 shipped with the environment (CXX is set by conda activate)
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
echo 'Installing Deepwave 0.0.8 from ./deepwave-old using CUDA at' "$CUDA_HOME"
cd ./deepwave-old/
rm -rf build  # drop objects left by a previous build
python setup.py install
cd ..

# Check deepwave works as expected (run outside deepwave-old to test the installed copy)
echo 'Checking deepwave on the GPU...'
python - <<'EOF'
import torch
import deepwave

# small 2D forward modelling on the GPU
dev = torch.device('cuda:0')
model = torch.ones(50, 50, device=dev) * 1500.0
prop = deepwave.scalar.Propagator({'vp': model}, 5.0)
src_amp = deepwave.wavelets.ricker(10.0, 100, 0.001, 0.1).reshape(-1, 1, 1).to(dev)
src_loc = torch.tensor([[[25.0, 125.0]]], device=dev)
rec_loc = torch.tensor([[[25.0, 200.0]]], device=dev)
rec = prop(src_amp, src_loc, rec_loc, 0.001)
print('deepwave', deepwave.__file__)
print('receiver data:', tuple(rec.shape), 'max amplitude:', float(rec.abs().max()))
EOF

echo 'Done!'
