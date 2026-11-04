#!/usr/bin/env bash
set -euo pipefail
CV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-/home/delen/miniforge3/bin/conda}"
export CONDA_PKGS_DIRS="$CV_DIR/cache/conda"
export PIP_CACHE_DIR="$CV_DIR/cache/pip"
export HF_HOME="$CV_DIR/cache/huggingface"
mkdir -p "$CV_DIR/cache" "$CV_DIR/checkpoints" "$CV_DIR/upstream" "$CV_DIR/logs"
if [ ! -x "$CV_DIR/.conda/bin/python" ]; then
  "$CONDA_BIN" create --yes --prefix "$CV_DIR/.conda" --override-channels -c conda-forge python=3.12 pip
fi
"$CV_DIR/.conda/bin/python" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$CV_DIR/.conda/bin/python" -m pip install numpy==1.26.4 pillow==11.3.0 scipy==1.16.3 pycocotools==2.0.11 hydra-core==1.3.2 tqdm==4.67.1
if [ ! -d "$CV_DIR/upstream/sam2/.git" ]; then
  git clone https://github.com/facebookresearch/sam2.git "$CV_DIR/upstream/sam2"
fi
git -C "$CV_DIR/upstream/sam2" checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 "$CV_DIR/.conda/bin/python" -m pip install -e "$CV_DIR/upstream/sam2" --no-build-isolation
if [ ! -s "$CV_DIR/checkpoints/sam2.1_hiera_small.pt" ]; then
  curl --fail --location --retry 3 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt --output "$CV_DIR/checkpoints/sam2.1_hiera_small.pt.tmp"
  mv "$CV_DIR/checkpoints/sam2.1_hiera_small.pt.tmp" "$CV_DIR/checkpoints/sam2.1_hiera_small.pt"
fi
"$CV_DIR/.conda/bin/python" -m pip freeze | sed "s|^-e $CV_DIR/upstream/sam2|-e ./upstream/sam2|" > "$CV_DIR/requirements.lock.txt"
git -C "$CV_DIR/upstream/sam2" rev-parse HEAD > "$CV_DIR/upstream-sam2-revision.txt"
"$CV_DIR/.conda/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
