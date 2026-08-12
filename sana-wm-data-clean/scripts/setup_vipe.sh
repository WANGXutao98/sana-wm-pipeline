#!/bin/bash
# Build the real VIPE pose engine (nv-tlabs/vipe) in its own venv.
# VIPE is a DROID-SLAM derivative needing cu128/torch2.7+ and compiles CUDA
# extensions; the container's torch is 2.5/cu124, so VIPE gets an isolated venv
# (Pi3/MoGe also installed there for the fused-depth backend integration later).
# Editable clone so the per-frame-intrinsics BA modification can be patched in.
REPO_ROOT="${SANA_WM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
THIRD_PARTY="$REPO_ROOT/third_party"
VENV="${VIPE_VENV:-$REPO_ROOT/.venv-vipe}"   # set to fast local storage if available
mkdir -p "$THIRD_PARTY"

# Preflight: the --system-site-packages venv below inherits the container's torch. This
# assumes an NVIDIA NGC PyTorch image (torch 2.8 / cu12.9). On any other host VIPE's CUDA
# ext won't match — fail loudly here instead of building a subtly-broken env (see README).
python3 -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.8') else 1)" 2>/dev/null || {
  echo "PREFLIGHT FAIL: need container torch 2.8.x (NGC image); got '$(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)'." >&2; exit 30; }

[ -d "$THIRD_PARTY/vipe" ] || git clone --depth 1 https://github.com/nv-tlabs/vipe.git "$THIRD_PARTY/vipe" || exit 31

# IMPORTANT (NGC container): do NOT install a separate torch. VIPE's pyproject
# leaves torch UNPINNED (match-runtime), and this container already ships
# torch 2.8.0a0 + CUDA 12.9 (>= VIPE's torch2.7+ requirement, matches nvcc 12.9).
# A `--system-site-packages` venv inherits that torch, so VIPE's CUDA ext compiles
# against it. Trying to `pip install torch==2.7.1` instead FAILS: the NGC image
# pins torch via PIP_CONSTRAINT=/etc/pip/constraint.txt (torch==2.8.0a0...nv25.6),
# which a fresh venv still inherits -> ResolutionImpossible. Inheriting the
# container torch sidesteps the whole conflict (verified: vipe builds + SLAM runs).
python3 -m venv --system-site-packages "$VENV" || exit 32
VENV_PIP="$VENV/bin/pip"
$VENV_PIP install -U pip wheel setuptools ninja cmake pybind11 || exit 33
"$VENV/bin/python" -c "import torch;print('venv torch',torch.__version__,'cuda',torch.version.cuda)" || exit 35

echo "=== build VIPE (editable, compiles CUDA ext for Hopper sm_90, against container torch) ==="
cd "$THIRD_PARTY/vipe"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=32
$VENV_PIP install -e . --no-build-isolation 2>&1 | tail -45 || exit 36

# CRITICAL (numpy ABI): the container torch 2.8 is compiled against numpy 1.x, but
# VIPE's deps (opencv-python>=4.13 requires numpy>=2) drag numpy 2.x into the venv,
# which silently breaks torch's numpy bridge -> every `np.diag(tensor)` /
# `tensor.__array__()` raises "RuntimeError: Numpy is not available" at the geocalib
# step, failing EVERY default-mode (VIPE) clip. Pin numpy<2 AFTER the VIPE install so
# it overrides whatever opencv pulled. The pip
# resolver will warn opencv "wants numpy>=2" — harmless, cv2 runs fine on 1.26.)
export PIP_CONSTRAINT=
$VENV_PIP install "numpy<2" || exit 39

echo "=== install SANA-WM pipeline profile (so 'vipe infer -p sanawm' resolves) ==="
cp "$REPO_ROOT/vipe_patches/sanawm_pipeline.yaml" "$THIRD_PARTY/vipe/configs/pipeline/sanawm.yaml" || exit 38

echo "=== verify import + torch<->numpy bridge (the thing that silently broke the pods) ==="
"$VENV/bin/python" -c "import vipe; print('vipe import OK')" || exit 37
"$VENV/bin/python" -c "import torch,numpy as np; np.diag(torch.eye(3).numpy()); print('numpy bridge OK', np.__version__)" || exit 40

echo "VIPE_SETUP_DONE"
touch "$REPO_ROOT/.vipe_setup_done"
