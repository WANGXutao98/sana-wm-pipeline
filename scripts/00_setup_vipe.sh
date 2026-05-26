#!/usr/bin/env bash
# Clone VIPE (Apache-2.0) and apply our Pi3X + MoGe-2 patches.
#
# Idempotent: re-running keeps the working tree pristine.
# Paper: SANA-WM (arXiv:2605.15178), App. B.1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIPE_DIR="${REPO_ROOT}/third_party/vipe"
PATCH_DIR="${REPO_ROOT}/third_party/vipe_patch"

# 1. Clone if missing.  Real VIPE repo is at nv-tlabs/vipe (verified).
if [ ! -d "${VIPE_DIR}/.git" ]; then
  mkdir -p "$(dirname "${VIPE_DIR}")"
  echo "[setup_vipe] cloning nv-tlabs/vipe -> ${VIPE_DIR}"
  git clone --depth 1 https://github.com/nv-tlabs/vipe.git "${VIPE_DIR}"
else
  echo "[setup_vipe] vipe already present at ${VIPE_DIR}"
fi

# 2. Install editable.
( cd "${VIPE_DIR}" && pip install --no-user -e . )

# 3. Drop our patches into the upstream tree.  Subdirectory names are best-effort
# and may need adjustment once the upstream layout is inspected on this host.
mkdir -p "${VIPE_DIR}/vipe/backends" "${VIPE_DIR}/vipe/optim"
cp "${PATCH_DIR}/depth_backend_pi3x_moge2.py" "${VIPE_DIR}/vipe/backends/"
cp "${PATCH_DIR}/ba_per_frame_intrinsics.py"  "${VIPE_DIR}/vipe/optim/"

echo "[setup_vipe] patches applied:"
ls -la "${VIPE_DIR}/vipe/backends/depth_backend_pi3x_moge2.py"
ls -la "${VIPE_DIR}/vipe/optim/ba_per_frame_intrinsics.py"
