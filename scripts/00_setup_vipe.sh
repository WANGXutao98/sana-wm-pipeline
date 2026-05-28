#!/usr/bin/env bash
# Clone VIPE (Apache-2.0), install it, and apply SANA-WM patches.
#
# Idempotent: re-running keeps the working tree pristine.
# Paper: SANA-WM (arXiv:2605.15178), App. B.1.
#
# After running this script:
#   1.  `vipe infer` is on PATH.
#   2.  vipe/priors/depth/__init__.py registers "pi3x_moge2" backend.
#   3.  vipe/configs/pipeline/sana_wm_pose_only.yaml is installed.
#
# To use Pi3X+MoGe-2 (full paper-aligned) also set:
#   export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x
#   export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2
# and change keyframe_depth/depth_align_model in sana_wm_pose_only.yaml
# from unidepth-l to pi3x_moge2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIPE_DIR="${REPO_ROOT}/third_party/vipe"
PATCH_DIR="${REPO_ROOT}/third_party/vipe_patch"

# ---------------------------------------------------------------------------
# 1. Clone if missing.
# ---------------------------------------------------------------------------
if [ ! -d "${VIPE_DIR}/.git" ]; then
  mkdir -p "$(dirname "${VIPE_DIR}")"
  echo "[setup_vipe] cloning nv-tlabs/vipe -> ${VIPE_DIR}"
  git clone --depth 1 https://github.com/nv-tlabs/vipe.git "${VIPE_DIR}"
else
  echo "[setup_vipe] vipe already present at ${VIPE_DIR}"
fi

# ---------------------------------------------------------------------------
# 2. Install VIPE (editable).  Compile CUDA extensions if possible.
# ---------------------------------------------------------------------------
echo "[setup_vipe] installing vipe..."
( cd "${VIPE_DIR}" && PIP_USER=false pip install --no-user -e ".[all]" 2>&1 \
    || PIP_USER=false pip install --no-user -e . 2>&1 )
echo "[setup_vipe] vipe installed."

# ---------------------------------------------------------------------------
# 3. Register Pi3X+MoGe2 depth backend in VIPE's make_depth_model factory.
#    We patch vipe/priors/depth/__init__.py to add the "pi3x_moge2" branch.
# ---------------------------------------------------------------------------
DEPTH_INIT="${VIPE_DIR}/vipe/priors/depth/__init__.py"

if ! grep -q "pi3x_moge2" "${DEPTH_INIT}"; then
  echo "[setup_vipe] registering pi3x_moge2 in ${DEPTH_INIT}"
  # Append registration before the final else clause.
  python3 - <<'PYEOF'
import re, pathlib

path = pathlib.Path("${DEPTH_INIT}")
code = path.read_text()

insert = '''
    elif model_name == "pi3x_moge2":
        from .pi3x_moge2 import Pi3XMoGe2DepthModel
        return Pi3XMoGe2DepthModel()
'''
# Insert before the final else clause
code = code.replace(
    "    else:\n        raise ValueError(f\"Unknown depth model: {model}\")",
    insert + "    else:\n        raise ValueError(f\"Unknown depth model: {model}\")",
)
path.write_text(code)
print("  patched:", path)
PYEOF
  # Fix the path variable expansion inside the heredoc
  python3 - "${DEPTH_INIT}" <<'PYEOF'
import re, pathlib, sys

path = pathlib.Path(sys.argv[1])
code = path.read_text()

insert = '''
    elif model_name == "pi3x_moge2":
        from .pi3x_moge2 import Pi3XMoGe2DepthModel
        return Pi3XMoGe2DepthModel()
'''
if "pi3x_moge2" not in code:
    code = code.replace(
        "    else:\n        raise ValueError(f\"Unknown depth model: {model}\")",
        insert + "    else:\n        raise ValueError(f\"Unknown depth model: {model}\")",
    )
    path.write_text(code)
    print("[setup_vipe] patched:", path)
else:
    print("[setup_vipe] already patched:", path)
PYEOF
else
  echo "[setup_vipe] pi3x_moge2 already registered in ${DEPTH_INIT}"
fi

# ---------------------------------------------------------------------------
# 4. Copy depth backend + pipeline config into VIPE tree.
# ---------------------------------------------------------------------------
DEPTH_DIR="${VIPE_DIR}/vipe/priors/depth"
mkdir -p "${DEPTH_DIR}"

echo "[setup_vipe] copying pi3x_moge2 depth backend..."
cp "${PATCH_DIR}/depth_backend_pi3x_moge2.py" "${DEPTH_DIR}/pi3x_moge2.py"

echo "[setup_vipe] copying sana_wm_pose_only pipeline config..."
cp "${PATCH_DIR}/sana_wm_pose_only.yaml" "${VIPE_DIR}/configs/pipeline/sana_wm_pose_only.yaml"

# ---------------------------------------------------------------------------
# 5. Verify vipe infer is runnable.
# ---------------------------------------------------------------------------
echo "[setup_vipe] verifying vipe CLI..."
vipe --version 2>&1 || python -m vipe.cli --version 2>&1 || echo "[setup_vipe] WARNING: vipe CLI not on PATH; activate conda env first."

echo "[setup_vipe] done."
echo ""
echo "Smoke test with stub depth (no Pi3X weights needed):"
echo "  bash scripts/e2e_smoke.sh"
echo ""
echo "Full pipeline (Pi3X + MoGe-2):"
echo "  export SANA_WM_PI3X_WEIGHTS=/mnt/afs/davidwang/models/pi3x"
echo "  export SANA_WM_MOGE2_WEIGHTS=/mnt/afs/davidwang/models/moge2"
echo "  Update sana_wm_pose_only.yaml: keyframe_depth + depth_align_model -> pi3x_moge2"
echo "  SANA_WM_POSE_STUB=0 bash scripts/e2e_smoke.sh"
