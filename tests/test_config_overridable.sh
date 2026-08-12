#!/bin/bash
# 断言 config.sh 尊重外部预设的 OUT_BASE / DATA_ROOT
set -euo pipefail
CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../experiments/batch_production" && pwd)/config.sh"

# 屏蔽 activate（测试机无该环境）：用一个假的 NEW_BASE 指向临时空 activate
TMP="$(mktemp -d)"
printf '#!/bin/bash\n' > "$TMP/activate_sana_wm.sh"

export NEW_BASE="$TMP"
export OUT_BASE="/custom/out/path"
export DATA_ROOT="/custom/data/path"
# shellcheck disable=SC1090
source "$CFG" >/dev/null 2>&1 || true

[[ "$OUT_BASE" == "/custom/out/path" ]]   || { echo "FAIL: OUT_BASE 被覆盖成 $OUT_BASE"; exit 1; }
[[ "$DATA_ROOT" == "/custom/data/path" ]] || { echo "FAIL: DATA_ROOT 被覆盖成 $DATA_ROOT"; exit 1; }
echo "PASS: config.sh 尊重外部 OUT_BASE / DATA_ROOT"
