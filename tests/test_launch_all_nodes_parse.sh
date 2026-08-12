#!/bin/bash
# 用 PATH 前置 mock 掉 ssh/scp，验证 launch_all_nodes.sh 的参数解析 + rank 拉起命令
set -euo pipefail
BP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../experiments/batch_production" && pwd)"
WORK="$(mktemp -d)"

# mock ssh：预检永远成功(回显 OK)，拉起阶段把远程命令记录到文件
cat > "$WORK/ssh" <<'EOF'
#!/bin/bash
# 末参是远程命令串
CMD="${@: -1}"
echo "$CMD" >> "$MOCK_SSH_LOG"
# pgrep 计数返回 0；其余回显成功
case "$CMD" in
  *pgrep*) echo 0 ;;
  *) echo "[OK] mock" ;;
esac
EOF
chmod +x "$WORK/ssh"
cp "$WORK/ssh" "$WORK/scp"   # scp 同样吞掉

# 假 config.sh：不 source activate，只设最小变量
cat > "$WORK/config.sh" <<EOF
export NEW_BASE=/fake/base
export ENV_DIR=/fake/base/sana_wm_env
export PROJ_DIR=/fake/base/sana_wm_pipeline
export DATA_ROOT="\${DATA_ROOT:-/fake/data}"
export OUT_BASE="\${OUT_BASE:-/fake/out}"
EOF

# hostfile：3 个有效节点
printf 'nodeA slots=8\nnodeB slots=8\nnodeC slots=8\n' > "$WORK/hostfile"

export MOCK_SSH_LOG="$WORK/ssh.log"; : > "$MOCK_SSH_LOG"
# 用 mock config 覆盖：把 launch_all_nodes 里的 source 指到我们的假 config
SCRIPT="$WORK/launch_all_nodes.sh"
sed "s#source \"\$SCRIPT_DIR/config.sh\"#source \"$WORK/config.sh\"#" \
    "$BP/launch_all_nodes.sh" > "$SCRIPT"

PATH="$WORK:$PATH" bash "$SCRIPT" wds-sekai-real-walking-hq /my/out "$WORK/hostfile" \
    > "$WORK/out.log" 2>&1 || true

# 断言1：拉起命令含 --groups wds-sekai-real-walking-hq
grep -q -- "--groups wds-sekai-real-walking-hq" "$MOCK_SSH_LOG" \
    || { echo "FAIL: 未透传 --groups dataset"; cat "$WORK/out.log"; exit 1; }
# 断言2：远程 export OUT_BASE=/my/out
grep -q "OUT_BASE=/my/out\|OUT_BASE='/my/out'" "$MOCK_SSH_LOG" \
    || { echo "FAIL: 未 export 自定义 OUT_BASE"; exit 1; }
# 断言3：3 节点 → 出现 rank 0/1/2 且 NUM_NODES=3
grep -q "run_groups_sequential.sh --groups wds-sekai-real-walking-hq 0 3" "$MOCK_SSH_LOG" \
    || { echo "FAIL: rank0/NUM_NODES=3 拉起命令不对"; grep run_groups "$MOCK_SSH_LOG"; exit 1; }
grep -q " 2 3$" "$MOCK_SSH_LOG" \
    || { echo "FAIL: rank2 拉起命令不对"; exit 1; }
# 断言4：不得把 master 的 LD_LIBRARY_PATH 注入远程
grep -q 'export LD_LIBRARY_PATH="' "$MOCK_SSH_LOG" \
    && { echo "FAIL: 仍在向远程注入 master LD_LIBRARY_PATH"; exit 1; }
# 断言4b：不得把 master 的 PATH 注入远程
grep -q 'export PATH="' "$MOCK_SSH_LOG" \
    && { echo "FAIL: 仍在向远程注入 master PATH"; exit 1; }
# 断言5：远程必须 unset CUDA_VISIBLE_DEVICES
grep -q "unset CUDA_VISIBLE_DEVICES" "$MOCK_SSH_LOG" \
    || { echo "FAIL: 远程未 unset CUDA_VISIBLE_DEVICES"; exit 1; }
echo "PASS: launch_all_nodes 参数解析 + rank + 环境修复 全部正确"
