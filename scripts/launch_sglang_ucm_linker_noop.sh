#!/usr/bin/env bash
# Single-host Linux + NVIDIA launcher for the UCM L1 linker patch.
# The no-op linker validates SGLang patching and linker attachment only: it
# always misses external cache and never persists KV data.
#
# Edit the configuration block below for the target host and model. Extra
# SGLang arguments may be supplied after --; managed linker/network arguments
# are rejected. Ctrl-C stops the SGLang process group and retains its log.

set -Eeuo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    sed -n '2,10p' "$0"
    exit 0
fi
[[ $(uname -s) == Linux ]] || { echo 'Run on a Linux GPU host.' >&2; exit 1; }

# ---------------------------------------------------------------------------
# Local configuration
# ---------------------------------------------------------------------------
PYTHON=python3
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
TP_SIZE=1
SGLANG_HOST=127.0.0.1
SGLANG_PORT=30000
MAX_TOTAL_TOKENS=4096
MAX_RUNNING_REQUESTS=1
MEM_FRACTION_STATIC=0.7
START_TIMEOUT=600
LOG_DIR=./logs/ucm-linker-noop-$(date +%Y%m%d-%H%M%S)-$$
# ---------------------------------------------------------------------------

[[ ${1:-} != -- ]] || shift
for executable in "$PYTHON" setsid curl; do
    command -v "$executable" >/dev/null || {
        echo "Missing executable: $executable" >&2
        exit 1
    }
done
for number in "$TP_SIZE" "$SGLANG_PORT" "$MAX_TOTAL_TOKENS" \
    "$MAX_RUNNING_REQUESTS" "$START_TIMEOUT"; do
    [[ $number =~ ^[1-9][0-9]*$ ]] || {
        echo 'TP_SIZE, ports, limits, and timeout must be positive integers.' >&2
        exit 1
    }
done
for argument in "$@"; do
    case "$argument" in
        --model-path|--model-path=*|--tp-size|--tp-size=*|--host|--host=*|--port|--port=*|--enable-hierarchical-cache|--enable-lmcache|--enable-unified-cache-external-linker|--unified-cache-external-linker-backend|--unified-cache-external-linker-backend=*)
            echo "Conflicting or managed argument: $argument" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$LOG_DIR"
LOG_DIR=$(cd "$LOG_DIR" && pwd)

"$PYTHON" - "$TP_SIZE" "$SGLANG_PORT" <<'PY'
import socket
import sys
import torch

assert torch.cuda.is_available() and torch.version.hip is None, 'Requires NVIDIA CUDA GPUs'
assert torch.cuda.device_count() >= int(sys.argv[1]), 'Insufficient visible GPUs for TP_SIZE'
port = int(sys.argv[2])
assert 0 < port < 65536, f'Invalid port: {port}'
with socket.socket() as sock:
    sock.bind(('0.0.0.0', port))
PY
"$PYTHON" -m sglang.launch_server --help >"$LOG_DIR/sglang-help.log" 2>&1
grep -q -- '--enable-unified-cache-external-linker' "$LOG_DIR/sglang-help.log" || {
    echo "Installed SGLang lacks external linker support; see $LOG_DIR/sglang-help.log" >&2
    exit 1
}

export ENABLE_UCM_SGLANG_PATCH=1
export UCM_SGLANG_VERSION=0.5.20
export UCM_SGLANG_ALLOW_NOOP_LINKER=1

pid=''
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n $pid ]]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    echo "Stopped SGLang. Log: $LOG_DIR/sglang.log"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size "$TP_SIZE" \
    --page-size 64 \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --enable-cache-report \
    --enable-unified-cache-external-linker \
    --unified-cache-external-linker-backend ucm \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    "$@" >"$LOG_DIR/sglang.log" 2>&1 &
pid=$!
echo "Started SGLang (PID $pid), log: $LOG_DIR/sglang.log"

health_host=$SGLANG_HOST
[[ $health_host != 0.0.0.0 ]] || health_host=127.0.0.1
deadline=$((SECONDS + START_TIMEOUT))
until curl --noproxy '*' -fsS --max-time 2 \
    "http://$health_host:$SGLANG_PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo 'SGLang exited; recent log:' >&2
        tail -n 40 "$LOG_DIR/sglang.log" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for SGLang; log: $LOG_DIR/sglang.log" >&2
        exit 1
    fi
    sleep 2
done

echo "SGLang ready: http://$health_host:$SGLANG_PORT"
echo "No-op UCM linker active: no external KV reads or writes occur."
wait "$pid"
