#!/usr/bin/env bash
# Start SGLang through the UCM L1 linker patch without a real UCM Store.
# This exercises patch application and linker attachment only: every external
# lookup misses and no KV data is written outside SGLang.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-path> [additional sglang options]" >&2
    exit 2
fi

model_path=$1
shift

export ENABLE_UCM_SGLANG_PATCH=1
export UCM_SGLANG_VERSION=0.5.20
export UCM_SGLANG_ALLOW_NOOP_LINKER=1

exec python -m sglang.launch_server \
    --model-path "$model_path" \
    --enable-unified-cache-external-linker \
    --unified-cache-external-linker-backend ucm \
    "$@"
