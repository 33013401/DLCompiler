#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/init_tx81_env.sh"
export USE_SIM_MODE=0 WAFER_ENABLE_RUNTIME=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TX81_WORKSPACE_ROOT/build/wafer-acceptance/cache}
export TRITON_DUMP_PATH=${TRITON_DUMP_PATH:-$TX81_WORKSPACE_ROOT/build/wafer-acceptance/dump}
RESULT_DIR=${WAFER_RESULT_DIR:-$TX81_WORKSPACE_ROOT/build/wafer-acceptance}
mkdir -p "$RESULT_DIR" "$TRITON_CACHE_DIR" "$TRITON_DUMP_PATH"
cd "$TX81_WORKSPACE_ROOT"
LOG_PATH="$RESULT_DIR/acceptance-$(date +%Y%m%d-%H%M%S-%N).log"
echo "Acceptance log: $LOG_PATH"
timeout --signal=TERM --kill-after=5s "${WAFER_TEST_TIMEOUT:-120}s" \
    "$PYTHON" "$SCRIPT_DIR/verify_wafer_runtime.py" "$@" 2>&1 | tee "$LOG_PATH"
