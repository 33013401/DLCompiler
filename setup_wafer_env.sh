#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LLVM_COMMIT=7d5de3033187c8a3bb4d2e322f5462cdaf49808f
LLVM_SYSPATH=${LLVM_SYSPATH:-$WORK_ROOT/llvm-7d5de303-ubuntu-x64}
TX8_COMPILER_ROOT=${TX8_DEPS_ROOT:-$WORK_ROOT/tx8_deps}
ENV_FILE=${WAFER_ENV_FILE:-$SCRIPT_DIR/wafer_env.sh}

if [[ ! -x "$LLVM_SYSPATH/bin/llvm-config" ]]; then
    echo "ERROR: LLVM 22 package not found at $LLVM_SYSPATH" >&2
    exit 1
fi
if [[ $("$LLVM_SYSPATH/bin/llvm-config" --version) != "22.0.0git" ]]; then
    echo "ERROR: $LLVM_SYSPATH is not the required LLVM 22 package" >&2
    exit 1
fi
if [[ ! -f "$TX8_COMPILER_ROOT/include/instr_def.h" ]]; then
    echo "ERROR: TX8 compiler header not found under $TX8_COMPILER_ROOT/include" >&2
    exit 1
fi

cat >"$ENV_FILE" <<EOF
#!/usr/bin/env bash

export LLVM_COMMIT="$LLVM_COMMIT"
export LLVM_SYSPATH="$LLVM_SYSPATH"
export LLVM_BINARY_DIR="\$LLVM_SYSPATH/bin"
export LLVM_DIR="\$LLVM_SYSPATH/lib/cmake/llvm"
export MLIR_DIR="\$LLVM_SYSPATH/lib/cmake/mlir"
export WAFER_TX8_INCLUDE_DIR="$TX8_COMPILER_ROOT/include"
export PATH="\$LLVM_BINARY_DIR:\${PATH:-}"
export DICP_BACKEND=wafer
export USE_SIM_MODE=1
EOF
chmod +x "$ENV_FILE"

echo "Wafer compiler environment is ready:"
echo "  LLVM_SYSPATH:  $LLVM_SYSPATH"
echo "  TX8 headers:   $TX8_COMPILER_ROOT/include"
echo "  activation:    source $ENV_FILE"
