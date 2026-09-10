#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this script instead of executing it:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

TX81_WORKSPACE_ROOT=${TX81_WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_ROOT=${CONDA_ROOT:-$TX81_WORKSPACE_ROOT/miniconda3}
ENV_NAME=${ENV_NAME:-wafer310}

if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Conda not found at $CONDA_ROOT" >&2
    return 1
fi
if [[ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]]; then
    echo "ERROR: Conda environment not found: $ENV_NAME" >&2
    return 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export TX81_WORKSPACE_ROOT
export CONDA_ROOT
export ENV_NAME
export REPO_DIR=${REPO_DIR:-$TX81_WORKSPACE_ROOT/DLCompiler}
export DEPS_ROOT=${DEPS_ROOT:-$TX81_WORKSPACE_ROOT/deps}
export PACKAGE_ROOT=${PACKAGE_ROOT:-$TX81_WORKSPACE_ROOT/packages}
export PYTHON="$CONDA_PREFIX/bin/python"
export LLVM_COMMIT=${LLVM_COMMIT:-7d5de3033187c8a3bb4d2e322f5462cdaf49808f}
export LLVM_SYSPATH=${LLVM_SYSPATH:-$DEPS_ROOT/llvm-7d5de303-ubuntu-x64}
export LLVM_BINARY_DIR="$LLVM_SYSPATH/bin"
export LLVM_DIR="$LLVM_SYSPATH/lib/cmake/llvm"
export MLIR_DIR="$LLVM_SYSPATH/lib/cmake/mlir"
export TX8_DEPS_ROOT=${TX8_DEPS_ROOT:-$DEPS_ROOT/tx8_deps}
export KUIPER_ROOT=${KUIPER_ROOT:-/usr/local/kuiper}
export WAFER_TX8_INCLUDE_DIR=${WAFER_TX8_INCLUDE_DIR:-$TX8_DEPS_ROOT/include}
export TX8_YOC_RT_THREAD_SMP=${TX8_YOC_RT_THREAD_SMP:-$TX8_DEPS_ROOT/tx8-yoc-rt-thread-smp}
export XUANTIE_NAME=${XUANTIE_NAME:-$TX8_DEPS_ROOT/Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2}
export WAFER_BUILD_DIR=${WAFER_BUILD_DIR:-$TX81_WORKSPACE_ROOT/build/wafer}
export WAFER_RUNTIME_LIB_DIR=${WAFER_RUNTIME_LIB_DIR:-$WAFER_BUILD_DIR/third_party/wafer/crt/lib}
export DICP_BACKEND=wafer
export USE_SIM_MODE=${USE_SIM_MODE:-1}
# This workspace uses Kuiper 1.4 firmware with the RCS device logging API.
export WAFER_DEVICE_LOG_ABI=${WAFER_DEVICE_LOG_ABI:-rcs}
export PATH="$LLVM_BINARY_DIR:$PATH"
export LD_LIBRARY_PATH="$KUIPER_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

hash -r 2>/dev/null || true

echo "TX81 build environment activated"
echo "  Conda: $CONDA_DEFAULT_ENV"
echo "  Python: $PYTHON"
echo "  LLVM: $LLVM_SYSPATH"
echo "  TX8: $TX8_DEPS_ROOT"
echo "  Kuiper: $KUIPER_ROOT"
