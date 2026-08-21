#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRITON_DIR="$SCRIPT_DIR/third_party/triton"
WAFER_DIR="$SCRIPT_DIR/third_party/wafer"
BUILD_DIR=${WAFER_BUILD_DIR:-$WAFER_DIR/build_manual}
INSTALL_DIR="$BUILD_DIR/install"
WHEEL_DIR="$BUILD_DIR/wheel"
BUILD_TYPE=${BUILD_TYPE:-Release}
WAFER_TX8_INCLUDE_DIR=${WAFER_TX8_INCLUDE_DIR:-$SCRIPT_DIR/../../tx8_deps/include}

if [[ ${1:-} == "--clean" ]]; then
    rm -rf "$BUILD_DIR"
elif [[ -n ${1:-} ]]; then
    echo "Usage: $0 [--clean]" >&2
    exit 2
fi

: "${LLVM_SYSPATH:?Run 'source llvm22_env.sh' before building Wafer}"
: "${LLVM_DIR:?LLVM_DIR is not set}"
: "${MLIR_DIR:?MLIR_DIR is not set}"

if [[ ! -f "$TRITON_DIR/CMakeLists.txt" ]]; then
    echo "ERROR: Triton is not initialized at $TRITON_DIR" >&2
    exit 1
fi

if [[ ! -f "$WAFER_TX8_INCLUDE_DIR/instr_def.h" ]]; then
    echo "ERROR: instr_def.h not found under $WAFER_TX8_INCLUDE_DIR" >&2
    echo "Set WAFER_TX8_INCLUDE_DIR to the TX8 compiler header directory" >&2
    exit 1
fi

bash "$SCRIPT_DIR/scripts/apply_wafer_triton_patches.sh"

if [[ "$(tr -d '[:space:]' < "$TRITON_DIR/cmake/llvm-hash.txt")" != "${LLVM_COMMIT:-}" ]]; then
    echo "ERROR: active LLVM does not match the Triton LLVM pin" >&2
    exit 1
fi

plugin_dirs="$SCRIPT_DIR;$WAFER_DIR"
cmake_args=(
    -S "$TRITON_DIR"
    -B "$BUILD_DIR"
    -G Ninja
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_C_COMPILER="${CC:-/usr/bin/cc}"
    -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/c++}"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
    -DTRITON_WHEEL_DIR="$WHEEL_DIR"
    -DLLVM_DIR="$LLVM_DIR"
    -DMLIR_DIR="$MLIR_DIR"
    -DLLVM_SYSPATH="$LLVM_SYSPATH"
    -DTRITON_BUILD_PYTHON_MODULE=ON
    -DTRITON_PLUGIN_DIRS="$plugin_dirs"
    -DTRITON_CODEGEN_BACKENDS=nvidia
    -DTRITON_BUILD_GLUON_IR=OFF
    -DTRITON_BUILD_PROTON=OFF
    -DTRITON_BUILD_UT=OFF
    -DTRITON_BUILD_TUTORIALS=OFF
    -DDICP_WAFER_COMBINED_BUILD=ON
    -DWAFER_USE_EXTERNAL_TLE=OFF
    -DWAFER_TX8_INCLUDE_DIR="$WAFER_TX8_INCLUDE_DIR"
)

if [[ -n ${TX8_DEPS_ROOT:-} && -d $TX8_DEPS_ROOT ]]; then
    cmake_args+=("-DTX8_DEPS_ROOT=$TX8_DEPS_ROOT")
else
    unset TX8_DEPS_ROOT
    cmake_args+=("-U" "TX8_DEPS_ROOT")
fi

export DICP_BACKEND=wafer
export TRITON_OFFLINE_BUILD=ON

mkdir -p "$WHEEL_DIR"

echo "Configuring Triton with plugins: $plugin_dirs"
echo "Kernel acceptance backend: $DICP_BACKEND"
cmake "${cmake_args[@]}"

cmake --build "$BUILD_DIR" --target wafer-opt libtriton.so --parallel

echo "Wafer build completed:"
echo "  wafer-opt:   $BUILD_DIR/third_party/wafer/bin/wafer-opt"
echo "  libtriton.so: $BUILD_DIR/libtriton.so"