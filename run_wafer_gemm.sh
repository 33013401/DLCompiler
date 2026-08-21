#!/usr/bin/env bash

set -euo pipefail

# 所有相对路径都以当前脚本所在的 DLCompiler 仓库根目录为基准。
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${WAFER_BUILD_DIR:-$SCRIPT_DIR/third_party/wafer/build_manual}
# 默认使用已验证过的独立 Python 环境；也可以通过 PYTHON 覆盖。
PYTHON=${PYTHON:-/tmp/wafer-venv/bin/python}
# IR 和目标文件默认写到仓库外，避免污染 Git 工作树。
OUTPUT_DIR=${WAFER_GEMM_OUTPUT_DIR:-/tmp/wafer-gemm-run}
REBUILD=0

usage() {
    cat <<EOF
Usage: $0 [--rebuild] [--output-dir DIR]

Compile the built-in Triton BF16 GEMM through Wafer and dump every IR stage.

Options:
  --rebuild         Rebuild and reinstall the Wafer compiler before running
  --output-dir DIR  Artifact directory (default: /tmp/wafer-gemm-run)
  -h, --help        Show this help
EOF
}

# 解析可选参数：强制重建，或者指定 GEMM 编译产物目录。
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            REBUILD=1
            shift
            ;;
        --output-dir)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --output-dir requires a directory" >&2
                exit 2
            fi
            OUTPUT_DIR=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python environment not found: $PYTHON" >&2
    echo "Create it with: python3 -m venv /tmp/wafer-venv" >&2
    echo "Or set PYTHON=/path/to/python." >&2
    exit 1
fi

# 检查 LLVM 22 和 TX8 头文件，并生成/加载 wafer_env.sh。
cd "$SCRIPT_DIR"
bash setup_wafer_env.sh
source wafer_env.sh

# 同时满足以下条件才复用现有安装：
# 1. C++ 编译器产物存在；2. Python 环境能发现 dicp_triton backend。
compiler_ready=0
if [[ -x "$BUILD_DIR/third_party/wafer/bin/wafer-opt" && -f "$BUILD_DIR/libtriton.so" ]]; then
    # 必须从 /tmp 导入，避免仓库中的 triton/ 源码目录遮蔽已安装的 wheel。
    if (cd /tmp && DICP_BACKEND=wafer USE_SIM_MODE=1 "$PYTHON" -c \
        'from triton.backends import backends; assert "dicp_triton" in backends') \
        >/dev/null 2>&1; then
        compiler_ready=1
    fi
fi

# --rebuild 会清理 CMake build；普通模式只在编译器不可用时构建和安装。
if [[ $REBUILD == 1 || $compiler_ready == 0 ]]; then
    if [[ $REBUILD == 1 ]]; then
        bash compile_wafer.sh --clean
        PYTHON="$PYTHON" bash install_wafer.sh --skip-build
    else
        PYTHON="$PYTHON" bash install_wafer.sh
    fi
fi

# 每次清理旧 dump，确保输出全部来自本次真实 Triton GEMM 编译。
rm -rf "$OUTPUT_DIR"
WAFER_OPT_PATH="$BUILD_DIR/third_party/wafer/bin/wafer-opt" \
    "$PYTHON" scripts/verify_wafer_acceptance.py --output-dir "$OUTPUT_DIR"

# wheel 只在发生安装时生成；快速路径可能只复用已有 wheel。
WHEEL_DIR=${WAFER_WHEEL_DIR:-$BUILD_DIR/wheel}
wheel=$(find "$WHEEL_DIR" -maxdepth 1 -name 'triton-*.whl' -type f -print -quit 2>/dev/null || true)

# 最后集中打印编译器本体和各级 GEMM IR/object 的固定位置。
cat <<EOF

Wafer one-command run completed.

Compiler artifacts:
  wafer-opt:    $BUILD_DIR/third_party/wafer/bin/wafer-opt
  libtriton.so: $BUILD_DIR/libtriton.so
  wheel:        ${wheel:-not built in this run}

GEMM artifacts:
  TTIR:         $OUTPUT_DIR/ttir.mlir
  CoreIR:       $OUTPUT_DIR/coreir.mlir
  TXIR:         $OUTPUT_DIR/txir.mlir
  LLVM dialect: $OUTPUT_DIR/llvm.mlir
  LLVM IR:      $OUTPUT_DIR/kernel.ll
  object:       $OUTPUT_DIR/kernel.o
EOF