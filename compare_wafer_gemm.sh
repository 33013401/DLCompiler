#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-/tmp/wafer-venv/bin/python}
OUTPUT_DIR=${WAFER_AB_OUTPUT_DIR:-/tmp/wafer-gemm-strict-ab}
INPUT_DIR=${WAFER_GEMM_INPUT_DIR:-/tmp/wafer-gemm-ab-input}
BASELINE_OPT=${WAFER_BASELINE_OPT:-$WORK_ROOT/DLCompiler/third_party/wafer/build_manual/install/bin/wafer-opt}
NEW_OPT=${WAFER_NEW_OPT:-$SCRIPT_DIR/third_party/wafer/build_manual/third_party/wafer/bin/wafer-opt}
BASELINE_LLVM=${WAFER_BASELINE_LLVM:-$WORK_ROOT/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.10-x64}
NEW_LLVM=${WAFER_NEW_LLVM:-$WORK_ROOT/llvm-7d5de303-ubuntu-x64}
CLANGXX=${CLANGXX:-/usr/bin/clang++}

for path in "$PYTHON" "$BASELINE_OPT" "$NEW_OPT" \
    "$BASELINE_LLVM/bin/mlir-translate" "$NEW_LLVM/bin/mlir-translate" \
    "$NEW_LLVM/bin/llvm-readobj" "$NEW_LLVM/bin/llvm-objdump" "$CLANGXX"; do
    if [[ ! -x "$path" ]]; then
        echo "ERROR: required executable not found: $path" >&2
        exit 1
    fi
done

cd "$SCRIPT_DIR"
source wafer_env.sh

# 先生成一份固定的 Triton GEMM TTIR，两边必须消费完全相同的输入。
./run_wafer_gemm.sh --output-dir "$INPUT_DIR" >/dev/null

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/baseline" "$OUTPUT_DIR/new"

# 复用回归脚本中的同一套 lowering pipeline，只替换 wafer-opt 二进制。
BASELINE_OPT="$BASELINE_OPT" NEW_OPT="$NEW_OPT" INPUT_TTIR="$INPUT_DIR/ttir.mlir" \
    OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON" - <<'PY'
import importlib.util
import os
from pathlib import Path

script = Path("scripts/verify_wafer_examples.py").resolve()
spec = importlib.util.spec_from_file_location("verify_wafer_examples", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

source = Path(os.environ["INPUT_TTIR"])
for label, compiler in (
    ("baseline", Path(os.environ["BASELINE_OPT"])),
    ("new", Path(os.environ["NEW_OPT"])),
):
    module.lower_case(compiler, source, Path(os.environ["OUTPUT_DIR"]) / label)
    print(f"PASS GEMM lowering: {label}")
PY

# MLIR 21/22 文本格式存在版本边界，各自使用匹配版本转换到标准 LLVM IR。
"$BASELINE_LLVM/bin/mlir-translate" "$OUTPUT_DIR/baseline/llvm.mlir" \
    --mlir-to-llvmir -o "$OUTPUT_DIR/baseline/kernel.ll"
"$NEW_LLVM/bin/mlir-translate" "$OUTPUT_DIR/new/llvm.mlir" \
    --mlir-to-llvmir -o "$OUTPUT_DIR/new/kernel.ll"

# 与当前 compiler backend 一致，SIM 验收对象由宿主机 clang 生成。
"$CLANGXX" "$OUTPUT_DIR/baseline/kernel.ll" -O2 -c -fPIC \
    -o "$OUTPUT_DIR/baseline/kernel.o"
"$CLANGXX" "$OUTPUT_DIR/new/kernel.ll" -O2 -c -fPIC \
    -o "$OUTPUT_DIR/new/kernel.o"

# 保留完整逐行差异；diff 有差异时返回 1，因此这里显式放行以继续生成报告。
diff -u "$OUTPUT_DIR/baseline/llvm.mlir" "$OUTPUT_DIR/new/llvm.mlir" \
    >"$OUTPUT_DIR/llvm-mlir.diff" || true
diff -u "$OUTPUT_DIR/baseline/kernel.ll" "$OUTPUT_DIR/new/kernel.ll" \
    >"$OUTPUT_DIR/llvm-ir.diff" || true

for side in baseline new; do
    "$NEW_LLVM/bin/llvm-readobj" --file-headers --sections --symbols --relocations \
        "$OUTPUT_DIR/$side/kernel.o" >"$OUTPUT_DIR/$side/kernel.readobj"
    "$NEW_LLVM/bin/llvm-objdump" -dr --section-headers --syms \
        "$OUTPUT_DIR/$side/kernel.o" >"$OUTPUT_DIR/$side/kernel.objdump"
done

# 第一行含输入文件路径，不属于 object 内容；归一化后再做结构和反汇编对照。
sed 's|File: .*/kernel.o|File: kernel.o|' "$OUTPUT_DIR/baseline/kernel.readobj" \
    >"$OUTPUT_DIR/baseline/kernel.normalized.readobj"
sed 's|File: .*/kernel.o|File: kernel.o|' "$OUTPUT_DIR/new/kernel.readobj" \
    >"$OUTPUT_DIR/new/kernel.normalized.readobj"
sed 's|^.*/kernel.o:|kernel.o:|' "$OUTPUT_DIR/baseline/kernel.objdump" \
    >"$OUTPUT_DIR/baseline/kernel.normalized.objdump"
sed 's|^.*/kernel.o:|kernel.o:|' "$OUTPUT_DIR/new/kernel.objdump" \
    >"$OUTPUT_DIR/new/kernel.normalized.objdump"

diff -u "$OUTPUT_DIR/baseline/kernel.normalized.readobj" \
    "$OUTPUT_DIR/new/kernel.normalized.readobj" >"$OUTPUT_DIR/object-structure.diff" || true
diff -u "$OUTPUT_DIR/baseline/kernel.normalized.objdump" \
    "$OUTPUT_DIR/new/kernel.normalized.objdump" >"$OUTPUT_DIR/disassembly.diff" || true

sha256sum "$OUTPUT_DIR/baseline/kernel.o" "$OUTPUT_DIR/new/kernel.o" \
    | tee "$OUTPUT_DIR/object.sha256"

status=0
for artifact in llvm.mlir kernel.ll kernel.o; do
    if cmp -s "$OUTPUT_DIR/baseline/$artifact" "$OUTPUT_DIR/new/$artifact"; then
        echo "PASS byte-identical: $artifact"
    else
        echo "DIFF: $artifact"
        status=1
    fi
done
for report in object-structure.diff disassembly.diff; do
    if [[ ! -s "$OUTPUT_DIR/$report" ]]; then
        echo "PASS empty normalized diff: $report"
    else
        echo "DIFF: $report"
        status=1
    fi
done

echo "A/B artifacts: $OUTPUT_DIR"
exit "$status"