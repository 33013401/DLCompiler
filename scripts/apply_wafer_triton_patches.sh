#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TRITON_DIR="$ROOT_DIR/third_party/triton"
TRITON_BASE_COMMIT=c3c476f357f1e9768ea4e45aa5c17528449ab9ef
PATCHES=(
    "$ROOT_DIR/patch/triton/wafer_builder_optional_gluon.patch"
    "$ROOT_DIR/patch/triton/wafer_proton_backend_filter.patch"
    "$ROOT_DIR/patch/triton/python_triton_compiler_optional_gluon_py.patch"
)

if [[ $(git -C "$TRITON_DIR" rev-parse HEAD) != "$TRITON_BASE_COMMIT" ]]; then
    echo "ERROR: Wafer patches require official Triton v3.5.0 commit $TRITON_BASE_COMMIT" >&2
    exit 1
fi

for patch in "${PATCHES[@]}"; do
    if git -C "$TRITON_DIR" apply --check "$patch" 2>/dev/null; then
        git -C "$TRITON_DIR" apply "$patch"
    elif git -C "$TRITON_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        :
    else
        echo "ERROR: patch does not match the pinned Triton source: $patch" >&2
        exit 1
    fi
done