#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
TRITON_DIR="$ROOT_DIR/third_party/triton"
PATCH="$ROOT_DIR/patch/triton/python_triton_compiler_optional_gluon_py.patch"

if git -C "$TRITON_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$TRITON_DIR" apply "$PATCH"
elif git -C "$TRITON_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    :
else
    echo "ERROR: optional-Gluon patch does not match the pinned Triton source" >&2
    exit 1
fi