#!/usr/bin/env bash
# Compatibility entry; source the canonical Wafer activation script.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this script instead of executing it" >&2
    exit 1
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/init_wafer_env.sh"
