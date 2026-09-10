#!/usr/bin/env bash
# Compatibility entry for existing dependency-build automation.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_wafer_deps.sh" "$@"
