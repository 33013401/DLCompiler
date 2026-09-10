#!/usr/bin/env python3
"""Audit a DLCompiler runtime-linked TX81 kernel before submitting device work."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def audit_kernel(path, log_abi="rcs"):
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected a little-endian ELF64 kernel: {path}")
    elf_type, machine = struct.unpack_from("<HH", data, 16)
    flags = struct.unpack_from("<I", data, 48)[0]
    if elf_type != 3 or machine != 243 or flags & 7 != 5:
        raise ValueError(
            f"Expected RISC-V shared ELF with RVC/double-float ABI: {path}"
        )
    binary_dir = Path(os.environ["LLVM_BINARY_DIR"])
    dynamic = subprocess.check_output(
        [str(binary_dir / "llvm-readelf"), "-l", "-d", "-S", str(path)], text=True
    )
    if "INTERP" in dynamic or "(NEEDED)" in dynamic:
        raise ValueError(
            f"Device ELF contains an interpreter or shared-library dependency: {path}"
        )
    if "ExportedDYNSYMTab" not in dynamic:
        raise ValueError(f"Device ELF has no ExportedDYNSYMTab: {path}")
    symbols = subprocess.check_output(
        [str(binary_dir / "llvm-nm"), "--undefined-only", "--format=posix", str(path)],
        text=True,
    )
    undefined = {line.split()[0] for line in symbols.splitlines() if line.strip()}
    allowed = {
        "__get_pid",
        "get_log_level",
        "monitor_write_log",
        "rt_free",
        "rt_malloc",
        "rt_thread_mdelay",
    }
    allowed.update(
        {"rcs_ep_log", "rcs_kernel_printf", "rcs_kernel_vprintf"}
        if log_abi == "rcs"
        else {"tsm_ep_log", "tx8_kernel_printf", "tx8_kernel_vprintf"}
    )
    unknown = undefined - allowed
    if unknown:
        raise ValueError(f"Unreviewed firmware imports in {path}: {sorted(unknown)}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "flags": flags,
        "device_log_abi": log_abi,
        "firmware_imports": sorted(undefined),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernels", nargs="+", type=Path)
    parser.add_argument("--log-abi", choices=("tx8", "rcs"), default="rcs")
    args = parser.parse_args()
    print(
        json.dumps(
            [audit_kernel(path, args.log_abi) for path in args.kernels], indent=2
        )
    )
