#!/usr/bin/env python3
"""Check naming migration boundaries without submitting device work."""

import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[1]
PROTECTED = re.compile(
    r"\b(?:txGetDevice|txSetDevice|txLaunchKernelGGL|txStreamSynchronize|"
    r"txMalloc|txFree|txMemcpy|txMemGetInfo|txStream_t|txError_t|TX_SUCCESS|"
    r"tx8_kernel_printf|tx8_kernel_vprintf|tx8_kernel_vsnprintf|"
    r"CONFIG_TX8_KERNEL_PRINTF_SUPPORT|txda_stream)\b"
)


def git_source(commit, filename):
    return subprocess.check_output(["git", "show", f"{commit}:{filename}"], cwd=REPO, text=True)


def syntax_item(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.dump(node.value, include_attributes=False)
    raise AssertionError(f"Missing source item: {name}")


def exported_symbols(nm, archive):
    result = subprocess.check_output([str(nm), "--defined-only", "--extern-only", "--format=posix", str(archive)], text=True)
    return {line.split()[0] for line in result.splitlines() if len(line.split()) >= 2 and len(line.split()[1]) == 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="248357c")
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = []
    for filename, items in {
        "backend/wafer_runtime.py": ("make_launcher", "_KuiperRuntime", "get_runtime"),
        "backend/wafer.py": ("RCS_LOG_SYMBOLS", "LINK_FLAGS"),
    }.items():
        before, after = git_source(args.baseline, filename), (REPO / filename).read_text()
        for name in items:
            assert syntax_item(before, name) == syntax_item(after, name), name
            checked.append(f"unchanged implementation: {name}")
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.baseline], cwd=REPO, text=True).splitlines()
    protected_files = 0
    for name in files:
        if not name.startswith(("backend/", "scripts/", "test/wafer/", "third_party/wafer/")):
            continue
        # Only text source/configuration files can contain protected call sites.
        if Path(name).suffix not in (".py", ".c", ".cc", ".cpp", ".h", ".td", ".sh", ".txt", ".cmake"):
            continue
        before = git_source(args.baseline, name)
        hits = Counter(PROTECTED.findall(before))
        if not hits:
            continue
        current = name.replace("crt/lib/Tx81/tx81.c", "crt/lib/Wafer/wafer.c").replace("crt/lib/Tx81/", "crt/lib/Wafer/")
        assert Counter(PROTECTED.findall((REPO / current).read_text())) == hits, name
        protected_files += 1
    opt = args.build_dir / "third_party/wafer/bin/wafer-opt"
    help_text = subprocess.check_output([str(opt), "--help"], text=True)
    aliases = {
        "mk-to-wafer": "mk-to-tx81",
        "wafer-to-llvm": "tx81-to-llvm",
        "wafer-memref-to-llvm": "tx81-memref-to-llvm",
    }
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "empty.mlir"
        source.write_text("module {}\n")
        for canonical, legacy in aliases.items():
            assert f"--{canonical}" in help_text and f"--{legacy}" in help_text
            outputs = [subprocess.check_output([str(opt), str(source), f"--{name}"], text=True) for name in (canonical, legacy)]
            assert outputs[0] == outputs[1], (canonical, legacy)
    nm = Path(os.environ["LLVM_BINARY_DIR"]) / "llvm-nm"
    archive = Path("third_party/wafer/crt/lib/libvr.a")
    old_symbols = exported_symbols(nm, args.baseline_build / archive)
    new_symbols = exported_symbols(nm, args.build_dir / archive)
    assert not old_symbols - new_symbols, sorted(old_symbols - new_symbols)
    assert {"wafer_memcpy", "tx81_memcpy"} <= new_symbols
    report = {
        "baseline": args.baseline,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "checks": checked,
        "protected_source_files": protected_files,
        "pass_aliases": aliases,
        "removed_crt_exports": sorted(old_symbols - new_symbols),
        "added_crt_exports": sorted(new_symbols - old_symbols),
        "device_work_submitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
