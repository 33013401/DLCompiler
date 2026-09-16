#!/usr/bin/env python3
"""Build DICP+DSA frontend and Wafer tools in separate CMake/source trees."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from apply_triton_profile import ROOT, CATALOG, apply_profile, git
from wafer_artifacts import sha256


def prepare_source(path, seed, profile):
    path = path.resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--shared", "--no-checkout", str(seed), str(path)], check=True)
        commit = json.loads(CATALOG.read_text())["triton_commit"]
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
    return apply_profile(path, profile)


def audit_boundaries(frontend, tools):
    def sources(build):
        return [row["file"] for row in json.loads((build / "compile_commands.json").read_text())]
    front = sources(frontend)
    back = sources(tools)
    if any("third_party/wafer/third_party/flir/" in p for p in front):
        raise RuntimeError("FLIR leaked into the shared frontend")
    if any("ascendnpu-ir/" in p or "/compiler/lib/" in p for p in back):
        raise RuntimeError("DICP/Ascend implementation leaked into wafer-opt")
    for suffix in (
        "/compiler/lib/TritonToLinalg/TritonToLinalgPass.cpp",
        "/compiler/lib/TritonToUnstructure/UnstructureConversionPass.cpp",
        "/compiler/lib/DiscreteMaskAccessConversion/DiscreteMaskAccessConversionPass.cpp",
        "/compiler/lib/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp",
    ):
        if not any(p.endswith(suffix) for p in front):
            raise RuntimeError(f"Original DICP source is missing: {suffix}")
    if not any(p.endswith("/python/triton_wafer_frontend.cc") for p in front):
        raise RuntimeError("Wafer frontend binding is missing")
    return {"frontend_sources": len(front), "tools_sources": len(back), "passed": True}


def audit_cache(build, source, role, llvm):
    entries = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            entries[key.split(":", 1)[0]] = value
    expected = {
        "CMAKE_HOME_DIRECTORY": str(source.resolve()),
        "WAFER_BUILD_ROLE": role,
        "LLVM_SYSPATH": str(llvm),
        "TRITON_PLUGIN_DIRS": (str(ROOT) + ";" if role == "frontend" else "") + str(ROOT / "third_party/wafer"),
    }
    for key, value in expected.items():
        if entries.get(key) != value:
            raise RuntimeError(f"Unexpected {key} in {build}: {entries.get(key)!r}, expected {value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path(os.getenv("WAFER_BUILD_DIR", ROOT / "third_party/wafer/build_manual")))
    parser.add_argument("--frontend-source", type=Path)
    parser.add_argument("--tools-source", type=Path)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--record-only", action="store_true", help="Verify completed builds and write their manifest")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    frontend = build / "frontend"
    tools = build / "tools"
    if args.clean:
        if args.record_only:
            parser.error("--clean cannot be combined with --record-only")
        for directory in (frontend, tools):
            if directory.exists():
                if not (directory / "CMakeCache.txt").is_file() or directory.is_symlink():
                    raise RuntimeError(f"Refusing to clean an unrecognized build directory: {directory}")
                shutil.rmtree(directory)

    front_source = args.frontend_source or build / "sources/triton-frontend"
    tool_source = args.tools_source or build / "sources/triton-tools"
    if front_source.resolve() == tool_source.resolve():
        raise RuntimeError("Frontend and tools cannot share their Triton source directory")
    front_profile = prepare_source(front_source, ROOT / "third_party/triton", "wafer-frontend")
    tool_profile = prepare_source(tool_source, ROOT / "third_party/triton", "wafer-tools")
    llvm = Path(os.environ["LLVM_SYSPATH"]).resolve()
    llvm_commit = (front_source / "cmake/llvm-hash.txt").read_text().strip()
    if (tool_source / "cmake/llvm-hash.txt").read_text().strip() != llvm_commit:
        raise RuntimeError("Frontend and tools request different LLVM builds")
    if os.environ.get("LLVM_COMMIT", llvm_commit) != llvm_commit:
        raise RuntimeError("LLVM_COMMIT does not match the pinned Triton")
    sdk = Path(os.environ.get("WAFER_DEPS_ROOT", os.environ.get("TX8_DEPS_ROOT", ""))).resolve()
    sdk_include = Path(os.environ.get("WAFER_SDK_INCLUDE_DIR", sdk / "include"))
    if not (sdk_include / "instr_def.h").is_file():
        raise RuntimeError("Wafer tools require WAFER_SDK_INCLUDE_DIR/instr_def.h")
    if not (ROOT / "third_party/ascendnpu-ir/CMakeLists.txt").is_file():
        raise RuntimeError("Initialize the pinned AscendNPU-IR dependency for the DICP frontend")

    base_args = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=" + os.getenv("BUILD_TYPE", "Release"),
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 "-DCMAKE_C_COMPILER=" + os.getenv("CC", "/usr/bin/cc"),
                 "-DCMAKE_CXX_COMPILER=" + os.getenv("CXX", "/usr/bin/c++"),
                 f"-DLLVM_SYSPATH={llvm}", f"-DLLVM_DIR={llvm}/lib/cmake/llvm",
                 f"-DMLIR_DIR={llvm}/lib/cmake/mlir", "-DTRITON_BUILD_PYTHON_MODULE=ON",
                 "-DTRITON_BUILD_UT=OFF", "-DTRITON_BUILD_TUTORIALS=OFF",
                 "-DTRITON_BUILD_PROTON=OFF", f"-DPython3_EXECUTABLE={sys.executable}"]
    jobs = os.getenv("CMAKE_BUILD_PARALLEL_LEVEL", "4")
    if not args.record_only:
        # These are the optional Wafer package's settings. compile_shared.sh
        # continues to select the normal Ascend profile and its own defaults.
        front_args = [f"-DTRITON_PLUGIN_DIRS={ROOT};{ROOT}/third_party/wafer",
                      "-DWAFER_BUILD_ROLE=frontend",
                      "-DTRITON_CODEGEN_BACKENDS=" + os.getenv("WAFER_FRONTEND_CODEGEN_BACKENDS", "nvidia"),
                      "-DTRITON_BUILD_GLUON_IR=" + os.getenv("WAFER_FRONTEND_GLUON", "OFF")]
        tool_args = [f"-DTRITON_PLUGIN_DIRS={ROOT}/third_party/wafer", "-DWAFER_BUILD_ROLE=tools",
                     "-DTRITON_CODEGEN_BACKENDS=nvidia", "-DTRITON_BUILD_GLUON_IR=OFF",
                     "-DTRITON_SHARED_BUILD_CPU_BACKEND=OFF",
                     f"-DWAFER_SDK_INCLUDE_DIR={sdk_include}", f"-DWAFER_DEPS_ROOT={sdk}",
                     "-DUSE_SIM_MODE=OFF"]
        tool_env = dict(os.environ, USE_SIM_MODE="0")
        tool_env.setdefault("WAFER_RT_THREAD_SMP_ROOT", str(sdk / "tx8-yoc-rt-thread-smp"))
        tool_env.setdefault("XUANTIE_NAME", str(sdk / "Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2"))
        for source, directory, extra, targets, env in (
            (front_source, frontend, front_args, ["libtriton.so", "dicp_opt"], os.environ),
            (tool_source, tools, tool_args, ["wafer-opt", "vr"], tool_env),
        ):
            subprocess.run(["cmake", "-S", str(source.resolve()), "-B", str(directory), *base_args,
                            f"-DTRITON_WHEEL_DIR={directory}/wheel", *extra], check=True, env=env)
            subprocess.run(["cmake", "--build", str(directory), "--target", *targets, "--parallel", jobs], check=True, env=env)

    audit_cache(frontend, front_source, "frontend", llvm)
    audit_cache(tools, tool_source, "tools", llvm)
    # --record-only must not bless a stale binary after a C++/CMake edit.
    for directory, targets in ((frontend, ["libtriton.so", "dicp_opt"]),
                               (tools, ["wafer-opt", "vr"])):
        pending = subprocess.check_output(["ninja", "-C", str(directory), "-n", *targets], text=True)
        if "ninja: no work to do." not in pending:
            raise RuntimeError(f"Build is not up to date: {directory}\n{pending}")
    artifacts = {"libtriton.so": frontend / "libtriton.so",
                 "dicp_opt": frontend / "third_party/dicp_triton/tools/dicp_triton_opt/dicp_opt",
                 "FileCheck": llvm / "bin/FileCheck",
                 "wafer-opt": tools / "third_party/wafer/bin/wafer-opt",
                 "libvr.a": tools / "third_party/wafer/crt/lib/libvr.a"}
    manifest = {"schema": 1, "layout": "isolated-frontend-tools", "llvm_commit": llvm_commit,
                "llvm_path": str(llvm), "dlcompiler_commit": git(ROOT, "rev-parse", "HEAD").decode().strip(),
                "frontend": front_profile, "tools": tool_profile,
                "boundaries": audit_boundaries(frontend, tools),
                "artifacts": {name: {"path": str(path), "sha256": sha256(path)} for name, path in artifacts.items()}}
    (build / "wafer-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Isolated build verified: {build / 'wafer-build.json'}")


if __name__ == "__main__":
    main()
