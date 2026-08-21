# Wafer Triton Plugin

## Current Status

This directory contains the Wafer external Triton plugin imported for the
DLCompiler Triton 3.5 migration. The import includes the bundled FLIR and TLE
sources and excludes prior build trees, generated IR, binaries, caches, and
machine-specific paths.

The plugin is adapted to LLVM/MLIR 22 and connected to the combined DLCompiler
and Wafer plugin build. A real Triton BF16 GEMM containing `tl.dot` is verified
through TTIR, CoreIR, TXIR, LLVM dialect IR, LLVM IR, and object generation.

This is currently a compiler-only integration. The FlagTree
`third_party/tsingmicro` launcher is the reference for the future TX81 runtime
migration, but Kuiper linking, `torch_txda`, and hardware launch are not yet
enabled in this repository.

## One-Command GEMM Compilation

From the DLCompiler repository root, run:

```bash
./run_wafer_gemm.sh
```

The script prepares and loads `wafer_env.sh`, checks the compiler installation,
builds and installs it when necessary, and compiles the built-in real
`@triton.jit` BF16 GEMM. Use `--rebuild` to force a clean compiler rebuild:

```bash
./run_wafer_gemm.sh --rebuild
```

Use a different output directory with:

```bash
./run_wafer_gemm.sh --output-dir /tmp/my-wafer-gemm
```

The default GEMM artifacts are:

```text
/tmp/wafer-gemm-run/ttir.mlir
/tmp/wafer-gemm-run/coreir.mlir
/tmp/wafer-gemm-run/txir.mlir
/tmp/wafer-gemm-run/llvm.mlir
/tmp/wafer-gemm-run/kernel.ll
/tmp/wafer-gemm-run/kernel.o
```

The compiler artifacts are:

```text
third_party/wafer/build_manual/third_party/wafer/bin/wafer-opt
third_party/wafer/build_manual/libtriton.so
third_party/wafer/build_manual/wheel/triton-*.whl
```

`llvm.mlir` is MLIR LLVM dialect IR, while `kernel.ll` is standard textual LLVM
IR. The script sets `USE_SIM_MODE=1`, so `kernel.o` is the host compiler
acceptance object and is not a directly deployable TX81 binary.

## Baseline/Current Artifact Comparison

Run a strict A/B comparison with one common GEMM TTIR input:

```bash
./compare_wafer_gemm.sh
```

The script lowers the input with the original and current `wafer-opt`, generates
LLVM IR and host objects, and compares the final files byte-for-byte. It also
produces normalized ELF structure and disassembly reports under:

```text
/tmp/wafer-gemm-strict-ab/
├── baseline/
├── new/
├── llvm-mlir.diff
├── llvm-ir.diff
├── object-structure.diff
├── disassembly.diff
└── object.sha256
```

Byte-identical objects strongly validate compiler equivalence for this input,
but do not prove GEMM numerical correctness. Numerical simulation still needs
the unavailable `libtriton_cmodel` and `libtx8be_op_cmodel` libraries; hardware
validation needs a linked RISC-V binary, TXDA runtime, and a TX81 device.

## Source Layout

- `backend/`: Python backend implementation imported for later adaptation
- `bin/`: Wafer command-line tools and dialect registration
- `include/` and `lib/`: dialects, analyses, and conversion passes
- `crt/` and `profiler/`: optional TX8 runtime components
- `third_party/flir/`: bundled FLIR sources
- `third_party/tle/`: bundled TLE sources

## Optional TX8 Dependencies

`TX8_DEPS_ROOT` is optional at plugin configuration time. When it is unset or
points to a missing directory, the top-level CMake configuration skips both
`crt/` and `profiler/`.

The main compiler and dialect sources remain available without TX8 runtime
dependencies. A later build that enables CRT or the profiler must provide a
valid `TX8_DEPS_ROOT` and a compatible LLVM toolchain containing Clang.

## LLVM Environment

Prepare the Triton-pinned LLVM/MLIR 22 environment from the repository root:

```bash
./setup_llvm22_env.sh
source llvm22_env.sh
```

Device code generation and linking require a matching LLVM RISC-V toolchain
and TX8 runtime libraries in addition to the compiler-only package.
