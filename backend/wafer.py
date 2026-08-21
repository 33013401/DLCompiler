import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget


@dataclass(frozen=True)
class TXDAOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs: tuple = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str, ...] = ("ieee", )
    sanitize_overflow: bool = True
    max_num_imprecise_acc_default: int = 0
    supported_fp8_dtypes: Tuple[str, ...] = ("fp8e5", "fp8e4b15", "fp8e4nv")
    deprecated_fp8_dtypes: Tuple[str, ...] = ()

    def hash(self):
        key = "_".join(f"{name}-{value}" for name, value in self.__dict__.items())
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _run_tool(arguments):
    subprocess.check_call(arguments, stdout=None if os.getenv("MLIR_ENABLE_DUMP") == "1" else subprocess.DEVNULL)


def _dump_file(path):
    dump_dir = os.getenv("TRITON_DUMP_PATH")
    if dump_dir:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(path, Path(dump_dir) / Path(path).name)


def _find_wafer_opt():
    override = os.getenv("WAFER_OPT_PATH")
    if override:
        path = Path(override)
        if path.is_file():
            return str(path)
        raise RuntimeError(f"WAFER_OPT_PATH does not name a file: {path}")

    backend_dir = Path(__file__).resolve().parent
    candidates = (
        backend_dir / "bin" / "wafer-opt",
        backend_dir.parent / "third_party" / "wafer" / "build_manual" / "third_party" / "wafer" / "bin" / "wafer-opt",
        backend_dir.parent
        / "third_party"
        / "wafer"
        / "build_manual"
        / "install"
        / "triton"
        / "backends"
        / "wafer"
        / "bin"
        / "wafer-opt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    path = shutil.which("wafer-opt")
    if path:
        return path
    raise RuntimeError("wafer-opt not found; run compile_wafer.sh or set WAFER_OPT_PATH")


def _find_llvm_tool(name):
    llvm_bin = os.getenv("LLVM_BINARY_DIR")
    if llvm_bin:
        candidate = Path(llvm_bin) / name
        if candidate.is_file():
            return str(candidate)
    path = shutil.which(name)
    if path:
        return path
    raise RuntimeError(f"{name} not found; set LLVM_BINARY_DIR")


def _run_wafer_stage(source, arguments, source_name, output_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / source_name
        output_path = Path(tmpdir) / output_name
        source_path.write_text(str(source), encoding="utf-8")
        command = [_find_wafer_opt(), str(source_path), *arguments, "-o", str(output_path)]
        _run_tool(command)
        _dump_file(source_path)
        _dump_file(output_path)
        return output_path.read_text(encoding="utf-8")


def ttir_to_coreir(module):
    core_to_mk = "--core-dialects-to-mk"
    if os.getenv("PRECISION_PRIORITY", "0").lower() in ("1", "true", "yes"):
        core_to_mk += "=precision-priority"
    return _run_wafer_stage(
        module,
        [
            "--triton-to-core-dialects",
            "--tle-to-mk",
            "--dsa-memory-to-core",
            "--linalg-tiling",
            core_to_mk,
            "--linalg-fusion",
            "--legalize-tensor-form-loops",
            "--one-shot-bufferize",
            "--convert-bufferization-to-memref",
            "--cse",
            "--canonicalize",
        ],
        "ttir.mlir",
        "coreir.mlir",
    )


def coreir_to_txir(module):
    return _run_wafer_stage(
        module,
        [
            "--spmd-allocate-shared-memory",
            "--expand-strided-metadata",
            "--lower-affine",
            "--mk-to-tx81",
            "--cse",
        ],
        "coreir.mlir",
        "txir.mlir",
    )


def txir_to_llir(module, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "txir.mlir"
        llvm_mlir_path = Path(tmpdir) / "llvm.mlir"
        llvm_ir_path = Path(tmpdir) / "kernel.ll"
        source_path.write_text(str(module), encoding="utf-8")
        wafer_arguments = [
            _find_wafer_opt(),
            str(source_path),
            "--tx81-memref-to-llvm",
            "--addr-to-llvm",
            "--convert-scf-to-cf",
            "--convert-math-to-llvm",
            "--convert-math-to-libm",
            "--convert-cf-to-llvm",
            "--convert-func-to-llvm",
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
            "--kernel-arg-buffer",
            "--tx81-to-llvm",
            "--convert-arith-to-llvm",
            "--reconcile-unrealized-casts",
            "--canonicalize",
            "--export-kernel-symbols",
            "-o",
            str(llvm_mlir_path),
        ]
        _run_tool(wafer_arguments)
        _run_tool([
            _find_llvm_tool("mlir-translate"),
            str(llvm_mlir_path),
            "--mlir-to-llvmir",
            "-o",
            str(llvm_ir_path),
        ])
        llvm_ir = llvm_ir_path.read_text(encoding="utf-8")
        names = re.findall(r"define\s+(?:\w+\s+)*@([\w.$]+)\(", llvm_ir)
        if names:
            metadata["name"] = names[0]
        metadata.setdefault("shared", 0)
        _dump_file(llvm_mlir_path)
        _dump_file(llvm_ir_path)
        return llvm_ir


def llir_to_object(llvm_ir, metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "kernel.ll"
        object_path = Path(tmpdir) / "kernel.o"
        source_path.write_text(llvm_ir, encoding="utf-8")
        compiler = _find_llvm_tool("clang++")
        arguments = [compiler, str(source_path), "-O2", "-c", "-fPIC", "-o", str(object_path)]
        if os.getenv("USE_SIM_MODE", "0").lower() not in ("1", "true", "yes"):
            arguments.extend(["--target=riscv64-unknown-elf", "-march=rv64imfdc"])
        _run_tool(arguments)
        _dump_file(object_path)
        return object_path.read_bytes()


class TXDABackend(BaseBackend):
    binary_ext = "o"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend in ("wafer", "txda")

    def parse_options(self, options: dict) -> Any:
        arguments = {
            name: options[name]
            for name in TXDAOptions.__dataclass_fields__
            if name in options
        }
        arguments.setdefault("arch", self.target.arch)
        return TXDAOptions(**arguments)

    def hash(self):
        return f"{self.target.backend}-{self.target.arch}-{self.target.warp_size}"

    def get_codegen_implementation(self, options):
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def load_dialects(self, context):
        from triton._C.libtriton import wafer

        wafer.load_dialects(context)
        wafer.tle.load_dialects(context)

    @staticmethod
    def make_ttir(module, metadata, options):
        pass_manager = ir.pass_manager(module.context)
        pass_manager.enable_debug()
        passes.common.add_inliner(pass_manager)
        passes.ttir.add_combine(pass_manager)
        passes.common.add_canonicalizer(pass_manager)
        passes.ttir.add_reorder_broadcast(pass_manager)
        passes.common.add_cse(pass_manager)
        passes.common.add_licm(pass_manager)
        passes.common.add_symbol_dce(pass_manager)
        pass_manager.run(module)
        metadata.setdefault("shared", 0)
        return module

    def add_stages(self, stages, options, language=None):
        stages["ttir"] = lambda source, metadata: self.make_ttir(source, metadata, options)
        stages["coreir"] = lambda source, metadata: ttir_to_coreir(source)
        stages["txir"] = lambda source, metadata: coreir_to_txir(source)
        stages["llir"] = lambda source, metadata: txir_to_llir(source, metadata)
        stages[self.binary_ext] = lambda source, metadata: llir_to_object(source, metadata)

    def get_module_map(self) -> Dict[str, ModuleType]:
        try:
            from triton.language.extra.txda import libdevice

            return {"triton.language.extra.libdevice": libdevice}
        except ImportError:
            return {}