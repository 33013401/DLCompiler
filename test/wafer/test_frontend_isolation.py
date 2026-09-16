"""Run against the isolated wheel; no vendor runtime or device is required."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir, wafer
from triton.backends.compiler import GPUTarget
from triton.backends.dicp_triton.wafer import WaferBackend, _find_wafer_opt
from triton.compiler import ASTSource


pytestmark = pytest.mark.skipif(
    getattr(wafer, "build_role", None) != "frontend",
    reason="requires the isolated DICP + Wafer frontend wheel",
)


@triton.jit
def loop_kernel(out):
    i = tl.arange(0, 16)
    x = i.to(tl.float32)
    for _ in range(2):
        x = x + 1.0
    tl.store(out + i, x)


@triton.jit
def non_power_of_two(out):
    i = tl.arange(0, 3)
    tl.store(out + i, i.to(tl.float32))


def make_module(fn, ascend=False, signature=None, constexprs=None):
    from triton._C.libtriton import dicp_triton

    backend = WaferBackend(GPUTarget("wafer", "tx81", 32))
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    if ascend:
        dicp_triton.load_dialects(context)
        dicp_triton.ir.load_dialects(context)
    codegen = backend.get_codegen_implementation(options)
    codegen["dicp_ascend"] = ascend
    return ASTSource(fn, signature=signature or {"out": "*fp32"}, constexprs=constexprs).make_ir(
        backend.target, options, codegen, {}, context
    )


def test_original_dicp_bindings_and_linalg_options():
    from triton._C.libtriton import dicp_triton

    context = ir.context()
    ir.load_dialects(context)
    dicp_triton.load_dialects(context)
    bindings = dicp_triton.passes.ttir
    pm = ir.pass_manager(context)
    bindings.add_triton_to_structure(pm, False, False)
    bindings.add_discrete_mask_access_conversion(pm, False, False, False)
    bindings.add_triton_to_unstructure(pm, False, False)
    pipeline = pm.get_pipeline_str()
    assert "triton-to-structured" in pipeline
    assert "triton-to-unstructure" in pipeline
    pipelines = set()
    # Every option must reach the real pass, not a compatibility stub.
    for selected in range(-1, 5):
        pm = ir.pass_manager(context)
        bindings.add_triton_to_linalg(pm, *[i == selected for i in range(5)])
        pipelines.add(pm.get_pipeline_str())
    assert len(pipelines) == 6


@pytest.mark.parametrize("target", ["wafer", "ascend", "ascend-dsl"])
def test_codegen_in_separate_processes(target):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), target],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_frontend_text_can_enter_wafer_lowering(tmp_path):
    source = tmp_path / "frontend.mlir"
    source.write_text(str(make_module(loop_kernel)))
    result = subprocess.run(
        [os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt()), str(source),
         "--triton-to-core-dialects", "-o", str(tmp_path / "core.mlir")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "linalg" in (tmp_path / "core.mlir").read_text()


def test_tle_text_can_enter_separate_tools(tmp_path):
    from test_tle_frontend import local_kernel

    source = tmp_path / "dsa.mlir"
    source.write_text(str(make_module(local_kernel)))
    assert "dsa.alloc" in source.read_text()
    result = subprocess.run(
        [os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt()), str(source),
         "--triton-to-core-dialects", "--tle-to-mk", "--dsa-memory-to-core",
         "-o", str(tmp_path / "core.mlir")], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "dsa.alloc" not in (tmp_path / "core.mlir").read_text()


def check_original_ascend_dsl():
    import importlib.util

    root = Path(__file__).resolve().parents[1] / "dsl"
    cases = [
        ("test_alloc.py", "custom_func_kernel",
         {"x_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
         {"BLOCK_SIZE": 16}, "tt.store"),
        ("test_compile_hint.py", "triton_compile_hint",
         {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32", "XBLOCK": "constexpr", "XBLOCK_SUB": "constexpr"},
         {"XBLOCK": 32, "XBLOCK_SUB": 16}, "hint_a"),
        ("passed_tests/test_lambda.py", "add_kernel",
         {"x_ptr": "*fp32", "y_ptr": "*fp32", "output_ptr": "*fp32", "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
         {"BLOCK_SIZE": 16}, "arith.addf"),
    ]
    for i, (file, name, signature, constants, expected) in enumerate(cases):
        spec = importlib.util.spec_from_file_location(f"ascend_dsl_{i}", root / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        text = str(make_module(getattr(module, name), True, signature, constants))
        assert expected in text, text


if __name__ == "__main__":
    assert wafer.build_role == "frontend"
    if sys.argv[1] == "ascend-dsl":
        check_original_ascend_dsl()
        sys.exit(0)
    ascend = sys.argv[1] == "ascend"
    original_tanh = getattr(tl.math, "tanh", None)
    module = str(make_module(loop_kernel, ascend))
    assert "scf.for" in module and "tt.store" in module
    assert ("dicp.disable_addptr_fold" in module) == ascend
    if ascend:
        assert "tensor<3xi32>" in str(make_module(non_power_of_two, True))
        assert "triton.language.extra.deeplink.cann" in sys.modules
    else:
        assert not any(".deeplink.cann" in name for name in sys.modules)
        assert getattr(tl.math, "tanh", None) is original_tanh
        with pytest.raises(triton.compiler.errors.CompilationError, match="power of 2"):
            make_module(non_power_of_two)
