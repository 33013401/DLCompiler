"""Run against the isolated wheel; no vendor runtime or device is required."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


@triton.jit
def scalar_copy_kernel(src, out):
    tl.store(out, tl.load(src))


@triton.jit
def member_slice_kernel(out):
    x = tl.full((16,), 1, tl.float32)
    sub = x.extract_slice(offsets=(0,), sizes=(8,), strides=(1,))
    y = x.insert_slice(sub + 1, offsets=(8,))
    tl.store(out + tl.arange(0, 16), y)


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


@pytest.mark.parametrize("ascend", [False, True])
def test_dicp_linalg_consumes_fold_guard(ascend):
    from triton._C.libtriton import dicp_triton

    # Bare scalar pointers exercise the zero-offset AddPtr rewrite that used
    # to fight folding. Cover both raw modules and the Ascend frontend marker.
    module = make_module(scalar_copy_kernel, ascend, {"src": "*fp32", "out": "*fp32"})
    module.set_attr("test.preserved", ir.builder(module.context).get_unit_attr())
    dicp_triton.load_dialects(module.context)
    dicp_triton.ir.load_dialects(module.context)
    pm = ir.pass_manager(module.context)
    dicp_triton.passes.ttir.add_triton_to_linalg(pm, False, False, False, False, False)
    pm.run(module)
    text = str(module)
    assert "dicp.disable_addptr_fold" not in text
    assert "test.preserved" in text
    assert "func.func" in text and "memref.load" in text
    assert "tt.load" not in text and "tt.store" not in text


@pytest.mark.parametrize("ascend", [False, True])
def test_simt_export_consumes_only_fold_guard(ascend, monkeypatch):
    from triton.backends.dicp_triton import npu

    # Execute real IR cleanup and serialization using the isolated build.
    # Only the external Bisheng invocation is replaced, so no NPU is required.
    module = make_module(scalar_copy_kernel, ascend, {"src": "*fp32", "out": "*fp32"})
    module.set_attr("test.preserved", ir.builder(module.context).get_unit_attr())
    original = str(module)
    captured = []

    def compile_stub(command, **kwargs):
        text = Path(command[1]).read_text()
        captured.append(text)
        assert "dicp.disable_addptr_fold" not in text
        assert "test.preserved" in text
        assert "tt.load" in text and "tt.store" in text
        # Preserve the kernel body; cleanup is restricted to the module marker.
        assert text[text.index("tt.func"):] == original[original.index("tt.func"):]
        Path(command[command.index("-o") + 1] + ".o").write_bytes(b"test-binary")
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(npu, "_get_npucompiler_path", lambda: ("bisheng-test-stub", {}))
    monkeypatch.setattr(npu, "triton_enable_libdevice_simt", lambda: False)
    monkeypatch.setattr(npu.subprocess, "run", compile_stub)
    options = npu.NPUOptions(force_simt_only=True)
    result = npu.ttir_to_npubin(module, {"target": GPUTarget("ascend", "Ascend910B", 32)}, options)
    assert result == b"test-binary"
    assert len(captured) == 1


@pytest.mark.parametrize("target", ["wafer", "wafer-cache-before-tle", "wafer-cache-after-tle", "ascend", "ascend-dsl"])
def test_codegen_in_separate_processes(target):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), target],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cache_tracks_vendor_sources_without_importing_them(tmp_path, monkeypatch):
    from triton.runtime import cache
    import sysconfig

    root = tmp_path / "triton"
    # A package initializer must never run just to compute a cache key. Both
    # package and nested source edits must nevertheless invalidate that key.
    source = root / "language/extra/vendor_probe/ops.py"
    files = {
        "runtime/cache.py": "# cache input\n",
        "_C/libtriton." + sysconfig.get_config_var("EXT_SUFFIX").split(".")[-1]: "binary input",
        "language/extra/__init__.py": "raise RuntimeError('unexpected import')\n",
        "language/extra/vendor_probe/__init__.py": "raise RuntimeError('unexpected import')\n",
        "language/extra/vendor_probe/ops.py": "VALUE = 1\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(cache, "__file__", str(root / "runtime/cache.py"))
    initial = cache.triton_key.__wrapped__()
    source.write_text("VALUE = 2\n")
    changed_source = cache.triton_key.__wrapped__()
    source.with_name("__init__.py").write_text("raise RuntimeError('still must not run')\n")
    assert len({initial, changed_source, cache.triton_key.__wrapped__()}) == 3


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
    if sys.argv[1].startswith("wafer-cache-"):
        from triton.runtime.cache import triton_key

        if sys.argv[1] == "wafer-cache-before-tle":
            triton_key()
        import triton.experimental.tle.language  # registers Wafer tensor members
        members = (tl.tensor.extract_slice, tl.tensor.insert_slice, tl.tensor.__getitem__)
        original_tanh = getattr(tl.math, "tanh", None)
        triton_key()
        assert not any(".deeplink.cann" in name for name in sys.modules)
        assert members == (tl.tensor.extract_slice, tl.tensor.insert_slice, tl.tensor.__getitem__)
        assert getattr(tl.math, "tanh", None) is original_tanh
        text = str(make_module(member_slice_kernel))
        assert '"dsa.extract_slice"' in text and '"dsa.insert_slice"' in text
        sys.exit(0)
    # Exercise the cache path before Ascend's explicit imports too.
    from triton.runtime.cache import triton_key
    triton_key()
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
