"""Shape vectors and nonuniform tensors must lower and bufferize on Wafer."""
import os
import subprocess

import pytest


@pytest.mark.parametrize("dtype,shape,values", [
    ("i64", "2", "[16, 512]"),
    ("i32", "2x3", "[[1, 2, 3], [-4, 5, 6]]"),
    ("f32", "2x2", "[[1.25, -2.5], [3.0, 0.0]]"),
    ("index", "2", "[16, 512]"),
])
def test_non_splat_constant_bufferization(dtype, shape, values, tmp_path):
    from triton.backends.dicp_triton.wafer import _find_wafer_opt

    ty = f"tensor<{shape}x{dtype}>"
    source = tmp_path / "constant.mlir"
    source.write_text(f"""module {{
      func.func @constant() -> {ty} {{
        %value = arith.constant dense<{values}> : {ty}
        return %value : {ty}
      }}
    }}""")
    result = subprocess.run([
        os.getenv("WAFER_TEST_OPT") or str(_find_wafer_opt()), str(source),
        "--linalg-to-mk=precision-mode=2",
        "--one-shot-bufferize=bufferize-function-boundaries",
        "--convert-bufferization-to-memref",
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "arith.constant dense<" not in result.stdout
    assert f"memref<{shape}x{dtype}>" in result.stdout
