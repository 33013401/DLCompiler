"""Compatibility behavior at the Wafer naming migration boundaries."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_legacy_imports_keep_one_discoverable_backend(wafer_modules):
    from triton.backends import _find_concrete_subclasses
    from triton.backends.compiler import BaseBackend

    _, compiler, runtime = wafer_modules
    assert compiler.TXDABackend is compiler.WaferBackend
    assert compiler.TXDAOptions is compiler.WaferOptions
    assert runtime.TXDALauncher is runtime.WaferLauncher
    assert runtime.TXDAUtils is runtime.WaferUtils
    assert _find_concrete_subclasses(compiler, BaseBackend) is compiler.WaferBackend


@pytest.mark.parametrize("mode", ["legacy", "canonical", "both"])
def test_runtime_dependencies_accept_old_and_new_configuration(
    tmp_path, monkeypatch, wafer_modules, mode
):
    _, compiler, _ = wafer_modules
    roots = {}
    for label in ("legacy", "canonical"):
        roots[label] = tmp_path / label
        (roots[label] / "lib").mkdir(parents=True)
        for name in ("libcommon_util.a", "libinstr_tx81.a", "liblibc_stub.a"):
            (roots[label] / "lib" / name).write_bytes(label.encode())
    toolchain = tmp_path / "toolchain"
    (toolchain / "bin").mkdir(parents=True)
    (toolchain / "bin/riscv64-unknown-elf-gcc").touch()
    archive_dir = tmp_path / "crt"
    archive_dir.mkdir()
    (archive_dir / "libvr.a").touch()
    for name in ("libm.a", "libc.a", "libgcc.a"):
        (archive_dir / name).touch()
    monkeypatch.setenv("XUANTIE_NAME", str(toolchain))
    monkeypatch.setenv("WAFER_RUNTIME_LIB_DIR", str(archive_dir))
    monkeypatch.delenv("WAFER_DEPS_ROOT", raising=False)
    monkeypatch.delenv("TX8_DEPS_ROOT", raising=False)
    if mode != "canonical":
        monkeypatch.setenv("TX8_DEPS_ROOT", str(roots["legacy"]))
    if mode != "legacy":
        monkeypatch.setenv("WAFER_DEPS_ROOT", str(roots["canonical"]))
    monkeypatch.setattr(compiler, "_find_linker_library", lambda linker, name: archive_dir / name)
    _, libraries = compiler._runtime_link_inputs()
    expected = roots["legacy" if mode == "legacy" else "canonical"]
    assert all(path.parent == expected / "lib" for path in libraries[:3])


@pytest.mark.parametrize("mode", ["legacy", "canonical", "both"])
def test_cmake_configuration_precedence(tmp_path, mode):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to verify configuration compatibility")
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "config.cmake"
    script.write_text(
        f'include("{repo / "third_party/wafer/cmake/WaferConfig.cmake"}")\n'
        'if(NOT WAFER_DEPS_ROOT STREQUAL EXPECT OR NOT TX8_DEPS_ROOT STREQUAL EXPECT)\n'
        '  message(FATAL_ERROR "configuration precedence mismatch")\n'
        'endif()\n'
    )
    arguments = []
    if mode != "canonical":
        arguments.append("-DTX8_DEPS_ROOT=/legacy")
    if mode != "legacy":
        arguments.append("-DWAFER_DEPS_ROOT=/canonical")
    expected = "/legacy" if mode == "legacy" else "/canonical"
    environment = os.environ.copy()
    environment.pop("TX8_DEPS_ROOT", None)
    environment.pop("WAFER_DEPS_ROOT", None)
    subprocess.run(
        [cmake, *arguments, f"-DEXPECT={expected}", "-P", str(script)],
        check=True, capture_output=True, text=True, env=environment,
    )
