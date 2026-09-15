"""Opt-in native TXDA math tests; CPU tensors only supply inputs and references."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import pytest


STATE = {"launches": 0, "error": False}


def record(event, **values):
    path = os.getenv("WAFER_MATH_EVENTS")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps(dict(event=event, time=time.time(), **values)) + "\n")


def pytest_addoption(parser):
    parser.addoption("--wafer-math-hardware", action="store_true",
                     help="Execute native math tests on a Wafer board")


@pytest.fixture(scope="session", autouse=True)
def native_launcher(request):
    if not request.config.getoption("--wafer-math-hardware"):
        pytest.skip("use --wafer-math-hardware for real Wafer execution")
    for name, expected in {"DICP_BACKEND": "wafer", "USE_SIM_MODE": "0",
                           "WAFER_ENABLE_RUNTIME": "1"}.items():
        assert os.getenv(name) == expected, f"requires {name}={expected}"
    import torch
    import torch_txda  # noqa: F401 -- registers the native device
    import triton
    from triton.backends.dicp_triton.wafer_runtime import WaferLauncher

    assert torch.txda.is_available()
    driver = triton.runtime.driver.active
    assert driver.get_current_target().backend == "wafer"
    assert driver.get_active_torch_device().type == "txda"
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from audit_wafer_elf import audit_kernel

    original = WaferLauncher.__call__
    audited = set()
    patch = pytest.MonkeyPatch()

    def launch(launcher, *args, **kwargs):
        if STATE["error"]:
            pytest.exit("Device error: inspect evidence before further launches", returncode=3)
        metadata = launcher.metadata
        if metadata.kernel_path not in audited:
            audit_kernel(metadata.kernel_path, metadata.device_log_abi,
                         os.getenv("WAFER_NOC_FIRMWARE_ELF"))
            audited.add(metadata.kernel_path)
        assert all(not isinstance(x, torch.Tensor) or x.device.type == "txda" for x in args[9:])
        started = time.time()
        record("launch_start", kernel=metadata.name, path=metadata.kernel_path, grid=list(args[:3]))
        handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
        timer = signal.setitimer(signal.ITIMER_REAL, float(os.getenv("WAFER_MATH_LAUNCH_TIMEOUT", "60")))
        try:
            result = original(launcher, *args, **kwargs)
            torch.txda.synchronize()
        except BaseException as error:
            STATE["error"] = True
            record("launch_error", error=repr(error))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, *timer)
            signal.signal(signal.SIGALRM, handler)
        STATE["launches"] += 1
        record("launch_complete", kernel=metadata.name)
        # Some firmware errors do not reach the launch API. Check the kernel log
        # before allowing another launch, as required after the CV attention XID.
        journal = subprocess.run(["journalctl", "-k", "--since", f"@{started:.6f}",
                                  "--no-pager", "-o", "cat"], capture_output=True, text=True, timeout=10)
        if journal.returncode:
            STATE["error"] = True
            pytest.fail(f"Cannot check device journal: {journal.stderr}")
        if re.search(r"XID|AP resetting|Clean Resource Timeout|NPU LSU.*Timeout", journal.stdout):
            STATE["error"] = True
            record("device_journal_error", log=journal.stdout)
            pytest.fail("Device journal reported an error after launch")
        return result

    patch.setattr(WaferLauncher, "__call__", launch)
    record("configured", target=str(driver.get_current_target()), triton=triton.__file__, execution="native-txda")
    yield
    patch.undo()


def pytest_runtest_setup(item):
    if STATE["error"]:
        pytest.exit("Device error: stopped remaining math tests", returncode=3)
    STATE["launches"] = 0


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.when == "call" and report.passed and not STATE["launches"]:
        report.outcome = "failed"
        report.longrepr = "No Wafer kernel launch completed"
    if report.when == "call" or report.failed or report.skipped:
        record("test_result", nodeid=report.nodeid, outcome=report.outcome,
               phase=report.when, launches=STATE["launches"],
               detail=str(report.longrepr) if report.longrepr else None)


@pytest.fixture
def device_tensor():
    """Upload a guarded native storage, preserving input/output dtype and shape."""
    import torch
    allocations = []
    guard = 64

    def upload(host):
        assert host.device.type == "cpu" and host.is_contiguous()
        full = torch.full((host.numel() + 2 * guard,), -83, dtype=host.dtype)
        full[guard:-guard].copy_(host.reshape(-1))
        native = full.txda()
        allocations.append((native, full))
        return native[guard:-guard].view(host.shape)

    yield upload
    for native, expected in allocations:
        actual = native.cpu()
        try:
            torch.testing.assert_close(actual[:guard], expected[:guard], rtol=0, atol=0)
            torch.testing.assert_close(actual[-guard:], expected[-guard:], rtol=0, atol=0)
        except AssertionError as error:
            STATE["error"] = True
            record("memory_guard_error", error=str(error))
            raise
