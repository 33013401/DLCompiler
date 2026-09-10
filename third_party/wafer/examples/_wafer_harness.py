"""Test-only host storage transport; kernels use the installed Wafer launcher."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest


STATE = {"nodeid": None, "launches": 0, "compiled": 0}
DEVICE_ERROR = False


def record(event, **values):
    path = os.getenv("WAFER_EXAMPLE_EVENTS")
    if path:
        with open(path, "a") as stream:
            stream.write(json.dumps({"event": event, "time": time.time(), **STATE, **values}, default=str) + "\n")


def pytest_addoption(parser):
    parser.addoption("--wafer-execution", choices=("compile", "hardware"), default="compile")


@pytest.fixture
def device():
    # Torch generates the oracle on CPU. Only Triton kernels run on Wafer.
    return "cpu"


def pytest_configure(config):
    import torch
    import triton
    from triton.backends.dicp_triton import wafer, wafer_runtime

    if os.getenv("WAFER_ENABLE_RUNTIME") != "1" or os.getenv("USE_SIM_MODE") != "0":
        raise pytest.UsageError("Wafer examples require WAFER_ENABLE_RUNTIME=1 USE_SIM_MODE=0")
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "scripts"))
    from audit_wafer_elf import audit_kernel

    original_tool = wafer._run_tool

    def run_tool(arguments):
        try:
            return original_tool(arguments)
        except subprocess.CalledProcessError:
            # Save failing compiler inputs before its TemporaryDirectory exits.
            event_path = os.getenv("WAFER_EXAMPLE_EVENTS")
            if event_path:
                inputs = [Path(arg) for arg in arguments if str(arg).endswith(".mlir") and Path(arg).is_file()]
                digest = hashlib.sha256(b"".join(path.read_bytes() for path in inputs)).hexdigest()[:16]
                directory = Path(event_path).parent / "failed-ir" / digest
                directory.mkdir(parents=True, exist_ok=True)
                replacements = {}
                for path in inputs:
                    destination = directory / path.name
                    shutil.copyfile(path, destination)
                    replacements[str(path)] = str(destination)
                command = [replacements.get(str(arg), str(arg)) for arg in arguments]
                if "-o" in command:
                    command[command.index("-o") + 1] = str(directory / "reproduced.mlir")
                (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                record("compiler_error", artifact_dir=str(directory), command=command)
            raise

    wafer._run_tool = run_tool

    # Select the native Kuiper runtime without invoking Torch vendor operators.
    runtime = wafer_runtime._KuiperRuntime()
    wafer_runtime.get_runtime = lambda: runtime
    driver = triton.runtime.driver.active
    target = driver.get_current_target()
    if target.backend != "wafer":
        raise pytest.UsageError(f"Expected Wafer target, got {target}")
    driver.get_active_torch_device = lambda: torch.device("cpu")
    config.addinivalue_line("markers", "interpreter: historical upstream marker; execution remains Wafer")
    original_launch = wafer_runtime.WaferLauncher.__call__
    transport = None
    checked = set()

    def launch(launcher, *args, **kwargs):
        global DEVICE_ERROR
        nonlocal transport
        metadata = launcher.metadata
        if metadata.kernel_path not in checked:
            audit_kernel(metadata.kernel_path, metadata.device_log_abi,
                         os.getenv("WAFER_NOC_FIRMWARE_ELF"))
            checked.add(metadata.kernel_path)
        STATE["compiled"] += 1
        record("compiled", kernel=metadata.name, path=metadata.kernel_path)
        if config.getoption("--wafer-execution") == "compile":
            if metadata.name == "__wafer_noc_init":
                # Let the dependent GEMM compile and be audited as well.
                return None
            pytest.skip("Compiled and ELF-audited; hardware execution not requested in this phase")
        if transport is None:
            transport = StorageTransport(runtime.library)
        buffers = {}
        arguments = list(args)
        try:
            for index in range(9, len(arguments)):
                value = arguments[index]
                # Triton's reinterpret wrapper retains the actual tensor in base.
                while not isinstance(value, torch.Tensor) and hasattr(value, "base"):
                    value = value.base
                if not isinstance(value, torch.Tensor):
                    continue
                if value.device.type != "cpu":
                    raise TypeError(f"Expected CPU oracle storage, got {value.device}")
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key not in buffers:
                    buffers[key] = transport.upload(storage)
                arguments[index] = buffers[key]["device"] + value.data_ptr() - storage.data_ptr()
            record("launch_start", kernel=metadata.name, grid=list(args[:3]))
            previous_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
            previous_timer = signal.setitimer(signal.ITIMER_REAL, float(os.getenv("WAFER_EXAMPLE_LAUNCH_TIMEOUT", "30")))
            try:
                result = original_launch(launcher, *arguments, **kwargs)
            except BaseException as error:
                DEVICE_ERROR = True
                record("launch_error", kernel=metadata.name, error=repr(error))
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
                signal.signal(signal.SIGALRM, previous_handler)
            STATE["launches"] += 1
            record("launch_complete", kernel=metadata.name)
            for buffer in buffers.values():
                transport.download(buffer)
            return result
        finally:
            for buffer in buffers.values():
                transport.free(buffer)

    wafer_runtime.WaferLauncher.__call__ = launch
    record("configured", target=str(target), execution=config.getoption("--wafer-execution"),
           triton_path=triton.__file__)


class StorageTransport:
    """Transfer a storage once, retaining alias offsets and checking guard bytes."""

    GUARD = 256

    def __init__(self, library):
        self.library = library
        for name, types in {
            "txMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64],
            "txFree": [ctypes.c_void_p],
            "txMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int],
            "txSetDevice": [ctypes.c_uint32],
        }.items():
            function = getattr(library, name)
            function.argtypes, function.restype = types, ctypes.c_int
        self.check(library.txSetDevice(0), "txSetDevice")

    @staticmethod
    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation} failed with status 0x{status:x}")

    def upload(self, storage):
        size = storage.nbytes()
        payload = ctypes.create_string_buffer(size + 2 * self.GUARD)
        ctypes.memset(payload, 0xA5, len(payload))
        if size:
            ctypes.memmove(ctypes.addressof(payload) + self.GUARD, storage.data_ptr(), size)
        pointer = ctypes.c_void_p()
        self.check(self.library.txMalloc(ctypes.byref(pointer), len(payload)), "txMalloc")
        try:
            self.check(self.library.txMemcpy(pointer, payload, len(payload), 1), "H2D")
        except BaseException:
            self.library.txFree(pointer)
            raise
        return {"allocation": pointer, "device": pointer.value + self.GUARD,
                "storage": storage, "payload": payload, "size": size}

    def download(self, buffer):
        payload, size = buffer["payload"], buffer["size"]
        self.check(self.library.txMemcpy(payload, buffer["allocation"], len(payload), 2), "D2H")
        guard = b"\xa5" * self.GUARD
        assert payload.raw[:self.GUARD] == guard, "Kernel wrote before the allocated storage"
        assert payload.raw[self.GUARD + size:] == guard, "Kernel wrote beyond the allocated storage"
        if size:
            ctypes.memmove(buffer["storage"].data_ptr(), ctypes.addressof(payload) + self.GUARD, size)

    def free(self, buffer):
        self.check(self.library.txFree(buffer["allocation"]), "txFree")


def pytest_collection_finish(session):
    record("collection", nodeids=[item.nodeid for item in session.items])


def pytest_runtest_setup(item):
    import numpy as np
    import torch
    STATE.update(nodeid=item.nodeid, launches=0, compiled=0)
    np.random.seed(0)
    torch.manual_seed(0)
    record("test_start")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.when == "call" or report.failed:
        record("test_result", outcome=report.outcome, phase=report.when,
               detail=str(report.longrepr) if report.longrepr else None)
    if DEVICE_ERROR:
        pytest.exit("Stopping this process after a device launch error", returncode=3)
