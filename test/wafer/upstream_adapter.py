"""Explicit host adapter for running unchanged Ascend tests through Wafer.

NPU tensor creation becomes CPU oracle storage. _wafer_harness uploads every
kernel argument storage, launches the real Wafer ELF, downloads outputs, and
checks guards. Vendor-only operators/profilers are deliberately unavailable.
This is a test transport, not a torch_npu implementation or a CPU kernel path.
"""
import functools
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'third_party/wafer/examples'))
import _wafer_harness as hardware

pytest_collection_finish = hardware.pytest_collection_finish
pytest_runtest_setup = hardware.pytest_runtest_setup
device = hardware.device
PATCH = None


def pytest_addoption(parser):
    hardware.pytest_addoption(parser)
    parser.addoption('--wafer-nodeids', type=Path,
                     help='JSON array of original nodeids selected for regression')


def pytest_collection_modifyitems(config, items):
    selection = config.getoption('--wafer-nodeids')
    if selection is None:
        return
    wanted = set(json.loads(selection.read_text()))
    collected = {item.nodeid for item in items}
    collected_files = {node.split('::', 1)[0] for node in collected}
    missing = {node for node in wanted if node.split('::', 1)[0] in collected_files} - collected
    if missing:
        raise pytest.UsageError(f'Regression nodeids no longer collect: {sorted(missing)}')
    retained = [item for item in items if item.nodeid in wanted]
    deselected = [item for item in items if item.nodeid not in wanted]
    items[:] = retained
    config.hook.pytest_deselected(items=deselected)
    hardware.record('regression_selection', manifest=str(selection),
                    selected=[item.nodeid for item in retained],
                    deselected=[item.nodeid for item in deselected])


def pytest_configure(config):
    global PATCH
    import torch
    import triton
    from triton.backends.dicp_triton.wafer_runtime import get_runtime

    hardware.pytest_configure(config)
    PATCH = pytest.MonkeyPatch()
    original_device = torch.device
    original_to = torch.Tensor.to

    def oracle_device(value):
        if isinstance(value, str) and (value == 'npu' or value.startswith('npu:')):
            if value not in ('npu', 'npu:0'):
                raise NotImplementedError('Wafer upstream adapter currently selects device 0 only')
            return 'cpu'
        return value

    class DeviceMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, original_device)

    class OracleDevice(metaclass=DeviceMeta):
        def __new__(cls, value, index=None):
            mapped = oracle_device(value)
            if mapped == 'cpu' and mapped != value:
                return original_device('cpu')
            return original_device(mapped) if index is None else original_device(mapped, index)

    def factory(original):
        @functools.wraps(original)
        def create(*args, **kwargs):
            if 'device' in kwargs:
                kwargs['device'] = oracle_device(kwargs['device'])
            return original(*args, **kwargs)
        return create

    def tensor_to(self, *args, **kwargs):
        if args:
            args = (oracle_device(args[0]), *args[1:])
        if 'device' in kwargs:
            kwargs['device'] = oracle_device(kwargs['device'])
        return original_to(self, *args, **kwargs)

    def tensor_npu(self, device=None, non_blocking=False, **kwargs):
        if device not in (None, 0, 'npu', 'npu:0'):
            raise NotImplementedError('Wafer upstream adapter currently selects device 0 only')
        return original_to(self, 'cpu', non_blocking=non_blocking, **kwargs)

    for name in ('tensor', 'as_tensor', 'empty', 'empty_like', 'empty_strided',
                 'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like',
                 'rand', 'rand_like', 'randn', 'randn_like', 'randint',
                 'randint_like', 'arange', 'linspace', 'logspace', 'eye', 'randperm'):
        PATCH.setattr(torch, name, factory(getattr(torch, name)))
    PATCH.setattr(torch, 'device', OracleDevice)
    PATCH.setattr(torch.Tensor, 'to', tensor_to)
    PATCH.setattr(torch.Tensor, 'npu', tensor_npu, raising=False)

    def current_stream(device=None):
        stream = get_runtime().current_stream(device)
        return SimpleNamespace(npu_stream=stream.txda_stream, synchronize=stream.synchronize)

    interface = SimpleNamespace(current_device=lambda: get_runtime().current_device(),
                                set_device=lambda device: get_runtime().set_device(device),
                                current_stream=current_stream,
                                synchronize=lambda: get_runtime().synchronize())
    interface.utils = SimpleNamespace(set_device=interface.set_device)
    PATCH.setattr(torch, 'npu', interface, raising=False)
    vendor = ModuleType('torch_npu')
    vendor.__file__ = __file__
    vendor.npu = interface

    def unsupported(name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(f'Ascend-only torch_npu.{name} is not provided by the Wafer host adapter')

    vendor.__getattr__ = unsupported
    PATCH.setitem(sys.modules, 'torch_npu', vendor)
    hardware.record('upstream_host_adapter', source_root=str(REPO / 'test/ascend/passed_tests'),
                    reference_device='cpu', kernel_backend=str(triton.runtime.driver.active.get_current_target()),
                    vendor_operators='unsupported; never silently emulated')


def pytest_collectreport(report):
    if report.failed:
        hardware.record('collection_error', nodeid=report.nodeid, detail=str(report.longrepr))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.when == 'call' or report.failed or report.skipped:
        hardware.record('test_result', outcome=report.outcome, phase=report.when,
                        detail=str(report.longrepr) if report.longrepr else None)
    if hardware.DEVICE_ERROR:
        pytest.exit('Stopping after a real device launch error', returncode=3)


def pytest_unconfigure(config):
    if PATCH:
        PATCH.undo()
