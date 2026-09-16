"""Exercise vendor loader contracts without loading a device program."""

import importlib
import sys
from types import SimpleNamespace

import pytest


def driver_for(wafer_modules, target, load, cpu=False):
    module = importlib.import_module("_wafer_under_test.driver")
    driver = object.__new__(module.DICPDriver)
    driver.target = target
    driver.is_cpu_verify = cpu
    driver.utils = SimpleNamespace(load_binary=load)
    return driver


@pytest.mark.parametrize("target", ["wafer", "nvidia", "ascend", "maca", "mlu"])
def test_vendor_loader_arguments_and_results(wafer_modules, target):
    calls = []
    handles = (object(), object(), 7, 2)
    native = handles + (1024,) if target in ("wafer", "nvidia") else handles

    def load(*args):
        calls.append(args)
        return native

    metadata = SimpleNamespace(shared=64, kernel_name="ascend_entry", mix_mode="MIX_AIC")
    driver = driver_for(wafer_modules, target, load)
    result = driver.load_binary_for_triton("jit_entry", b"ELF", metadata, 3)
    expected_args = (("ascend_entry", b"ELF", 64, 3, "MIX_AIC") if target == "ascend"
                     else ("jit_entry", b"ELF", 64, 3))
    assert calls == [expected_args]
    assert result == (native if len(native) == 5 else handles + (sys.maxsize,))


def test_cpu_verify_does_not_use_ascend_signature(wafer_modules):
    calls = []

    def load(name, binary, shared, device):
        calls.append((name, binary, shared, device))
        return None, binary, None, None

    driver = driver_for(wafer_modules, "ascend", load, cpu=True)
    result = driver.load_binary_for_triton("cpu_entry", b"object", SimpleNamespace(shared=0), 0)
    assert calls == [("cpu_entry", b"object", 0, 0)]
    assert result == (None, b"object", None, None, sys.maxsize)


@pytest.mark.parametrize("target", ["ascend", "wafer"])
def test_real_loader_type_error_is_not_retried(wafer_modules, target):
    calls = []

    def load(*args):
        calls.append(args)
        raise TypeError("invalid binary in vendor loader")

    driver = driver_for(wafer_modules, target, load)
    metadata = SimpleNamespace(shared=0, kernel_name="entry", mix_mode="AIC")
    with pytest.raises(TypeError, match="invalid binary"):
        driver.load_binary_for_triton("entry", b"bad", metadata, 0)
    assert len(calls) == 1
