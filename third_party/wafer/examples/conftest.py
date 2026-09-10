"""Wafer examples use CPU reference data and real device kernel execution.

The harness transports complete storages through Kuiper, preserving offsets,
strides and aliases. Run using scripts/run_wafer_example_suite.py.
"""
from _wafer_harness import (  # noqa: F401
    device,
    pytest_addoption,
    pytest_collection_finish,
    pytest_configure,
    pytest_runtest_makereport,
    pytest_runtest_setup,
)

# Former CPU-specific ignore/skip lists do not describe Wafer support.
collect_ignore = []
