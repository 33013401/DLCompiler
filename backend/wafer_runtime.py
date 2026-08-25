import ctypes
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path

from triton.runtime.cache import get_cache_manager


def _sdk_path(name):
    root = os.getenv("KUIPER_ROOT")
    if not root:
        raise RuntimeError("KUIPER_ROOT is not set; source init_tx81_env.sh first.")
    return os.path.join(root, name)


class _KuiperRuntime:
    def __init__(self):
        self.library = ctypes.CDLL(os.path.join(_sdk_path("lib"), "libhpgr.so"))
        self.library.txGetDevice.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self.library.txGetDevice.restype = ctypes.c_int
        self.library.txSetDevice.argtypes = [ctypes.c_uint32]
        self.library.txSetDevice.restype = ctypes.c_int

    def current_device(self):
        device = ctypes.c_uint32()
        status = self.library.txGetDevice(ctypes.byref(device))
        if status != 0:
            raise RuntimeError(f"txGetDevice failed with status 0x{status:x}")
        return device.value

    def set_device(self, device):
        status = self.library.txSetDevice(device)
        if status != 0:
            raise RuntimeError(f"txSetDevice failed with status 0x{status:x}")


def get_runtime():
    try:
        import torch
        import torch_txda  # noqa: F401
        if hasattr(torch, "txda"):
            return torch.txda
    except (ImportError, AttributeError):
        pass
    return _KuiperRuntime()


def _build_launcher(name, source, directory):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = os.path.join(directory, f"{name}{suffix}")
    compiler = os.getenv("CXX") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Failed to find a C++ compiler; set CXX.")
    include_dirs = [_sdk_path("include"), sysconfig.get_path("include")]
    library_dirs = [_sdk_path("lib")]
    libraries = ["hpgr"]
    command = [compiler, source, "-O3", "-shared", "-fPIC", "-std=c++17", "-Wno-psabi", "-o", output]
    command += [f"-I{path}" for path in include_dirs]
    command += [f"-L{path}" for path in library_dirs]
    command += [f"-l{library}" for library in libraries]
    subprocess.check_call(command)
    return output


def compile_launcher(source):
    name = "__triton_launcher"
    cache = get_cache_manager(hashlib.sha256(source.encode("utf-8")).hexdigest())
    cache_path = cache.get_file(f"{name}.so")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, f"{name}.cpp")
            Path(source_path).write_text(source, encoding="utf-8")
            shared_object = _build_launcher(name, source_path, directory)
            cache_path = cache.put(Path(shared_object).read_bytes(), f"{name}.so", binary=True)
    spec = importlib.util.spec_from_file_location(name, cache_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cpp_type(type_name):
    if type_name.startswith("*"):
        return "PyObject*"
    return {
        "i1": "int32_t", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t",
        "u1": "uint32_t", "u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t",
        "fp16": "float", "bf16": "float", "fp32": "float", "f32": "float", "fp64": "double",
    }[type_name]


def _parse_format(type_name):
    if type_name.startswith("*"):
        return "O"
    return {
        "int8_t": "b", "int16_t": "h", "int32_t": "i", "int64_t": "L",
        "uint8_t": "B", "uint16_t": "H", "uint32_t": "I", "uint64_t": "K",
        "float": "f", "double": "d",
    }[_cpp_type(type_name)]


def make_launcher(signature):
    declarations = " ".join(f"{_cpp_type(type_name)} arg{index};" for index, type_name in signature.items())
    parse_format = "iiiOKOOOO" + "".join(_parse_format(type_name) for type_name in signature.values())
    parse_args = "".join(f", &arg{index}" for index in signature)
    pointer_setup = "\n".join(
        f"void *ptr{index} = get_pointer(arg{index}); if (PyErr_Occurred()) return NULL;"
        for index, type_name in signature.items() if type_name.startswith("*")
    )
    kernel_args = "\n".join(
        f"runtime_args.push_back(1); runtime_args.push_back((uint64_t)ptr{index});"
        if type_name.startswith("*") else
        f"uint64_t scalar{index} = 0; memcpy(&scalar{index}, &arg{index}, sizeof(arg{index})); runtime_args.push_back(scalar{index});"
        for index, type_name in signature.items()
    )
    return f'''
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "tx_runtime.h"

static void *get_pointer(PyObject *object) {{
    if (object == Py_None) return nullptr;
    if (PyLong_Check(object)) return PyLong_AsVoidPtr(object);
    PyObject *value = PyObject_CallMethod(object, "data_ptr", nullptr);
    if (!value) return nullptr;
    void *pointer = PyLong_AsVoidPtr(value);
    Py_DECREF(value);
    return pointer;
}}

static PyObject *launch(PyObject *, PyObject *args) {{
    int grid_x, grid_y, grid_z;
    unsigned long long function;
    PyObject *stream_object, *kernel_metadata, *launch_metadata, *enter_hook, *exit_hook;
    {declarations}
    if (!PyArg_ParseTuple(args, "{parse_format}", &grid_x, &grid_y, &grid_z, &stream_object, &function,
            &kernel_metadata, &launch_metadata, &enter_hook, &exit_hook{parse_args})) return NULL;
    if (enter_hook != Py_None) {{
        PyObject *result = PyObject_CallFunctionObjArgs(enter_hook, launch_metadata, NULL);
        if (!result) return NULL;
        Py_DECREF(result);
    }}
    {pointer_setup}
    std::vector<uint64_t> runtime_args;
    {kernel_args}
    runtime_args.insert(runtime_args.end(), {{(uint64_t)grid_x, (uint64_t)grid_y, (uint64_t)grid_z, 0, 0, 0}});
    PyObject *path_object = PyObject_GetAttrString(kernel_metadata, "kernel_path");
    PyObject *name_object = PyObject_GetAttrString(kernel_metadata, "name");
    const char *kernel_path = PyUnicode_AsUTF8(path_object);
    const char *kernel_name = PyUnicode_AsUTF8(name_object);
    FILE *file = fopen(kernel_path, "rb");
    if (!file) {{ Py_DECREF(path_object); Py_DECREF(name_object); PyErr_SetFromErrnoWithFilename(PyExc_OSError, kernel_path); return NULL; }}
    fseek(file, 0, SEEK_END); size_t size = ftell(file); rewind(file);
    void *binary = malloc(size);
    if (!binary || fread(binary, 1, size, file) != size) {{ fclose(file); free(binary); Py_DECREF(path_object); Py_DECREF(name_object); PyErr_SetString(PyExc_RuntimeError, "Failed to read Wafer kernel"); return NULL; }}
    fclose(file);
    txStream_t stream = stream_object == Py_None ? nullptr : (txStream_t)PyLong_AsVoidPtr(stream_object);
    txError_t status = txLaunchKernelGGL(kernel_name, (uint64_t)binary, size,
        dim3({{(uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z}}), dim3({{1, 1, 1}}),
        runtime_args.data(), runtime_args.size() * sizeof(uint64_t), 0, stream);
    if (status == TX_SUCCESS) status = txStreamSynchronize(stream);
    free(binary);
    Py_DECREF(path_object);
    Py_DECREF(name_object);
    if (status != TX_SUCCESS) {{ PyErr_SetString(PyExc_RuntimeError, "Wafer kernel launch failed"); return NULL; }}
    if (exit_hook != Py_None) {{
        PyObject *result = PyObject_CallFunctionObjArgs(exit_hook, launch_metadata, NULL);
        if (!result) return NULL;
        Py_DECREF(result);
    }}
    Py_RETURN_NONE;
}}
static PyMethodDef methods[] = {{{{"launch", launch, METH_VARARGS, "Launch a Wafer kernel"}}, {{NULL, NULL, 0, NULL}}}};
static struct PyModuleDef module = {{PyModuleDef_HEAD_INIT, "__triton_launcher", NULL, -1, methods}};
PyMODINIT_FUNC PyInit___triton_launcher(void) {{ return PyModule_Create(&module); }}
'''


class TXDAUtils:
    def load_binary(self, name, kernel, shared_mem, device):
        return None, 0, 0, 0, 1024

    def get_device_properties(self, device=None):
        return {"max_shared_mem": 3 * 1024 * 1024 - 2 * 0x10000}


class SimulatorUtils:
    def load_binary(self, name, kernel, shared_mem, device):
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".so", delete=False) as file:
            file.write(kernel)
            path = file.name
        import ctypes

        module = ctypes.CDLL(path)
        os.unlink(path)
        function = ctypes.cast(getattr(module, name), ctypes.c_void_p).value
        return module, function, 0, 0, 1024

    def get_device_properties(self, device=None):
        return {"max_shared_mem": 3 * 1024 * 1024 - 2 * 0x10000}


class TXDALauncher:
    def __init__(self, src, metadata):
        argument_names = getattr(getattr(src, "fn", None), "arg_names", ())

        def argument_index(key):
            key = key[0] if isinstance(key, tuple) else key
            return argument_names.index(key) if isinstance(key, str) else int(key)

        signature = {argument_index(index): type_name for index, type_name in src.signature.items()}
        constants = {argument_index(index) for index in getattr(src, "constants", {})}
        signature = {index: type_name for index, type_name in signature.items() if index not in constants}
        self.metadata = metadata
        self.launch = compile_launcher(make_launcher(signature)).launch

    def __call__(self, *args, **kwargs):
        arguments = list(args)
        arguments[5] = self.metadata
        return self.launch(*arguments, **kwargs)