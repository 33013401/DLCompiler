if(NOT DEFINED TX8_DEPS_ROOT)
  if(DEFINED ENV{TX8_DEPS_ROOT})
    set(TX8_DEPS_ROOT $ENV{TX8_DEPS_ROOT})
  else()
    message(STATUS "TX8_DEPS_ROOT not set, CRT/Profiler will be disabled")
    set(WAFER_ENABLE_CRT OFF)
  endif()
endif()

# Check if TX8_DEPS_ROOT exists to enable CRT/Profiler
if(DEFINED TX8_DEPS_ROOT AND EXISTS ${TX8_DEPS_ROOT})
  set(WAFER_ENABLE_CRT ON)
else()
  set(WAFER_ENABLE_CRT OFF)
  if(DEFINED TX8_DEPS_ROOT)
    message(WARNING "TX8_DEPS_ROOT set but path does not exist: ${TX8_DEPS_ROOT}")
  endif()
endif()
# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

if(NOT DEFINED USE_HOST_PROFILE)
  if(DEFINED ENV{USE_HOST_PROFILE})
    set(USE_HOST_PROFILE $ENV{USE_HOST_PROFILE})
  endif()
endif()

if(NOT DEFINED ENABLE_PROFILING)
  if(DEFINED ENV{ENABLE_PROFILING})
    set(ENABLE_PROFILING $ENV{ENABLE_PROFILING})
  endif()
endif()

if(NOT DEFINED NO_INTRNISIC_RUN)
  if(DEFINED ENV{NO_INTRNISIC_RUN})
    set(NO_INTRNISIC_RUN $ENV{NO_INTRNISIC_RUN})
  endif()
endif()

if(NOT DEFINED ENABLE_SYNCHRONOUS_INTRINSIC)
  if(DEFINED ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
    set(ENABLE_SYNCHRONOUS_INTRINSIC $ENV{ENABLE_SYNCHRONOUS_INTRINSIC})
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set FLAGTREE_BACKEND for FLIR to skip test/tools directories
set(FLAGTREE_BACKEND "wafer")

set(XUANTIE_NAME Xuantie-900-gcc-elf-newlib-x86_64-V2.10.2)
set(INSTALL_WAFER_DIR ${CMAKE_INSTALL_PREFIX}/triton/backends/wafer/)
install(CODE "file(MAKE_DIRECTORY \"${INSTALL_WAFER_DIR}\")")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/crt/include)

if(NOT DEFINED WAFER_TX8_INCLUDE_DIR OR
   NOT EXISTS "${WAFER_TX8_INCLUDE_DIR}/instr_def.h")
  message(FATAL_ERROR
    "WAFER_TX8_INCLUDE_DIR must contain instr_def.h for Wafer lowering")
endif()
include_directories(${WAFER_TX8_INCLUDE_DIR})

# TX8_DEPS_ROOT include only if CRT enabled
if(WAFER_ENABLE_CRT)
  include_directories(${TX8_DEPS_ROOT}/include)
endif()

# FLIR (FlagTree Intermediate Representation) as local submodule
# FLIR provides TritonTilingExt and TritonStructured dialects
set(FLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flir)
set(FLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/flir)
include_directories(${FLIR_SOURCE_DIR}/include)
include_directories(${FLIR_BINARY_DIR}/include)

# TLE (Wafer Language Extension) - DSA dialect needed by TLEToMK pass
# Bundled in third_party/tle so wafer is independent of FlagTree
set(TLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tle)
set(TLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/tle)
# Make "tle/include/..." includes work (for lib/Conversion/TLEToMK/TLEToMK.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party)
# Make "third_party/tle/include/..." includes work (for bin/RegisterTritonDialects.h)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# When built via FlagTree (cmake source), FlagTree builds TLE itself.
# Pass -DWAFER_USE_EXTERNAL_TLE=ON to let the parent cmake provide TLE.
option(WAFER_USE_EXTERNAL_TLE "Use TLE provided by the parent cmake (e.g. FlagTree)" OFF)
if(NOT WAFER_USE_EXTERNAL_TLE)
    add_subdirectory(third_party/tle)
endif()
add_subdirectory(lib)
add_subdirectory(include)
add_subdirectory(third_party/flir)
add_subdirectory(bin)

# Conditionally add CRT and Profiler
if(WAFER_ENABLE_CRT)
  add_subdirectory(crt)
  add_subdirectory(profiler)
else()
  message(STATUS "Building wafer without CRT/Profiler")
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  # FIXME: Unify the libraries for Wafer into fewer ones
  # FLIR libraries (from third_party/flir):
  # - TritonTilingExtIR, TritonStructuredIR: Dialect libraries
  # - TritonToLinalg, TritonToStructured: Conversion passes
  add_triton_plugin(TritonWafer ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_wafer.cc
                    LINK_LIBS TritonSharedAnalysis TritonSharedAnalysisStructured MagicKernelIR
                    Tx81IR TritonTilingExtIR TritonStructuredIR TritonToCoreDialects
                    TritonToLinalg TritonToStructured StructuredToMemref LinalgToMagicKernel
                    TritonArithToLinalg CoreDialectsToMK Tx81ToLLVM AllocateSharedMemory ExportKernelSymbols
                    Tx81MemrefToLLVM MKToTx81
                    MLIRFuncAllExtensions)
  target_link_libraries(TritonWafer PRIVATE Python3::Module pybind11::headers)
endif()
#if(TRITON_BUILD_UT)
#  add_subdirectory(unittest)
#endif()
#add_subdirectory(test)
