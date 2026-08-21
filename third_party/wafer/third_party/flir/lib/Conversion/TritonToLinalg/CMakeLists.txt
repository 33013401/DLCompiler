#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

if(TARGET TritonToLinalg)
  get_filename_component(DICP_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../.." ABSOLUTE)
  set_property(TARGET TritonToLinalg PROPERTY SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DicpCompatibility.cpp
    ${DICP_SOURCE_DIR}/compiler/lib/TritonToLinalg/AscendNPUIRLegalizePass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TritonToLinalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TritonToLinalgPass.cpp
  )
  target_compile_definitions(tritonDicpTriton PRIVATE
    DICP_USE_WAFER_TRITON_TO_LINALG=1)
  target_include_directories(TritonToLinalg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../include
    ${TRITON_SHARED_SOURCE_DIR}/include
    ${TRITON_SHARED_BINARY_DIR}/include
  )
  target_link_libraries(TritonToLinalg PUBLIC
    TritonTilingExtIR
    TritonSharedAnalysis
  )
  add_dependencies(TritonToLinalg
    WaferTritonToLinalgConversionPassIncGen
    MLIRMathExtDialectIncGen
    MLIRMathExtOpsIncGen
  )
else()
  add_triton_library(TritonToLinalg
    TritonToLinalg.cpp
    TritonToLinalgPass.cpp

    DEPENDS
    TritonToLinalgConversionPassIncGen
    MLIRMathExtDialectIncGen
    MLIRMathExtOpsIncGen

    LINK_LIBS PUBLIC
    TritonTilingExtIR
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRMathDialect
    MLIRPass
    MLIRTensorDialect
    MLIRTransforms
    MLIRSupport
    TritonIR
    TritonTransforms
    TritonSharedAnalysis
  )
endif()
