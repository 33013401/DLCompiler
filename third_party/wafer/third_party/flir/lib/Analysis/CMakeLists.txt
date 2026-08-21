add_triton_library(TritonSharedAnalysis
  MaskAnalysis.cpp
  OpFoldResultUtils.cpp
  PtrAnalysis.cpp
  UseAnalysis.cpp

  DEPENDS
  TritonTableGen
  TritonStructuredTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)

if(TARGET MLIRTritonNPUUtils)
  target_compile_definitions(TritonSharedAnalysis PRIVATE
    WAFER_USE_EXTERNAL_GET_INT_ATTR)
  target_link_libraries(TritonSharedAnalysis PUBLIC MLIRTritonNPUUtils)
endif()
