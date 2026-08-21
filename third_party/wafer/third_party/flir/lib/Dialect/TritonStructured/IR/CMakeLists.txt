if(TARGET TritonStructuredIR)
  set_property(TARGET TritonStructuredIR PROPERTY SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/TritonStructuredOps.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TritonStructuredDialect.cpp
  )
  target_include_directories(TritonStructuredIR PRIVATE
    ${TRITON_SHARED_SOURCE_DIR}/include
    ${TRITON_SHARED_BINARY_DIR}/include
  )
  add_dependencies(TritonStructuredIR WaferTritonStructuredTableGen)
else()
  add_triton_library(TritonStructuredIR
    TritonStructuredOps.cpp
    TritonStructuredDialect.cpp

    DEPENDS
    TritonStructuredTableGen

    LINK_LIBS PUBLIC
    TritonIR
    MLIRIR
    )
endif()
