#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonToLinalg)
if(TARGET TritonToLinalgConversionPassIncGen)
  add_public_tablegen_target(WaferTritonToLinalgConversionPassIncGen)
  add_dependencies(TritonToLinalgConversionPassIncGen WaferTritonToLinalgConversionPassIncGen)
else()
  add_public_tablegen_target(TritonToLinalgConversionPassIncGen)
endif()
