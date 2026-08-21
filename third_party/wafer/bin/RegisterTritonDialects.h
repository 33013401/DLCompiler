#pragma once
#include "Address/Dialect/IR/AddressDialect.h"
#include "Address/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/AllInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "magic-kernel/Conversion/TLEToMK/Passes.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "third_party/tle/include/tle-dsa/Conversion/DsaToCore/DsaToCore.h"
#include "third_party/tle/include/tle-dsa/Dialect/IR/DsaDialect.h"
#include "triton-shared/Conversion/ConvertTritonPtr/Passes.h"
#include "triton-shared/Conversion/ReconcilePtrCasts/Passes.h"
#include "triton-shared/Conversion/StructuredToMemref/Passes.h"
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonPtrToMemref/Passes.h"
#include "triton-shared/Conversion/TritonToCoreDialects/Passes.h"
#include "triton-shared/Conversion/TritonToLinalg/Passes.h"
#include "triton-shared/Conversion/TritonToStructured/Passes.h"
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h"
#include "triton-shared/Conversion/UnstructuredToMemref/Passes.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "wafer-tx81/Conversion/LinalgFusion/Passes.h"
#include "wafer-tx81/Conversion/LinalgTiling/Passes.h"
#include "wafer-tx81/Dialect/IR/Tx81Dialect.h"

#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h"
#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h"
#include "magic-kernel/Conversion/LinalgToMK/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "wafer-tx81/Conversion/AllocateSharedMemory/Passes.h"
#include "wafer-tx81/Conversion/ExportKernelSymbols/Passes.h"
#include "wafer-tx81/Conversion/MKToTx81/Passes.h"
#include "wafer-tx81/Conversion/Tx81MemrefToLLVM/Passes.h"
#include "wafer-tx81/Conversion/Tx81ToLLVM/KernelArgBufferPass.h"
#include "wafer-tx81/Conversion/Tx81ToLLVM/Passes.h"

#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
    mlir::triton::registerTritonPasses();
  mlir::registerLinalgPasses();
  mlir::dsa::registerDsaMemoryToCorePass();
  mlir::triton::registerTLEToMKPass();
    mlir::triton::nvidia_gpu::registerTritonNvidiaGPUPasses();
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonToUnstructuredPass();
  mlir::triton::registerTritonArithToLinalgPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerStructuredToMemrefPasses();
  mlir::triton::registerUnstructuredToMemref();
  mlir::triton::registerTritonPtrToMemref();
  mlir::triton::registerTritonToCoreDialectsPass();
  mlir::triton::registerReconcilePtrCasts();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::registerLLVMDIScope();

  // Core dialects to MK layer conversion passes
  mlir::triton::registerTx81MemrefToLLVMPass();
  mlir::triton::registerLinalgToMKPass();
  mlir::triton::registerCoreDialectsToMKPass();
  mlir::triton::registerLegalizeTensorFormLoopsPass();
  mlir::addr::registerAddrToLLVMPass();
  mlir::triton::registerLinalgTilingPass();
  mlir::triton::registerLinalgFusionPass();

  // Wafer specific conversion passes
  mlir::triton::registerMKToTx81Pass();
  mlir::triton::alloc::registerAllocateSharedMemoryPass();
  mlir::triton::registerTx81ToLLVMPass();
  mlir::triton::registerExportKernelSymbols();
  mlir::triton::registerKernelArgBufferPass();

  // Register LLVM 22's standard external models and Wafer's custom model.
  mlir::registerAllExtensions(registry);
  mlir::linalg::registerAllDialectInterfaceImplementations(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::mk::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect,
      mlir::ttx::TritonTilingExtDialect, mlir::tts::TritonStructuredDialect,
      mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
      mlir::tensor::TensorDialect, mlir::memref::MemRefDialect,
      mlir::affine::AffineDialect, mlir::bufferization::BufferizationDialect,
      mlir::mk::MagicKernelDialect, mlir::tx::Tx81Dialect,
      mlir::addr::AddressDialect, mlir::dsa::DsaDialect>();
}
