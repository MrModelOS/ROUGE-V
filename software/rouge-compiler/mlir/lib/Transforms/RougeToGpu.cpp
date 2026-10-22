//=- ROUGE-V: rouge-to-gpu — the reverse lowering of our dialect to the
//   upstream `gpu` dialect, the step towards an NVPTX target.
//
//   The `rouge` dialect was deliberately kept close to `gpu` (the same SIMT
//   model, the same thread/block ids and barriers) so that this lowering is
//   mostly a rename. What it does:
//
//     * the SIMT ops become their gpu counterparts:
//         rouge.thread_id  -> gpu.thread_id
//         rouge.block_id   -> gpu.block_id
//         rouge.block_dim  -> gpu.block_dim
//         rouge.grid_dim   -> gpu.grid_dim
//         rouge.barrier    -> gpu.barrier
//     * the memory spaces become the gpu ones: the DEFAULT memref space is
//       the gpu GLOBAL space (1), the shared space (3) is the gpu workgroup
//       space (3) — the numbers coincide, but the meaning is now the gpu
//       dialect's;
//     * the kernel becomes a gpu.func (with the `kernel` attribute) inside a
//       gpu.module, and rouge.launch becomes gpu.launch_func.
//
//   The arithmetic, memref and scf ops are dialect-independent and stay as
//   they are.
//
//   rouge-opt --rouge-to-gpu kernel.mlir
//   mlir-opt --convert-gpu-to-nvvm -> LLVM IR (the kernel becomes an
//   llvm.func with the nvvm.kernel attribute) -> llc -march=nvptx64.
//
//=.

#include "rouge/Transforms/Passes.h"

#include "rouge/IR/RougeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace rouge {
namespace {

// The gpu address spaces (GPU_AddressSpaceEnum): global = 1, workgroup = 3,
// private = 2, constant = 0.
constexpr int64_t kGpuGlobalSpace = 1;

struct RougeToGpu : public PassWrapper<RougeToGpu, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RougeToGpu)

  StringRef getArgument() const final { return "rouge-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower the rouge dialect to the upstream gpu dialect (the first "
           "half of the NVPTX target)";
  }

  // The default memref space -> the gpu global space. The type is changed
  // IN PLACE (the value's type), so every use follows.
  static void convertMemrefSpaces(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    const Attribute defaultSpace =
        MemRefType::get({1}, Float32Type::get(ctx)).getMemorySpace();
    const Attribute globalSpace = IntegerAttr::get(IntegerType::get(ctx, 32),
                                                   kGpuGlobalSpace);
    auto convert = [&](Type t) -> std::optional<Type> {
      auto mr = dyn_cast<MemRefType>(t);
      if (!mr) return std::nullopt;
      if (mr.getMemorySpace() != defaultSpace) return std::nullopt;
      return MemRefType::get(mr.getShape(), mr.getElementType(), mr.getLayout(),
                             globalSpace);
    };
    // The functions: the signature and the block arguments.
    module.walk([&](func::FuncOp fn) {
      const auto fnType = fn.getFunctionType();
      SmallVector<Type> newInputs;
      bool changed = false;
      for (Type t : fnType.getInputs()) {
        auto n = convert(t);
        if (n) {
          newInputs.push_back(*n);
          changed = true;
        } else {
          newInputs.push_back(t);
        }
      }
      if (!changed) return;
      fn.setType(FunctionType::get(ctx, newInputs, fnType.getResults()));
      auto args = fn.getArguments();
      for (size_t i = 0; i < args.size() && i < newInputs.size(); ++i)
        args[i].setType(newInputs[i]);
    });
    // The allocations.
    module.walk([&](Operation *op) {
      if (!isa<memref::AllocOp, memref::AllocaOp>(op)) return;
      auto mr = cast<MemRefType>(op->getResult(0).getType());
      if (mr.getMemorySpace() != defaultSpace) return;
      op->getResult(0).setType(MemRefType::get(mr.getShape(), mr.getElementType(),
                                              mr.getLayout(), globalSpace));
    });
  }

  // A memref in the SHARED space is per-block scratch memory. On the GPU it
  // must be a STATIC .shared array: a memref.alloc in a non-default space
  // lowers to malloc, and a device function cannot call the host allocator
  // (the first MLIR fuzz run on the card failed exactly there). A module-level
  // memref.global in the same space is the static object: the LLVM lowering
  // turns it into an llvm.mlir.global in address space 3, which the NVPTX
  // backend emits as a .shared array, shared by the whole block.
  static void hoistSharedAllocs(Operation *scope) {
    unsigned counter = 0;
    SmallVector<memref::AllocOp> allocs;
    scope->walk([&](memref::AllocOp a) { allocs.push_back(a); });
    for (memref::AllocOp alloc : allocs) {
      auto type = dyn_cast<MemRefType>(alloc.getType());
      if (!type) continue;
      auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
      if (!space || space.getInt() != 3) continue;
      OpBuilder b(alloc);
      const std::string name = "rouge_smem_" + std::to_string(counter++);
      auto get = memref::GetGlobalOp::create(b, alloc.getLoc(), type, name);
      alloc.getResult().replaceAllUsesWith(get.getResult());
      alloc.erase();
      // Inside the scope that will own it: a gpu.module is IsolatedFromAbove
      // and a symbol table of its own, so a top-level global is INVISIBLE to
      // the kernel (the first version of this hoist failed exactly there).
      OpBuilder gb(scope->getContext());
      gb.setInsertionPointToStart(&scope->getRegion(0).front());
      memref::GlobalOp::create(gb, alloc.getLoc(), name,
                               /*sym_visibility=*/gb.getStringAttr("private"),
                               type, /*initial_value=*/Attribute(),
                               /*constant=*/false, /*alignment=*/IntegerAttr());
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.getContext()->loadDialect<gpu::GPUDialect>();
    // The gpu dialect requires the enclosing module to be marked as the
    // container of the gpu modules.
    if (!module->hasAttr("gpu.container_module"))
      module->setAttr("gpu.container_module",
                       UnitAttr::get(module.getContext()));

    // The SIMT ops first: they live inside the kernel, and the kernel is
    // moved into the gpu module below.
    SmallVector<Operation *> toConvert;
    module.walk([&](Operation *op) {
      if (isa<rouge::ThreadIdOp, rouge::BlockIdOp, rouge::BlockDimOp,
              rouge::GridDimOp, rouge::BarrierOp>(op))
        toConvert.push_back(op);
    });
    for (Operation *op : toConvert) {
      if (auto tid = dyn_cast<rouge::ThreadIdOp>(op)) {
        OpBuilder b(tid.getContext());
        b.setInsertionPoint(tid);
        auto gpuOp = gpu::ThreadIdOp::create(
            b, tid.getLoc(), b.getIndexType(),
            static_cast<gpu::Dimension>(tid.getDimension()));
        tid.getResult().replaceAllUsesWith(gpuOp.getResult());
        tid.erase();
      } else if (auto bid = dyn_cast<rouge::BlockIdOp>(op)) {
        OpBuilder b(bid.getContext());
        b.setInsertionPoint(bid);
        auto gpuOp = gpu::BlockIdOp::create(
            b, bid.getLoc(), b.getIndexType(),
            static_cast<gpu::Dimension>(bid.getDimension()));
        bid.getResult().replaceAllUsesWith(gpuOp.getResult());
        bid.erase();
      } else if (auto bdim = dyn_cast<rouge::BlockDimOp>(op)) {
        OpBuilder b(bdim.getContext());
        b.setInsertionPoint(bdim);
        auto gpuOp = gpu::BlockDimOp::create(
            b, bdim.getLoc(), b.getIndexType(),
            static_cast<gpu::Dimension>(bdim.getDimension()));
        bdim.getResult().replaceAllUsesWith(gpuOp.getResult());
        bdim.erase();
      } else if (auto gdim = dyn_cast<rouge::GridDimOp>(op)) {
        OpBuilder b(gdim.getContext());
        b.setInsertionPoint(gdim);
        auto gpuOp = gpu::GridDimOp::create(
            b, gdim.getLoc(), b.getIndexType(),
            static_cast<gpu::Dimension>(gdim.getDimension()));
        gdim.getResult().replaceAllUsesWith(gpuOp.getResult());
        gdim.erase();
      } else if (auto bar = dyn_cast<rouge::BarrierOp>(op)) {
        OpBuilder b(bar.getContext());
        b.setInsertionPoint(bar);
        b.create<gpu::BarrierOp>(bar.getLoc());
        bar.erase();
      }
    }

    // The memory spaces: the default one becomes the gpu global one.
    convertMemrefSpaces(module);

    // The launches: the kernel moves into a gpu.module as a gpu.func, and
    // the launch becomes a gpu.launch_func in the host.
    SmallVector<rouge::LaunchOp> launches;
    module.walk([&](rouge::LaunchOp launch) { launches.push_back(launch); });
    unsigned moduleCounter = 0;
    for (rouge::LaunchOp launch : launches) {
      auto kernel = module.lookupSymbol<func::FuncOp>(launch.getKernel());
      if (!kernel) {
        launch.emitError("the kernel of the launch is not found");
        signalPassFailure();
        return;
      }
      OpBuilder b(launch.getContext());
      b.setInsertionPoint(launch);

      // The gpu module: one per kernel, in the TOP-LEVEL module (the gpu
      // module is a sibling of the host function, not a nested symbol).
      const std::string gpuModuleName =
          kernel.getName().str() + "_gpu_module";
      OpBuilder moduleBuilder(launch.getContext());
      moduleBuilder.setInsertionPointToEnd(module.getBody());
      auto gpuModule = gpu::GPUModuleOp::create(moduleBuilder, launch.getLoc(),
                                                gpuModuleName);

      // The kernel: a gpu.func with the `kernel` attribute.
      b.setInsertionPointToEnd(&gpuModule.getBodyRegion().front());
      auto gpuFunc = gpu::GPUFuncOp::create(
          b, launch.getLoc(), kernel.getName(), kernel.getFunctionType(),
          /*workgroupAttributions=*/{}, /*privateAttributions=*/{},
          ArrayRef<NamedAttribute>{b.getNamedAttr(
              "kernel", b.getUnitAttr())});
      gpuFunc.getBody().takeBody(kernel.getBody());
      // The shared scratch buffers become static .shared objects of this gpu
      // module (the kernel cannot see the top-level module's symbols).
      hoistSharedAllocs(gpuModule);
      // A symbol inside a gpu.module is private: the module is the boundary.
      gpuFunc.setVisibility(SymbolTable::Visibility::Private);
      kernel.erase();
      // The terminator: func.return -> gpu.return.
      gpuFunc.getBody().walk([&](func::ReturnOp ret) {
        OpBuilder rb(ret.getContext());
        rb.setInsertionPoint(ret);
        rb.create<gpu::ReturnOp>(ret.getLoc());
        ret.erase();
      });
      b.setInsertionPoint(launch);

      // The launch: gpu.launch_func.
      gpu::KernelDim3 gridSize{launch.getGridSizeX(), launch.getGridSizeY(),
                               launch.getGridSizeZ()};
      gpu::KernelDim3 blockSize{launch.getBlockSizeX(), launch.getBlockSizeY(),
                               launch.getBlockSizeZ()};
      auto launchFunc = gpu::LaunchFuncOp::create(
          b, launch.getLoc(), gpuFunc, gridSize, blockSize,
          /*dynamicSharedMemorySize=*/Value(), launch.getKernelOperands());
      launch.erase();
    }
  }
};

} // namespace

void registerRougeToGpuPass() {
  PassRegistration<RougeToGpu>();
}

std::unique_ptr<mlir::Pass> createRougeToGpuPass() {
  return std::make_unique<RougeToGpu>();
}

} // namespace rouge