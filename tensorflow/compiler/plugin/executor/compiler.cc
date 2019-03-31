/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdlib.h>
#include <fstream>

//#include "tensorflow/compiler/plugin/executor/executable.h"
//#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
//#include "tensorflow/compiler/xla/service/computation_placer.h"
//#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
//#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
//#include "tensorflow/compiler/xla/service/hlo_cse.h"
//#include "tensorflow/compiler/xla/service/hlo_dce.h"
//#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
//#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
//#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
////#include "tensorflow/compiler/xla/service/inliner.h"
//#include "tensorflow/compiler/xla/service/layout_assignment.h"
//#include "tensorflow/compiler/xla/service/reshape_mover.h"
//#include "tensorflow/compiler/xla/status_macros.h"
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/stream_executor/lib/initialize.h"
////#include "tensorflow/stream_executor/lib/strcat.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/compiler/plugin/executor/executable.h"
#include "tensorflow/compiler/plugin/executor/compiler.h"

//namespace xla {
//namespace executorplugin {
namespace xla {
namespace executor {

///namespace se = ::perftools::gputools;
///namespace sep = ::perftools::gputools::executorplugin;

/*
 * Run optimization passes on the module.  The graph is transformed by
 * each pass in the optimization pipeline.  The service subdirectory
 * contains useful optimization passes.
 */
Status ExecutorCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("Executor");
  //pipeline.AddPass<Inliner>();
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<HloCSE>(false);

  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      false, [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<ReshapeMover>();
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<HloCSE>(true);
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());

  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  return pipeline.Run(hlo_module).status();
}

StatusOr<std::vector<std::unique_ptr<Executable>>> ExecutorCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  if (module_group->empty()) {
    return std::vector<std::unique_ptr<Executable>>();
  }
  if (module_group->size() > 1) {
    return tensorflow::errors::Unimplemented(
        "Compilation of multiple HLO modules is not supported on Interpreter.");
  }
  if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
    return tensorflow::errors::Unimplemented(
        "Unexpected number of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module,
                      RunHloPasses(std::move(hlo_modules[0]), stream_exec[0][0],
                                   device_allocator));
  TF_ASSIGN_OR_RETURN(
      auto executable,
      RunBackend(std::move(module), stream_exec[0][0], device_allocator));
  std::vector<std::unique_ptr<Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

//StatusOr<std::vector<std::unique_ptr<Executable>>> ExecutorCompiler::Compile(
//        std::vector<std::unique_ptr<HloModule>> hlo_modules,
//        std::vector<se::StreamExecutor*> stream_execs) {
//
//  return tensorflow::errors::Unimplemented(
//      "Compilation of multiple HLO modules is not supported on Executor.");
//}

//StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
//ExecutorCompiler::CompileAheadOfTime(
//    std::vector<std::unique_ptr<HloModule>> hlo_modules,
//    const AotCompilationOptions& aot_options) {
//
//  return tensorflow::errors::InvalidArgument(
//      "AOT compilation not supported on Executor");
//}
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
ExecutorCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& aot_options) {
  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on Interpreter");
}


se::Platform::Id ExecutorCompiler::PlatformId() const {
  //return sep::kExecutorPlatformId;
  return se::executor::kExecutorPlatformId;
}

HloCostAnalysis::ShapeSizeFunction
ExecutorCompiler::ShapeSizeBytesFunction() const {
  return ExecutorExecutable::ShapeSizeBytes;
}

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return xla::MakeUnique<xla::ComputationPlacer>();
}

REGISTER_MODULE_INITIALIZER(executor_compiler, {
  xla::Compiler::RegisterCompilerFactory(sep::kExecutorPlatformId, []() {
    return xla::MakeUnique<xla::executorplugin::ExecutorCompiler>();
  });
  xla::ComputationPlacer::RegisterComputationPlacer(sep::kExecutorPlatformId,
                                                    &CreateComputationPlacer);
});

}  // namespace executorplugin
}  // namespace xla
