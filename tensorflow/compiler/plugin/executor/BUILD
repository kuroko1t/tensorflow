licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "plugin_lib",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:xla_headers_lib",
        "//tensorflow/compiler/xla/service",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@protobuf_archive//:protobuf_headers",
    ],
    alwayslink = 1,
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

#load(
#    "//tensorflow/core:platform/default/build_config_root.bzl",
#    "if_static",
#)
#
#licenses(["notice"])  # Apache 2.0
#
#package(default_visibility = ["//visibility:public"])
#
#cc_library(
#    name = "transfer_manager",
#    srcs = ["transfer_manager.cc"],
#    hdrs = ["transfer_manager.h"],
#    deps = [
#        ":platform_id",
#        "//tensorflow/core:lib",
#        "@com_google_absl//absl/memory",
#    ],
#    alwayslink = True,  # Contains per-platform transfer manager registration
#)
#
#cc_library(
#    name = "compiler",
#    srcs = ["compiler.cc"],
#    hdrs = ["compiler.h"],
#    deps = [
#        ":executable",
#        ":platform_id",
#        "//tensorflow/compiler/xla:status",
#        "//tensorflow/compiler/xla:status_macros",
#        "//tensorflow/compiler/xla:statusor",
#        #"//tensorflow/compiler/plugin/executor:algebraic_simplifier",
#        #"//tensorflow/compiler/plugin/executor:cholesky_expander",
#        #"//tensorflow/compiler/plugin/executor:compiler",
#        #"//tensorflow/compiler/plugin/executor:computation_placer",
#        #"//tensorflow/compiler/plugin/executor:dynamic_index_splitter",
#        #"//tensorflow/compiler/plugin/executor:executable",
#        #"//tensorflow/compiler/plugin/executor:flatten_call_graph",
#        #"//tensorflow/compiler/plugin/executor:hlo",
#        #"//tensorflow/compiler/plugin/executor:hlo_constant_folding",
#        #"//tensorflow/compiler/plugin/executor:hlo_cost_analysis",
#        #"//tensorflow/compiler/plugin/executor:hlo_cse",
#        #"//tensorflow/compiler/plugin/executor:hlo_dce",
#        #"//tensorflow/compiler/plugin/executor:hlo_module_config",
#        #"//tensorflow/compiler/plugin/executor:hlo_pass",
#        #"//tensorflow/compiler/plugin/executor:hlo_pass_pipeline",
#        #"//tensorflow/compiler/plugin/executor:hlo_subcomputation_unification",
#        #"//tensorflow/compiler/plugin/executor:layout_assignment",
#        #"//tensorflow/compiler/plugin/executor:map_inliner",
#        #"//tensorflow/compiler/plugin/executor:reduce_precision_insertion",
#        #"//tensorflow/compiler/plugin/executor:reshape_mover",
#        #"//tensorflow/compiler/plugin/executor:triangular_solve_expander",
#        #"//tensorflow/compiler/plugin/executor:while_loop_simplifier",
#        #"//tensorflow/compiler/plugin/executor/cpu:custom_call_target_registry",
#        "//tensorflow/core:lib",
#        "//tensorflow/stream_executor",
#        "@com_google_absl//absl/memory",
#    ],
#    alwayslink = True,  # Contains compiler registration
#)
#
#cc_library(
#    name = "platform_id",
#    srcs = ["platform_id.cc"],
#    hdrs = ["platform_id.h"],
#    deps = ["//tensorflow/core:stream_executor_headers_lib"] + if_static(
#        ["@protobuf_archive//:protobuf"],
#        ["@protobuf_archive//:protobuf_headers"],
#    ),
#)
#
#cc_library(
#    name = "executable",
#    srcs = ["executable.cc"],
#    hdrs = ["executable.h"],
#    deps = [
#        ":executor",
#        "//tensorflow/compiler/xla:literal",
#        "//tensorflow/compiler/xla:shape_util",
#        "//tensorflow/compiler/xla:status_macros",
#        "//tensorflow/compiler/xla:statusor",
#        "//tensorflow/compiler/xla:types",
#        "//tensorflow/compiler/xla:xla_data_proto",
#        #"//tensorflow/compiler/plugin/executor:executable",
#        #"//tensorflow/compiler/plugin/executor:hlo",
#        #"//tensorflow/compiler/plugin/executor:hlo_cost_analysis",
#        #"//tensorflow/compiler/plugin/executor:hlo_evaluator",
#        #"//tensorflow/compiler/plugin/executor:hlo_execution_profile",
#        #"//tensorflow/compiler/plugin/executor:hlo_module_config",
#        #"//tensorflow/compiler/plugin/executor:shaped_buffer",
#        "//tensorflow/compiler/plugin/executor:transfer_manager",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:stream_executor_no_cuda",
#        "@com_google_absl//absl/memory",
#        "@com_google_absl//absl/types:span",
#    ],
#)
#
#cc_library(
#    name = "platform",
#    srcs = ["platform.cc"],
#    hdrs = ["platform.h"],
#    deps = [
#        ":executor",
#        ":platform_id",
#        "//tensorflow/core:stream_executor_headers_lib",
#    ],
#    alwayslink = True,  # Registers itself with the MultiPlatformManager.
#)
#
#cc_library(
#    name = "executor",
#    srcs = ["executor.cc"],
#    hdrs = ["executor.h"],
#    deps = [
#        "//tensorflow/compiler/xla:shape_util",
#        "//tensorflow/compiler/xla:status_macros",
#        "//tensorflow/compiler/xla:xla_data_proto",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:stream_executor_headers_lib",
#        "//tensorflow/stream_executor/host:host_stream",
#        "//tensorflow/stream_executor/host:host_timer",
#        "@com_google_absl//absl/types:span",
#    ],
#)
