load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Keep this list updated by running the following query:
    #   bazel query 'kind("(cc.*) rule", //...)'
    targets = [
        "//:Pipeline",
        "//:StablehloToTcp",
        "//:TcpConversionPasses",
        "//:TcpConversionPassesIncGen",
        "//:TcpDialect",
        "//:TcpDialectPasses",
        "//:TcpDialectPassesIncGen",
        "//:TcpInitAll",
        "//:TcpOpsIncGen",
        "//:TcpToArith",
        "//:TcpToLinalg",
        "//:TcpTypesIncGen",
        "//:TorchToTcp",
        "//:tcp-opt",
        "//test:AotCompile/test_aot_compiled_basic_tcp_ops",
        "//test:aot_compiled_basic_tcp_ops",
    ],
)
