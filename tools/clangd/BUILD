load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",
    # Targets obtained using the query:
    #   bazel query 'kind("(cc.*) rule", //...)'
    targets = {
        "//:Pipeline": "--config=clang_linux",
        "//:StablehloToTcp": "--config=clang_linux",
        "//:TcpConversionPasses": "--config=clang_linux",
        "//:TcpConversionPassesIncGen": "--config=clang_linux",
        "//:TcpDialect": "--config=clang_linux",
        "//:TcpDialectPasses": "--config=clang_linux",
        "//:TcpDialectPassesIncGen": "--config=clang_linux",
        "//:TcpInitAll": "--config=clang_linux",
        "//:TcpOpsIncGen": "--config=clang_linux",
        "//:TcpToArith": "--config=clang_linux",
        "//:TcpToLinalg": "--config=clang_linux",
        "//:TcpTypesIncGen": "--config=clang_linux",
        "//:TorchToTcp": "--config=clang_linux",
        "//:tcp-opt": "--config=clang_linux",
        "//test:AotCompile/test_aot_compiled_basic_tcp_ops": "--config=clang_linux",
        "//test:aot_compiled_basic_tcp_ops": "--config=clang_linux",
        "//tools/aot:abi": "--config=clang_linux",
    },
)
