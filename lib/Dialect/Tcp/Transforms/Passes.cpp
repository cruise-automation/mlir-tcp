#include "Dialect/Tcp/Transforms/Passes.h"
#include "Dialect/Tcp/Transforms/FuseTcpOpsPass.h"
#include "Dialect/Tcp/Transforms/IsolateGroupOpsPass.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace {
#define GEN_PASS_REGISTRATION
#include "Dialect/Tcp/Transforms/Passes.h.inc"
} // end namespace

void mlir::tcp::registerTcpPasses() { ::registerPasses(); }
