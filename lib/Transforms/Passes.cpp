#include "Transforms/Passes.h"
#include "Transforms/FuseTcpOpsPass.h"
#include "Transforms/IsolateGroupOpsPass.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace {
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"
} // end namespace

void mlir::tcp::registerTcpPasses() { ::registerPasses(); }
