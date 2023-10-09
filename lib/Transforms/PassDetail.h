#pragma once

#include "mlir/Pass/Pass.h"

#include "IR/TcpOps.h"

namespace mlir::tcp {

using namespace mlir;
#define GEN_PASS_CLASSES
#include "Transforms/Passes.h.inc"

} // end namespace mlir::tcp
