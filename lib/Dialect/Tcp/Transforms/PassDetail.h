#pragma once

#include "mlir/Pass/Pass.h"

#include "Dialect/Tcp/IR/TcpOps.h"

namespace mlir::tcp {

using namespace mlir;
#define GEN_PASS_CLASSES
#include "Dialect/Tcp/Transforms/Passes.h.inc"

} // end namespace mlir::tcp
