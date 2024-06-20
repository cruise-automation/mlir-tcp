#pragma once

namespace mlir {

class DialectRegistry;

namespace tcp {

void registerTilingInterfaceExternalModels(DialectRegistry &registry);

} // namespace tcp
} // namespace mlir
