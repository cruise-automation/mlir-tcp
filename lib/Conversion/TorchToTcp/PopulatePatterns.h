//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace torch_to_tcp {

void populateElementwisePatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet);

void populateMiscPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet);

void populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet);

void populateTcpCustomOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet);

namespace cruise {
void populateCruiseInternalPatternsAndLegality(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target);
} // namespace cruise

} // namespace torch_to_tcp
} // namespace mlir
