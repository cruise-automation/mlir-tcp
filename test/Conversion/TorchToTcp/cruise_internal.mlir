// RUN: tcp-opt %s -convert-torch-to-tcp-cruise-internal -split-input-file | FileCheck %s

// Since function arguments are not yet converted, casts will be added which will be
// resolved in the next pass
// CHECK:  @index_from_tensor_tensor_from_index(
// CHECK:   %[[ARG0:.+]]: !torch.vtensor<[1],si64>
// CHECK:        %[[CAST1:.+]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1],si64> -> tensor<1xi64>
// CHECK:        %[[INDEX:.+]] = tcp.caspr_index_from_tensor(%[[CAST1]]) : (tensor<1xi64>) -> index
// CHECK:        %[[RES:.+]] = tcp.caspr_create_tensor_from_index(%[[INDEX]]) : (index) -> tensor<1xi64>
// CHECK:        %[[CAST2:.+]] = torch_c.from_builtin_tensor %[[RES]] : tensor<1xi64> ->  !torch.vtensor<[1],si64>
// CHECK:        return %[[CAST2]] : !torch.vtensor<[1],si64>
func.func @index_from_tensor_tensor_from_index(%arg0: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> {
  %0 = torch.caspr_index_from_tensor(%arg0) : (!torch.vtensor<[1],si64>) -> !torch.index
  %1 = torch.caspr_create_tensor_from_index(%0) : (!torch.index) -> !torch.vtensor<[1],si64>
  return %1 : !torch.vtensor<[1],si64>
}