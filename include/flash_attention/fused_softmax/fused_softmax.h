#include "tensor.h"

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_masked_softmax {
void fwd_cuda(dllm::Tensor<4>& softmax_results, dllm::Tensor<4> const& input,
              dllm::Tensor<4> const& mask, float scale_factor);

void bwd_cuda(dllm::Tensor<4>& input_grads, dllm::Tensor<4>& output_grads,
              dllm::Tensor<4> const& softmax_results, float scale_factor);
}  // namespace scaled_masked_softmax
}  // namespace fused_softmax
}  // namespace multihead_attn

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_upper_triang_masked_softmax {

void fwd_cuda(dllm::Tensor<3>& softmax_results, dllm::Tensor<3> const& input,
              float scale_factor);

void bwd_cuda(dllm::Tensor<3>& output_grads,
              dllm::Tensor<3> const& softmax_results, float scale_factor);
}  // namespace scaled_upper_triang_masked_softmax
}  // namespace fused_softmax
}  // namespace multihead_attn
