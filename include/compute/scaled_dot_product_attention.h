#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct ScaledDotProductFlashAttention {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> query = nullptr;
      std::shared_ptr<const ReadOnlyTensor> key = nullptr;
      std::shared_ptr<const ReadOnlyTensor> value = nullptr;
      std::shared_ptr<const ReadOnlyTensor> out = nullptr;
      std::shared_ptr<const ReadOnlyTensor> logsumexp = nullptr;
      std::shared_ptr<const ReadOnlyTensor> cum_seq_q = nullptr;
      std::shared_ptr<const ReadOnlyTensor> cum_seq_k = nullptr;
      struct Max {
        int64_t max_q;
        int64_t max_k;
      };
      std::shared_ptr<Max> max = std::make_shared<Max>();
      std::shared_ptr<const ReadOnlyTensor> philox_seed = nullptr;
      std::shared_ptr<const ReadOnlyTensor> philox_offset = nullptr;
    } backward;
    struct Args {
      double dropout_p;
      bool is_causal;
      bool return_debug_mask;
      c10::optional<double> scale;
    } args;
  };

  static TaskCompute init(
      std::shared_ptr<State> &state, double dropout_p = 0,
      bool is_causal = false,
      bool return_debug_mask = false /* This must be false! */,
      c10::optional<double> scale = c10::nullopt);

  static TaskCompute forward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &output,
      const std::shared_ptr<const ReadOnlyTensor> &query,
      const std::shared_ptr<const ReadOnlyTensor> &key,
      const std::shared_ptr<const ReadOnlyTensor> &value);

  static TaskCompute backward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &grad_query,
      const std::shared_ptr<Tensor> &grad_key,
      const std::shared_ptr<Tensor> &grad_value,
      const std::shared_ptr<const ReadOnlyTensor> &grad_out);
};
}  // namespace dllm::compute
