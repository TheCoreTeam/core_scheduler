#pragma once
#include "arg.h"
#include "tensor.h"
#include "threading/scheduler.h"

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
      const double dropout_p = 0;
      const bool is_causal = false;
      const bool return_debug_mask = false /* This must be false! */;
      const c10::optional<double> scale = c10::nullopt;
    } args;
  };

  struct Options {
    Options() {}
    DLLM_ARG(double, dropout_p) = 0;
    DLLM_ARG(bool, is_causal) = false;
    DLLM_ARG(bool, return_debug_mask) = false;
    DLLM_ARG(c10::optional<double>, scale) = {};
  };

  static void init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                   const Options &options = {});

  static void forward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state,
                      const std::shared_ptr<Tensor> &output,
                      const std::shared_ptr<const ReadOnlyTensor> &query,
                      const std::shared_ptr<const ReadOnlyTensor> &key,
                      const std::shared_ptr<const ReadOnlyTensor> &value);

  static void backward(const Scheduler &scheduler,
                       const std::shared_ptr<State> &state,
                       const std::shared_ptr<Tensor> &grad_query,
                       const std::shared_ptr<Tensor> &grad_key,
                       const std::shared_ptr<Tensor> &grad_value,
                       const std::shared_ptr<const ReadOnlyTensor> &grad_out);
};
}  // namespace dllm::compute
