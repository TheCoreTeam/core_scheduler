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
      ReadOnlyTensor query{};
      ReadOnlyTensor key{};
      ReadOnlyTensor value{};
      ReadOnlyTensor out{};
      ReadOnlyTensor logsumexp{};
      ReadOnlyTensor cum_seq_q{};
      ReadOnlyTensor cum_seq_k{};
      struct Max {
        int64_t max_q;
        int64_t max_k;
      };
      std::shared_ptr<Max> max = std::make_shared<Max>();
      ReadOnlyTensor philox_seed{};
      ReadOnlyTensor philox_offset{};
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

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options = {});

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &query, const ReadOnlyTensor &key,
                        const ReadOnlyTensor &value);

  // grad_query, grad_key, grad_value
  static std::array<Tensor, 3> backward(const Scheduler &scheduler,
                                        const std::shared_ptr<State> &state,
                                        const ReadOnlyTensor &grad_out);
};
}  // namespace dllm::compute
