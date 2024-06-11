#include "compute/embedding.h"

#include <torch/nn/functional/embedding.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::compute {
OrderedDict<std::string, Tensor> Embedding::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  return dict;
}

OrderedDict<std::string, module::State::Increment>
Embedding::State::increments() {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight",
              {forward.weight, forward.grad_weight, forward.optimizer_weight});
  return dict;
}

void Embedding::init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                     const Options &options) {
  struct Impl : Task::Impl {
    const Options options;

    explicit Impl(std::vector<Tensor> output /* weight */,
                  const Options &options)
        : Task::Impl{std::move(output), {}, compute}, options{options} {}
    void operator()() const override {
      const auto weight_ =
          at::normal(0, 1, {options.num_embeddings(), options.embedding_dim()},
                     {}, options.dtype(), {}, options.device(), {});
      // if (max_norm != c10::nullopt) {
      //   input_ = input_.contiguous();
      //   _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
      // }
      output()[0].impl()->tensor() = weight_;
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Embedding::init";
    }
  };

  int64_t padding_idx = -1;
  if (options.padding_idx() != c10::nullopt) {
    if (*options.padding_idx() > 0) {
      TORCH_CHECK(*options.padding_idx() < options.num_embeddings(),
                  "Padding_idx must be within num_embeddings");
    } else if (*options.padding_idx() < 0) {
      TORCH_CHECK(*options.padding_idx() >= -options.num_embeddings(),
                  "Padding_idx must be within num_embedding");
      padding_idx = options.num_embeddings() + *options.padding_idx();
    }
  }

  TORCH_CHECK(options.max_norm() == c10::nullopt)

  Tensor weight;
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{weight}, options})});

  state = std::make_shared<State>(
      State::Forward{std::move(weight)}, State::Backward{},
      State::Args{options.num_embeddings(), padding_idx, options.max_norm(),
                  options.norm_type(), options.scale_grad_by_freq(),
                  options.sparse()});
}

void Embedding::forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state, Tensor &output,
                        const ReadOnlyTensor &indices) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(std::vector<Tensor> output /* output */,
                  std::vector<ReadOnlyTensor> input /* weight, indices */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), compute},
          args{args} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::embedding(
          input()[0].impl()->tensor(),
          input()[1].impl()->tensor().view(
              {-1, input()[1].impl()->tensor().size(-1)}),
          args.padding_idx, args.scale_grad_by_freq, args.sparse);
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Embedding::forward";
    }
  };

  output = Tensor{};
  state->backward.indices = indices;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{output}, {state->forward.weight, indices}, state->args})});
}

void Embedding::backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(std::vector<Tensor> output /* grad_weight */,
                  std::vector<ReadOnlyTensor> input /* grad_output, indices */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), compute},
          args{args} {}
    void operator()() const override {
      if (output()[0].impl()->tensor().defined()) {
        output()[0].impl()->tensor() += at::embedding_backward(
            input()[0].impl()->tensor(), input()[1].impl()->tensor(),
            args.num_weights, args.padding_idx, args.scale_grad_by_freq,
            args.sparse);
      } else /* accumulate grad */ {
        output()[0].impl()->tensor() = at::embedding_backward(
            input()[0].impl()->tensor(), input()[1].impl()->tensor(),
            args.num_weights, args.padding_idx, args.scale_grad_by_freq,
            args.sparse);
      }
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Embedding::backward";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{state->forward.grad_weight},
                                       {grad_output, state->backward.indices},
                                       state->args})});
  // decrease counter
  state->backward.indices.reset();
}
}  // namespace dllm::compute
