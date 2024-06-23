/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <cudnn_frontend.h>
#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "compute/linear_gelu.h"
#include "logger.h"
#include "random.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace {
bool cache_lookup_pre_built_graph(
    std::shared_ptr<cudnn_frontend::graph::Graph> &graph,
    const cudnnHandle_t handle) {
  using cache_t =
      std::unordered_map<std::size_t,
                         std::shared_ptr<cudnn_frontend::graph::Graph>>;
//  static std::mutex m;
//  std::lock_guard<std::mutex> lock(m);
  static cache_t user_maintained_cache;
  auto cache_key = graph->key();
  if (const auto it = user_maintained_cache.find(cache_key);
      it != user_maintained_cache.end()) {
    graph = it->second;
    return true;
  }

  CHECK_CUDNN_FE(graph->build(handle, {cudnn_frontend::HeurMode_t::A}));

  user_maintained_cache.emplace(cache_key, graph);
  return false;
}

cudnn_frontend::DataType_t torchToCudnnDataType(at::ScalarType torchType) {
  static const std::unordered_map<at::ScalarType, cudnn_frontend::DataType_t>
      torchToCudnnDataTypeMap = {
          {at::kHalf, cudnn_frontend::DataType_t::HALF},
          {at::kBFloat16, cudnn_frontend::DataType_t::BFLOAT16},
          {at::kFloat, cudnn_frontend::DataType_t::FLOAT},
};

  auto it = torchToCudnnDataTypeMap.find(torchType);
  CS_ASSERT_TRUE(it != torchToCudnnDataTypeMap.end(),
                 "Unsupported torch data type");
  return it->second;
}

}  // namespace

namespace cs {
cudnnHandle_t getCurrentCuDnnHandle();
}

#define X_UID 1
#define W_UID 2
#define Bias_UID 3
#define Y_UID 4

std::shared_ptr<cudnn_frontend::graph::Graph> create_linear_gelu_forward_graph(
    const cudnn_frontend::DataType_t dataType, const int64_t b, const int64_t m,
    const int64_t n, const int64_t k) {
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

  graph->set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("X")
                             .set_uid(X_UID)
                             .set_data_type(dataType)
                             .set_dim({1, b * m, k})
                             .set_stride({b * m * k, k, 1}));

  auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("W")
                             .set_uid(W_UID)
                             .set_data_type(dataType)
                             .set_dim({1, k, n})
                             .set_stride({k * n, 1, k}));

  auto Bias = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                .set_name("Bias")
                                .set_uid(Bias_UID)
                                .set_data_type(dataType)
                                .set_dim({1, 1, n})
                                .set_stride({n, n, 1}));

  auto matmul_attributes =
      cudnn_frontend::graph::Matmul_attributes()
          .set_name("GEMM");

  auto pw_add_attributes =
      cudnn_frontend::graph::Pointwise_attributes()
          .set_name("pw_Add")
          .set_mode(cudnn_frontend::PointwiseMode_t::ADD);

  auto response = graph->matmul(X, W, matmul_attributes);

  auto Y = graph->pointwise(response, Bias, pw_add_attributes);

  Y->set_output(true).set_name("Y").set_uid(Y_UID).set_data_type(dataType);

  return graph;
}

namespace cs::compute {
OrderedDict<std::string, Tensor> LinearGelu::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  if (args.bias) {
    dict.insert("bias", forward.bias);
  }
  return dict;
}

OrderedDict<std::string, module::State::Increment>
LinearGelu::State::increments() {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight",
              {forward.weight, forward.grad_weight, forward.optimizer_weight});
  if (args.bias) {
    dict.insert("bias",
                {forward.bias, forward.grad_bias, forward.optimizer_bias});
  }
  return dict;
}

std::shared_ptr<LinearGelu::State> LinearGelu::init(const Scheduler &scheduler,
                                                    const Options &options) {
  TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device().value());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype().value());
  }

  if (options.bias()) {
    struct Impl : Task::Impl {
      const TensorOptions options;
      const int64_t in_futures;
      const int64_t out_futures;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const TensorOptions options, const int64_t in_futures,
                    const int64_t out_futures)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            in_futures{in_futures},
            out_futures{out_futures} {}
      void operator()() const override {
        const auto weight_ = torch::empty({out_futures, in_futures}, options);
        output()[0].impl()->tensor() = weight_;
        const auto bias_ = torch::empty({out_futures}, options);
        output()[1].impl()->tensor() = bias_;
        torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(),
                                          std::sqrt(5));
        auto [fan_in, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(
            output()[0].impl()->tensor());
        const auto bound = 1 / std::sqrt(fan_in);
        torch::nn::init::uniform_(output()[1].impl()->tensor(), -bound, bound);
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::LinearGelu::init";
      }
    };

    Tensor weight;
    Tensor bias;
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight, bias},
                                         tensorOptions,
                                         options.in_futures(),
                                         options.out_futures()})});
    return std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.bias()});
  } else {
    struct Impl : Task::Impl {
      const TensorOptions options;
      const int64_t in_futures;
      const int64_t out_futures;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const TensorOptions options, const int64_t in_futures,
                    const int64_t out_futures)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            in_futures{in_futures},
            out_futures{out_futures} {}
      void operator()() const override {
        const auto weight_ = torch::empty({out_futures, in_futures}, options);
        output()[0].impl()->tensor() = weight_;
        torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(),
                                          std::sqrt(5));
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::LinearGelu::init";
      }
    };

    Tensor weight;
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight},
                                         tensorOptions,
                                         options.in_futures(),
                                         options.out_futures()})});
    return std::make_shared<State>(State::Forward{std::move(weight)},
                                   State::Backward{},
                                   State::Args{options.bias()});
  }
}

Tensor LinearGelu::forward(const Scheduler &scheduler,
                           const std::shared_ptr<State> &state,
                           const ReadOnlyTensor &input) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* output */,
                  std::vector<ReadOnlyTensor> input /* input, weight, bias */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      const auto &x = input()[0].impl()->tensor();
      const auto &w = input()[1].impl()->tensor();
      const auto &bias = input()[2].impl()->tensor();
      CS_ASSERT_TRUE(x.size(2) == w.size(1), "mismatch size for matmul.");
      CS_ASSERT_TRUE(bias.size(0) == w.size(0),
                     "mismatch size for adding bias.");
      const auto x_cudnn_type = torchToCudnnDataType(x.scalar_type());
      const auto w_cudnn_type = torchToCudnnDataType(w.scalar_type());
      const auto bias_cudnn_type = torchToCudnnDataType(bias.scalar_type());
      CS_ASSERT_TRUE(
          x_cudnn_type == w_cudnn_type && x_cudnn_type == bias_cudnn_type,
          "we do not support mixed input type.");
      const int64_t b = x.size(0);
      const int64_t m = x.size(1);
      const int64_t k = x.size(2);
      const int64_t n = w.size(0);
      auto graph = create_linear_gelu_forward_graph(x_cudnn_type, b, m, n, k);
      cache_lookup_pre_built_graph(graph, getCurrentCuDnnHandle());

      auto result = at::empty({b, m, n}, x.options());

      std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                         void *>
          variant_pack;
      variant_pack = {{X_UID, x.data_ptr()},
                      {W_UID, w.data_ptr()},
                      {Bias_UID, bias.data_ptr()},
                      {Y_UID, result.data_ptr()}};

      auto workspace = at::empty({graph->get_workspace_size()},
                                 x.options().dtype(at::kByte));

      output()[0].impl()->tensor() = result;

      CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                    workspace.data_ptr()));
      intermediate().emplace_back(std::move(workspace));
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::LinearGelu::forward";
    }
  };

  Tensor output;
  state->backward.input = input;
  // size
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{output}, {input, state->forward.weight, state->forward.bias}})});
  return output;
}

Tensor LinearGelu::backwardInput(const Scheduler &scheduler,
                                 const std::shared_ptr<State> &state,
                                 const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* grad_input */,
                   std::vector<ReadOnlyTensor> input /* grad_output, weight
                   */)
         : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::matmul(input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Linear::backwardInput";
    }
  };

  Tensor grad_input;
  // size
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input}, {grad_output, state->forward.weight}})});
  return grad_input;
}

void LinearGelu::backwardParameter(const Scheduler &scheduler,
                                   const std::shared_ptr<State> &state,
                                   const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* grad_weight */,
                  std::vector<ReadOnlyTensor> input /* grad_output, input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      if (output()[0].impl()->tensor().defined()) {
        const auto reshapedGradOutput = input()[0].impl()->tensor().reshape(
            {-1, input()[0].impl()->tensor().size(-1)});
        const auto transposedGradOutput = reshapedGradOutput.t();
        const auto reshapedInput = input()[1].impl()->tensor().reshape(
            {-1, input()[1].impl()->tensor().size(-1)});
        const auto result = at::matmul(transposedGradOutput, reshapedInput);
        output()[0].impl()->tensor() += result;
        intermediate().resize(4);
        intermediate().push_back(reshapedGradOutput);
        intermediate().push_back(transposedGradOutput);
        intermediate().push_back(reshapedInput);
        intermediate().push_back(result);
      } else {
        const auto reshapedGradOutput = input()[0].impl()->tensor().reshape(
            {-1, input()[0].impl()->tensor().size(-1)});
        const auto transposedGradOutput = reshapedGradOutput.t();
        const auto reshapedInput = input()[1].impl()->tensor().reshape(
            {-1, input()[1].impl()->tensor().size(-1)});
        const auto result = at::matmul(transposedGradOutput, reshapedInput);
        output()[0].impl()->tensor() = result;
        intermediate().resize(3);
        intermediate().push_back(reshapedGradOutput);
        intermediate().push_back(transposedGradOutput);
        intermediate().push_back(reshapedInput);
      }
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Linear::backwardParameter";
    }
  };

  // decrease counter
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {state->forward.grad_weight}, {grad_output, state->backward.input}})});
  state->backward.input.reset();
}
}  // namespace cs::compute
