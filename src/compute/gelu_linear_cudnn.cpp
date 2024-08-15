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
#include <torch/nn/init.h>

#include "compute/gelu_linear.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace {
bool cache_lookup_pre_built_forward_graph(
    std::shared_ptr<cudnn_frontend::graph::Graph> &graph,
    const cudnnHandle_t handle) {
  using cache_t =
      std::unordered_map<std::size_t,
                         std::shared_ptr<cudnn_frontend::graph::Graph>>;
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  static cache_t user_maintained_cache;
  auto cache_key = graph->key();
  if (const auto it = user_maintained_cache.find(cache_key);
      it != user_maintained_cache.end()) {
    graph = it->second;
    return true;
  }

  CS_CHECK_CUDNN_FE(graph->build(handle, {cudnn_frontend::HeurMode_t::A}));

  user_maintained_cache.emplace(cache_key, graph);
  return false;
}

bool cache_lookup_pre_built_backward_input_graph(
    std::shared_ptr<cudnn_frontend::graph::Graph> &graph,
    const cudnnHandle_t handle) {
  using cache_t =
      std::unordered_map<std::size_t,
                         std::shared_ptr<cudnn_frontend::graph::Graph>>;
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  static cache_t user_maintained_cache;
  auto cache_key = graph->key();
  if (const auto it = user_maintained_cache.find(cache_key);
      it != user_maintained_cache.end()) {
    graph = it->second;
    return true;
  }

  CS_CHECK_CUDNN_FE(graph->build(handle, {cudnn_frontend::HeurMode_t::A}));

  user_maintained_cache.emplace(cache_key, graph);
  return false;
}

bool cache_lookup_pre_built_backward_weight_graph(
    std::shared_ptr<cudnn_frontend::graph::Graph> &graph,
    const cudnnHandle_t handle) {
  using cache_t =
      std::unordered_map<std::size_t,
                         std::shared_ptr<cudnn_frontend::graph::Graph>>;
  static std::mutex m;
  std::lock_guard<std::mutex> lock(m);
  static cache_t user_maintained_cache;
  auto cache_key = graph->key();
  if (const auto it = user_maintained_cache.find(cache_key);
      it != user_maintained_cache.end()) {
    graph = it->second;
    return true;
  }

  CS_CHECK_CUDNN_FE(graph->build(handle, {cudnn_frontend::HeurMode_t::A}));

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
#define DX_UID 5
#define DW_UID 6
#define DY_UID 7

std::shared_ptr<cudnn_frontend::graph::Graph> create_gelu_linear_forward_graph(
    const cudnn_frontend::DataType_t input_type, const int64_t b,
    const int64_t n, const int64_t k, const bool with_bias) {
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

  graph->set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("X")
                             .set_uid(X_UID)
                             .set_data_type(input_type)
                             .set_dim({1, b, k})
                             .set_stride({b * k, k, 1}));

  auto pw_gelu_attributes =
      cudnn_frontend::graph::Pointwise_attributes()
          .set_name("pw_GeLU")
          .set_mode(cudnn_frontend::PointwiseMode_t::GELU_FWD);

  auto gelu_X = graph->pointwise(X, pw_gelu_attributes);
  gelu_X->set_data_type(input_type);

  auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("W")
                             .set_uid(W_UID)
                             .set_data_type(input_type)
                             .set_dim({1, k, n})
                             .set_stride({k * n, 1, k}));

  auto matmul_attributes =
      cudnn_frontend::graph::Matmul_attributes().set_name("GEMM");

  auto Y = graph->matmul(gelu_X, W, matmul_attributes);

  if (with_bias) {
    auto Bias = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                  .set_name("Bias")
                                  .set_uid(Bias_UID)
                                  .set_data_type(input_type)
                                  .set_dim({1, 1, n})
                                  .set_stride({n, n, 1}));

    auto pw_add_attributes =
        cudnn_frontend::graph::Pointwise_attributes()
            .set_name("pw_Add")
            .set_mode(cudnn_frontend::PointwiseMode_t::ADD);

    Y = graph->pointwise(Y, Bias, pw_add_attributes);
  }

  Y->set_output(true).set_name("Y").set_uid(Y_UID).set_data_type(input_type);

  return graph;
}

std::shared_ptr<cudnn_frontend::graph::Graph>
create_gelu_linear_backward_input_graph(
    const cudnn_frontend::DataType_t input_type, const int64_t b,
    const int64_t n, const int64_t k) {
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

  graph->set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  auto DY = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_uid(DY_UID)
                              .set_data_type(input_type)
                              .set_dim({1, b, n})
                              .set_stride({b * n, n, 1}));

  auto W = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("W")
                             .set_uid(W_UID)
                             .set_data_type(input_type)
                             .set_dim({1, n, k})
                             .set_stride({n * k, k, 1}));

  auto matmul_attributes =
      cudnn_frontend::graph::Matmul_attributes().set_name("GEMM");

  auto dGelu_x = graph->matmul(DY, W, matmul_attributes);

  auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("X")
                             .set_uid(X_UID)
                             .set_data_type(input_type)
                             .set_dim({1, b, k})
                             .set_stride({b * k, k, 1}));

  auto dgelu_attributes =
      cudnn_frontend::graph::Pointwise_attributes().set_name("GeLU").set_mode(
          cudnn_frontend::PointwiseMode_t::GELU_BWD);

  auto dX = graph->pointwise(dGelu_x, X, dgelu_attributes);
  dX->set_output(true).set_name("dX").set_uid(DX_UID).set_data_type(input_type);

  return graph;
}

std::shared_ptr<cudnn_frontend::graph::Graph>
create_gelu_linear_backward_weight_graph(
    const cudnn_frontend::DataType_t input_type, const int64_t b,
    const int64_t n, const int64_t k) {
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

  graph->set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("X")
                             .set_uid(X_UID)
                             .set_data_type(input_type)
                             .set_dim({1, b, k})
                             .set_stride({b * k, k, 1}));

  auto gelu_attributes =
      cudnn_frontend::graph::Pointwise_attributes().set_name("GeLU").set_mode(
          cudnn_frontend::PointwiseMode_t::GELU_FWD);

  X = graph->pointwise(X, gelu_attributes);
  X->set_data_type(input_type);

  auto DY = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_uid(DY_UID)
                              .set_data_type(input_type)
                              .set_dim({1, n, b})
                              .set_stride({n * b, 1, n}));

  auto matmul_attributes =
      cudnn_frontend::graph::Matmul_attributes().set_name("GEMM");

  auto DW = graph->matmul(DY, X, matmul_attributes);

  DW->set_output(true).set_name("DW").set_uid(DW_UID).set_data_type(input_type);

  return graph;
}

std::shared_ptr<cudnn_frontend::graph::Graph>
create_gelu_linear_backward_weight_accumulate_graph(
    const cudnn_frontend::DataType_t input_type, const int64_t b,
    const int64_t n, const int64_t k) {
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();

  graph->set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
  auto X = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("X")
                             .set_uid(X_UID)
                             .set_data_type(input_type)
                             .set_dim({1, b, k})
                             .set_stride({b * k, k, 1}));

  auto gelu_attributes =
      cudnn_frontend::graph::Pointwise_attributes().set_name("GeLU").set_mode(
          cudnn_frontend::PointwiseMode_t::GELU_FWD);

  X = graph->pointwise(X, gelu_attributes);
  X->set_data_type(input_type);

  auto DY = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                              .set_name("DY")
                              .set_uid(DY_UID)
                              .set_data_type(input_type)
                              .set_dim({1, n, b})
                              .set_stride({n * b, 1, n}));

  auto matmul_attributes =
      cudnn_frontend::graph::Matmul_attributes().set_name("GEMM");

  auto DW = graph->matmul(DY, X, matmul_attributes);

  auto DW_DST = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                  .set_name("DW")
                                  .set_uid(DW_UID)
                                  .set_data_type(input_type)
                                  .set_dim({1, n, k})
                                  .set_stride({n * k, k, 1}));

  auto add_attributes =
      cudnn_frontend::graph::Pointwise_attributes().set_name("ADD").set_mode(
          cudnn_frontend::PointwiseMode_t::ADD);
  DW_DST = graph->pointwise(DW, DW_DST, add_attributes);

  DW_DST->set_output(true).set_name("DW").set_uid(DW_UID).set_data_type(
      input_type);

  return graph;
}

namespace cs::compute {
GeluLinear::State::State(const Forward &forward, const Backward &backward,
                         const Args &args)
    : forward{forward}, backward{backward}, args{args} {}

OrderedDict<std::string, Tensor> GeluLinear::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  if (args.bias) {
    dict.insert("bias", forward.bias);
  }
  return dict;
}

OrderedDict<std::string, Tensor> GeluLinear::State::gradients() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.grad_weight);
  if (args.bias) {
    dict.insert("bias", forward.grad_bias);
  }
  return dict;
}

OrderedDict<std::string, module::State::Increment>
GeluLinear::State::increments() const {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight", {forward.weight, forward.grad_weight});
  if (args.bias) {
    dict.insert("bias", {forward.bias, forward.grad_bias});
  }
  return dict;
}

void GeluLinear::State::zero_grad() {
  forward.grad_weight = {};
  if (args.bias) {
    forward.grad_bias = {};
  }
}

std::shared_ptr<GeluLinear::State> GeluLinear::init(const Scheduler &scheduler,
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
          : Task::Impl{std::move(output), {}, kMain, kCompute},
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
        return "cs::compute::GeluLinear::init";
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
          : Task::Impl{std::move(output), {}, kMain, kCompute},
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
        return "cs::compute::GeluLinear::init";
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

Tensor GeluLinear::forward(const Scheduler &scheduler,
                           const std::shared_ptr<State> &state,
                           const ReadOnlyTensor &input) {
  struct Impl : Task::Impl {
    State::Args args;
    explicit Impl(std::vector<Tensor> output /* output */,
                  std::vector<ReadOnlyTensor> input /* input, weight, bias */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      const auto &x = input()[0].impl()->tensor();
      const auto &w = input()[1].impl()->tensor();
      const auto &bias = input()[2].impl()->tensor();
      CS_ASSERT_TRUE(x.size(-1) == w.size(1), "mismatch size for matmul.");
      const auto x_cudnn_type = torchToCudnnDataType(x.scalar_type());
      const auto w_cudnn_type = torchToCudnnDataType(w.scalar_type());
      CS_ASSERT_TRUE(x_cudnn_type == w_cudnn_type,
                     "we do not support mixed input type.");
      CS_ASSERT_TRUE(x_cudnn_type != cudnn_frontend::DataType_t::FLOAT,
                     "we do not support float for now.");
      auto xView = x.view({-1, x.size(-1)});
      const int64_t b = xView.size(0);
      const int64_t k = xView.size(1);
      const int64_t n = w.size(0);
      auto graph =
          create_gelu_linear_forward_graph(x_cudnn_type, b, n, k, args.bias);
      cache_lookup_pre_built_forward_graph(graph, getCurrentCuDnnHandle());

      IntArray size{x.sizes().begin(), x.sizes().end()};
      size[size.size() - 1] = n;
      auto result = at::empty(size, x.options());

      std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                         void *>
          variant_pack = {{X_UID, x.data_ptr()},
                          {W_UID, w.data_ptr()},
                          {Bias_UID, args.bias ? bias.data_ptr() : nullptr},
                          {Y_UID, result.data_ptr()}};

      auto workspace = at::empty({graph->get_workspace_size()},
                                 x.options().dtype(at::kByte));

      output()[0].impl()->tensor() = result;

      CS_CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                       workspace.data_ptr()));
      intermediate().emplace_back(std::move(workspace));
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::GeluLinear::forward";
    }
  };

  Tensor output;
  state->backward.input = input;
  // size
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{output},
           {input, state->forward.weight, state->forward.bias},
           state->args})});
  return output;
}

Tensor GeluLinear::backward_input(const Scheduler &scheduler,
                                  const std::shared_ptr<State> &state,
                                  const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    explicit Impl(
        std::vector<Tensor> output /* grad_input */,
        std::vector<ReadOnlyTensor> input /* grad_output, weight, input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      auto grad_output = input()[0].impl()->tensor();
      const auto &w = input()[1].impl()->tensor();
      auto x = input()[2].impl()->tensor();
      auto xView = x.view({-1, x.size(-1)});
      const int64_t b = xView.size(0);
      const int64_t k = xView.size(1);
      const int64_t n = w.size(0);
      const auto x_cudnn_type = torchToCudnnDataType(x.scalar_type());
      const auto w_cudnn_type = torchToCudnnDataType(w.scalar_type());
      CS_ASSERT_TRUE(x_cudnn_type == w_cudnn_type,
                     "we do not support mixed input type.");
      CS_ASSERT_TRUE(x_cudnn_type != cudnn_frontend::DataType_t::FLOAT,
                     "we do not support float for now.");
      auto graph =
          create_gelu_linear_backward_input_graph(x_cudnn_type, b, n, k);
      cache_lookup_pre_built_backward_input_graph(graph,
                                                  getCurrentCuDnnHandle());

      auto result = at::empty_like(x);

      std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                         void *>
          variant_pack = {{DY_UID, grad_output.data_ptr()},
                          {X_UID, x.data_ptr()},
                          {W_UID, w.data_ptr()},
                          {DX_UID, result.data_ptr()}};

      auto workspace = at::empty({graph->get_workspace_size()},
                                 x.options().dtype(at::kByte));

      output()[0].impl()->tensor() = result;

      CS_CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                       workspace.data_ptr()));
      intermediate().emplace_back(std::move(workspace));
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::GeluLinear::backward_input";
    }
  };

  Tensor grad_input;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input},
           {grad_output, state->forward.weight, state->backward.input}})});
  ++state->backward.input_count;
  if (state->backward.input_count == 2) {
    state->backward.input.reset();
    state->backward.input_count = 0;
  }
  return grad_input;
}

void GeluLinear::backward_parameter(const Scheduler &scheduler,
                                    const std::shared_ptr<State> &state,
                                    const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* grad_weight */,
                  std::vector<ReadOnlyTensor> input /* grad_output, input */)
        : Task::Impl{std::move(output), std::move(input), kAssist, kCompute} {}
    void operator()() const override {
      const auto &grad_output = input()[0].impl()->tensor();
      const auto &x = input()[1].impl()->tensor();
      auto xView = x.view({-1, x.size(-1)});
      const int64_t b = xView.size(0);
      const int64_t k = xView.size(1);
      const int64_t n = grad_output.size(-1);
      const auto x_cudnn_type = torchToCudnnDataType(x.scalar_type());

      if (output()[0].impl()->tensor().defined()) {
        auto graph = create_gelu_linear_backward_weight_accumulate_graph(
            x_cudnn_type, b, n, k);
        cache_lookup_pre_built_backward_weight_graph(graph,
                                                     getCurrentCuDnnHandle());

        auto result = at::empty({n, k}, x.options());

        std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                           void *>
            variant_pack = {{DY_UID, grad_output.data_ptr()},
                            {X_UID, x.data_ptr()},
                            {DW_UID, result.data_ptr()}};

        auto workspace = at::empty({graph->get_workspace_size()},
                                   x.options().dtype(at::kByte));

        output()[0].impl()->tensor() = result;

        CS_CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                         workspace.data_ptr()));
        intermediate().emplace_back(std::move(workspace));
      } else {
        auto graph =
            create_gelu_linear_backward_weight_graph(x_cudnn_type, b, n, k);
        cache_lookup_pre_built_backward_weight_graph(graph,
                                                     getCurrentCuDnnHandle());

        auto result = at::empty({n, k}, x.options());

        std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                           void *>
            variant_pack = {{DY_UID, grad_output.data_ptr()},
                            {X_UID, x.data_ptr()},
                            {DW_UID, result.data_ptr()}};

        auto workspace = at::empty({graph->get_workspace_size()},
                                   x.options().dtype(at::kByte));

        output()[0].impl()->tensor() = result;

        CS_CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                         workspace.data_ptr()));
        intermediate().emplace_back(std::move(workspace));
      }
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::GeluLinear::backward_weight";
    }
  };

  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {state->forward.grad_weight}, {grad_output, state->backward.input}})});
  if (state->args.bias) {
    struct Impl : Task::Impl {
      explicit Impl(std::vector<Tensor> output /* grad_bias */,
                    std::vector<ReadOnlyTensor> input /* grad_output */)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute} {
      }
      void operator()() const override {
        const auto &grad_output = input()[0].impl()->tensor();

        if (output()[0].impl()->tensor().defined()) {
          auto grad = grad_output.view({-1, grad_output.size(-1)}).sum({0});
          output()[0].impl()->tensor() += grad;
          intermediate().push_back(grad);
        } else {
          output()[0].impl()->tensor() =
              grad_output.view({-1, grad_output.size(-1)}).sum({0});
        }
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::GeluLinear::backward_bias";
      }
    };

    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{{state->forward.grad_bias}, {grad_output}})});
  }
  ++state->backward.input_count;
  if (state->backward.input_count == 2) {
    state->backward.input.reset();
    state->backward.input_count = 0;
  }
}
}  // namespace cs::compute
