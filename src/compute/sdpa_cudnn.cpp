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

#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward.h>
#include <ATen/ops/empty.h>
#include <cudnn_frontend.h>

#include "compute/scaled_dot_product_attention.h"
#include "logger.h"
#include "random.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs {
at::Generator &get_cuda_generator();
}

namespace {
bool cache_lookup_pre_built_graph(
    std::shared_ptr<cudnn_frontend::graph::Graph> &graph,
    const cudnnHandle_t handle) {
  using cache_t =
      std::unordered_map<std::size_t,
                         std::shared_ptr<cudnn_frontend::graph::Graph>>;
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

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6
#define SEQ_LEN_Q_UID 7
#define SEQ_LEN_KV_UID 8
#define SEED_UID 9
#define OFFSET_UID 10
#define DO_UID 101
#define DQ_UID 102
#define DK_UID 103
#define DV_UID 104

std::shared_ptr<cudnn_frontend::graph::Graph> create_sdpa_forward_graph(
    const cudnn_frontend::DataType_t dataType, const int64_t b,
    const int64_t h_q, const int64_t h_k, const int64_t h_v, const int64_t s_q,
    const int64_t s_kv, const int64_t d_qk, const int64_t d_v,
    at::IntArrayRef queryStride, at::IntArrayRef keyStride,
    at::IntArrayRef valueStride, double const attn_scale = 1.0f,
    double const dropout_p = 0.0f, bool const is_inference = false,
    bool const causal_mask = false, bool const alibi_mask = false,
    bool const padding_mask = false, bool has_attn_bias = false) {
  // Create a graph and set common global properties.
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
  graph->set_io_data_type(dataType)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  auto Q = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("Q")
                             .set_uid(Q_UID)
                             .set_dim({b, h_q, s_q, d_qk})
                             .set_stride({queryStride[0], queryStride[2],
                                          queryStride[1], queryStride[3]}));
  auto K = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("K")
                             .set_uid(K_UID)
                             .set_dim({b, h_k, s_kv, d_qk})
                             .set_stride({keyStride[0], keyStride[2],
                                          keyStride[1], keyStride[3]}));
  auto V = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("V")
                             .set_uid(V_UID)
                             .set_dim({b, h_v, s_kv, d_v})
                             .set_stride({valueStride[0], valueStride[2],
                                          valueStride[1], valueStride[3]}));

  auto sdpa_options = cudnn_frontend::graph::SDPA_attributes()
                          .set_name("flash_attention")
                          .set_is_inference(is_inference)
                          .set_alibi_mask(alibi_mask)
                          .set_causal_mask(causal_mask)
                          .set_attn_scale(attn_scale);

  if (has_attn_bias) {
    auto bias =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("bias")
                          .set_uid(BIAS_UID)
                          .set_dim({b, 1, s_q, s_kv})
                          .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    sdpa_options.set_bias(bias);
  }

  if (padding_mask) {
    auto seq_q =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("seq_q")
                          .set_uid(SEQ_LEN_Q_UID)
                          .set_dim({b, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT32));
    auto seq_kv =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("seq_kv")
                          .set_uid(SEQ_LEN_KV_UID)
                          .set_dim({b, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT32));
    sdpa_options.set_padding_mask(padding_mask)
        .set_seq_len_q(seq_q)
        .set_seq_len_kv(seq_kv);
  }

  if (dropout_p != 0.0f) {
    auto seed =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("Seed")
                          .set_uid(SEED_UID)
                          .set_dim({1, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT64));
    auto offset =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("Offset")
                          .set_uid(OFFSET_UID)
                          .set_dim({1, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT64));
    seed->set_is_pass_by_value(true);
    offset->set_is_pass_by_value(true);
    sdpa_options.set_dropout(dropout_p, seed, offset);
  }

  auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

  O->set_output(true)
      .set_dim({b, h_q, s_q, d_v})
      .set_stride({s_q * h_q * d_v, d_v, h_q * d_v, 1})
      .set_uid(O_UID);

  if (is_inference) {
    assert(Stats == nullptr);
  } else {
    Stats->set_output(true)
        .set_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_uid(STATS_UID);
  }

  return graph;
}

// Function to create the SDPA (Scaled Dot-Product Attention) backward graph
std::shared_ptr<cudnn_frontend::graph::Graph> create_sdpa_backward_graph(
    const cudnn_frontend::DataType_t dataType, const int64_t b,
    const int64_t h_q, const int64_t h_k, const int64_t h_v, const int64_t s_q,
    const int64_t s_kv, const int64_t d_qk, const int64_t d_v,
    at::IntArrayRef queryStride, at::IntArrayRef keyStride,
    at::IntArrayRef valueStride, at::IntArrayRef outputStride,
    at::IntArrayRef doutputStride, double const attn_scale = 1.0f,
    double const dropout_p = 0.0f,
    [[maybe_unused]] bool const is_inference = false,
    bool const causal_mask = false, bool const alibi_mask = false,
    bool const padding_mask = false, bool has_attn_bias = false) {
  // Create a graph and set common global properties
  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
  graph->set_io_data_type(dataType)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  // Define input tensors Q, K, V
  auto Q = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("Q")
                             .set_uid(Q_UID)
                             .set_dim({b, h_q, s_q, d_qk})
                             .set_stride({queryStride[0], queryStride[2],
                                          queryStride[1], queryStride[3]}));

  auto K = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("K")
                             .set_uid(K_UID)
                             .set_dim({b, h_k, s_kv, d_qk})
                             .set_stride({keyStride[0], keyStride[2],
                                          keyStride[1], keyStride[3]}));

  auto V = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("V")
                             .set_uid(V_UID)
                             .set_dim({b, h_v, s_kv, d_v})
                             .set_stride({valueStride[0], valueStride[2],
                                          valueStride[1], valueStride[3]}));

  // Define output tensor O
  auto O = graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                             .set_name("O")
                             .set_uid(O_UID)
                             .set_dim({b, h_q, s_q, d_v})
                             .set_stride({outputStride[0], outputStride[2],
                                          outputStride[1], outputStride[3]}));

  // Define gradient tensor dO
  auto dO =
      graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name("dO")
                        .set_uid(DO_UID)
                        .set_dim({b, h_q, s_q, d_v})
                        .set_stride({doutputStride[0], doutputStride[2],
                                     doutputStride[1], doutputStride[3]}));

  // Define stats tensor
  auto Stats =
      graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name("Stats")
                        .set_uid(STATS_UID)
                        .set_dim({b, h_q, s_q, 1})
                        .set_stride({h_q * s_q, s_q, 1, 1})
                        .set_data_type(cudnn_frontend::DataType_t::FLOAT));

  // Set SDPA backward options
  auto sdpa_options = cudnn_frontend::graph::SDPA_backward_attributes()
                          .set_name("flash_attention_backward")
                          .set_alibi_mask(alibi_mask)
                          .set_causal_mask(causal_mask)
                          .set_attn_scale(attn_scale);

  // If attention bias is provided, set it
  if (has_attn_bias) {
    auto bias =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("bias")
                          .set_uid(BIAS_UID)
                          .set_dim({b, 1, s_q, s_kv})
                          .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    sdpa_options.set_bias(bias);
  }

  // If padding mask is enabled, set sequence lengths
  if (padding_mask) {
    auto seq_q =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("seq_q")
                          .set_uid(SEQ_LEN_Q_UID)
                          .set_dim({b, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT32));
    auto seq_kv =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("seq_kv")
                          .set_uid(SEQ_LEN_KV_UID)
                          .set_dim({b, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT32));
    sdpa_options.set_padding_mask(padding_mask)
        .set_seq_len_q(seq_q)
        .set_seq_len_kv(seq_kv);
  }

  if (dropout_p != 0.0f) {
    auto seed =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("Seed")
                          .set_uid(SEED_UID)
                          .set_dim({1, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT64));
    auto offset =
        graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                          .set_name("Offset")
                          .set_uid(OFFSET_UID)
                          .set_dim({1, 1, 1, 1})
                          .set_stride({1, 1, 1, 1})
                          .set_data_type(cudnn_frontend::DataType_t::INT64));
    seed->set_is_pass_by_value(true);
    offset->set_is_pass_by_value(true);
    sdpa_options.set_dropout(dropout_p, seed, offset);
  }

  // Compute SDPA backward and get gradients dQ, dK, dV
  auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, sdpa_options);

  // Set output tensors dQ, dK, dV
  dQ->set_output(true)
      .set_uid(DQ_UID)
      .set_dim({b, h_q, s_q, d_qk})
      .set_stride({h_q * s_q * d_qk, d_qk, h_q * d_qk, 1});
  dK->set_output(true)
      .set_uid(DK_UID)
      .set_dim({b, h_k, s_kv, d_qk})
      .set_stride({h_k * s_kv * d_qk, d_qk, h_k * d_qk, 1});
  dV->set_output(true)
      .set_uid(DV_UID)
      .set_dim({b, h_v, s_kv, d_v})
      .set_stride({h_v * s_kv * d_v, d_v, h_v * d_v, 1});

  return graph;
}
}  // namespace

namespace cs {
cudnnHandle_t getCurrentCuDnnHandle();
}

namespace cs::compute {
std::shared_ptr<ScaledDotProductCuDnn::State> ScaledDotProductCuDnn::init(
    const Scheduler &scheduler, const Options &options) {
  return std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.dropout_p(), options.is_causal(),
                  options.return_debug_mask(), options.scale(),
                  std::make_shared<State::Args::RNG>()});
}

Tensor ScaledDotProductCuDnn::forward(const Scheduler &scheduler,
                                      const std::shared_ptr<State> &state,
                                      const ReadOnlyTensor &query,
                                      const ReadOnlyTensor &key,
                                      const ReadOnlyTensor &value) {
  struct Impl : Task::Impl {
    const State::Args args;

    explicit Impl(std::vector<Tensor> output /* output, state */
                  ,
                  std::vector<ReadOnlyTensor> input /* query, key, value */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      const auto &query = input()[0].impl()->tensor();
      const auto &key = input()[1].impl()->tensor();
      const auto &value = input()[2].impl()->tensor();
      CS_ASSERT_TRUE(query.scalar_type() == at::kHalf ||
                         query.scalar_type() == at::kBFloat16,
                     "SDPA only supports FP16 or BF16");
      CS_ASSERT_TRUE(
          key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
          "SDPA only supports FP16 or BF16");
      CS_ASSERT_TRUE(value.scalar_type() == at::kHalf ||
                         value.scalar_type() == at::kBFloat16,
                     "SDPA only supports FP16 or BF16");
      const int64_t b = query.size(0);
      const int64_t h_q = query.size(2);
      const int64_t h_k = key.size(2);
      const int64_t h_v = value.size(2);
      const int64_t s_q = query.size(1);
      const int64_t s_kv = key.size(1);
      const int64_t d_qk = query.size(3);
      const int64_t d_v = value.size(3);
      auto graph = create_sdpa_forward_graph(
          query.scalar_type() == at::kHalf
              ? cudnn_frontend::DataType_t::HALF
              : cudnn_frontend::DataType_t::BFLOAT16,
          b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v, query.strides(),
          key.strides(), value.strides(),
          args.scale.has_value() ? args.scale.value() : 1., args.dropout_p,
          false, args.is_causal);

      cache_lookup_pre_built_graph(graph, getCurrentCuDnnHandle());

      auto result = at::empty({b, s_q, h_q, d_v}, value.options());
      auto stats =
          at::empty({b * h_q * s_q}, query.options().dtype(at::kFloat));

      std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                         void *>
          variant_pack;
      if (args.dropout_p != 0.0f) {
        auto &generator = get_cuda_generator();
        args.rng->seed = generator.current_seed();
        args.rng->offset = generator.get_offset();
        generator.set_offset(args.rng->offset + b * h_q * s_q);
        variant_pack = {
            {Q_UID, query.data_ptr()},      {K_UID, key.data_ptr()},
            {V_UID, value.data_ptr()},      {O_UID, result.data_ptr()},
            {STATS_UID, stats.data_ptr()},  {SEED_UID, &args.rng->seed},
            {OFFSET_UID, &args.rng->offset}};
      } else {
        variant_pack = {{Q_UID, query.data_ptr()},
                        {K_UID, key.data_ptr()},
                        {V_UID, value.data_ptr()},
                        {O_UID, result.data_ptr()},
                        {STATS_UID, stats.data_ptr()}};
      }

      auto workspace = at::empty({graph->get_workspace_size()},
                                 query.options().dtype(at::kByte));

      output()[0].impl()->tensor() = result;
      output()[1].impl()->tensor() = stats;

      CS_CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                       workspace.data_ptr()));
      intermediate().emplace_back(std::move(workspace));
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::ScaledDotProductCuDnn::forward";
    }
  };

  Tensor output;
  const Tensor stats;
  state->backward.query = query;
  state->backward.key = key;
  state->backward.value = value;
  state->backward.out = output;
  state->backward.stats = stats;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{output, stats}, {query, key, value}, state->args})});
  return output;
}

std::array<Tensor, 3> ScaledDotProductCuDnn::backward(
    const Scheduler &scheduler, const std::shared_ptr<State> &state,
    const ReadOnlyTensor &grad_out) {
  struct Impl : Task::Impl {
    const State::Args args;

    explicit Impl(
        std::vector<Tensor> output /* grad_query, grad_key, grad_value */,
        std::vector<ReadOnlyTensor> input /* grad_out[0], query[1], key[2],
                                             value[3], out[4], stats[5] */
        ,
        const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      const auto &grad_out = input()[0].impl()->tensor();
      const auto &query = input()[1].impl()->tensor();
      const auto &key = input()[2].impl()->tensor();
      const auto &value = input()[3].impl()->tensor();
      const auto &out = input()[4].impl()->tensor();
      const auto &stats = input()[5].impl()->tensor();
      CS_ASSERT_TRUE(query.scalar_type() == at::kHalf ||
                         query.scalar_type() == at::kBFloat16,
                     "SDPA only supports FP16 or BF16");
      CS_ASSERT_TRUE(
          key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
          "SDPA only supports FP16 or BF16");
      CS_ASSERT_TRUE(value.scalar_type() == at::kHalf ||
                         value.scalar_type() == at::kBFloat16,
                     "SDPA only supports FP16 or BF16");
      const int64_t b = query.size(0);
      const int64_t h_q = query.size(2);
      const int64_t h_k = key.size(2);
      const int64_t h_v = value.size(2);
      const int64_t s_q = query.size(1);
      const int64_t s_kv = key.size(1);
      const int64_t d_qk = query.size(3);
      const int64_t d_v = value.size(3);
      auto graph = create_sdpa_backward_graph(
          query.scalar_type() == at::kHalf
              ? cudnn_frontend::DataType_t::HALF
              : cudnn_frontend::DataType_t::BFLOAT16,
          b, h_q, h_k, h_v, s_q, s_kv, d_qk, d_v, query.strides(),
          key.strides(), value.strides(), out.strides(), grad_out.strides(),
          args.scale.has_value() ? args.scale.value() : 1., args.dropout_p,
          false, args.is_causal);

      cache_lookup_pre_built_graph(graph, getCurrentCuDnnHandle());

      auto grad_query = at::empty(query.sizes(), query.options());
      auto grad_key = at::empty(key.sizes(), key.options());
      auto grad_value = at::empty(value.sizes(), value.options());

      std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t,
                         void *>
          variant_pack;

      if (args.dropout_p != 0.0f) {
        variant_pack = {// inputs
                        {Q_UID, query.data_ptr()},
                        {K_UID, key.data_ptr()},
                        {V_UID, value.data_ptr()},
                        {O_UID, out.data_ptr()},
                        {DO_UID, grad_out.data_ptr()},
                        {STATS_UID, stats.data_ptr()},
                        // outputs
                        {DQ_UID, grad_query.data_ptr()},
                        {DK_UID, grad_key.data_ptr()},
                        {DV_UID, grad_value.data_ptr()},
                        {SEED_UID, &args.rng->seed},
                        {OFFSET_UID, &args.rng->offset}};
      } else {
        variant_pack = {// inputs
                        {Q_UID, query.data_ptr()},
                        {K_UID, key.data_ptr()},
                        {V_UID, value.data_ptr()},
                        {O_UID, out.data_ptr()},
                        {DO_UID, grad_out.data_ptr()},
                        {STATS_UID, stats.data_ptr()},
                        // outputs
                        {DQ_UID, grad_query.data_ptr()},
                        {DK_UID, grad_key.data_ptr()},
                        {DV_UID, grad_value.data_ptr()}};
      }

      auto workspace = at::empty({graph->get_workspace_size()},
                                 query.options().dtype(at::kByte));
      output()[0].impl()->tensor() = grad_query;
      output()[1].impl()->tensor() = grad_key;
      output()[2].impl()->tensor() = grad_value;

      CS_CHECK_CUDNN_FE(graph->execute(getCurrentCuDnnHandle(), variant_pack,
                                       workspace.data_ptr()));
      intermediate().emplace_back(std::move(workspace));
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::ScaledDotProductCuDnn::backward";
    }
  };

  Tensor grad_query;
  Tensor grad_key;
  Tensor grad_value;

  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_query, grad_key, grad_value},
           {grad_out, state->backward.query, state->backward.key,
            state->backward.value, state->backward.out, state->backward.stats},
           state->args})});

  // decrease counter
  state->backward.query.reset();
  state->backward.key.reset();
  state->backward.value.reset();
  state->backward.out.reset();
  state->backward.stats.reset();
  return {grad_query, grad_key, grad_value};
}
}  // namespace cs::compute
