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

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "compute/add.h"
#include "compute/cross_entropy.h"
#include "compute/embedding.h"
#include "compute/gelu.h"
#include "compute/layer_norm.h"
#include "compute/linear.h"
#include "compute/scaled_dot_product_attention.h"
#include "compute/utils.h"
#include "data/dataloader.h"
#include "data/dataset.h"
#include "logger.h"
#include "memory/to_torch.h"
#include "module/embedding.h"
#include "module/layer_norm.h"
#include "module/linear.h"
#include "module/module.h"
#include "optimizer/adamw.h"
#include "tensor.h"
#include "threading/dynamic_scheduler.h"

struct ModelConfig {
  const int64_t batch_size = 2;
  const int64_t block_size = 1024;
  const int64_t vocab_size = 50257;
  const int64_t pad_size = 50304;  // pad vocab_size to be more efficient
  const int64_t n_embd = 2048;     // 2048
  const int64_t n_head = 32;       // 32
  const int64_t n_layer = 22;
  const bool use_bias = false;
  const float dropout = 0.0;
  const float epsilon = 1e-5;
  torch::Device device = torch::kCUDA;
  torch::Dtype dtype = torch::kBFloat16;
};

struct TrainConfig {
  const int64_t epoch = 5;
  const int64_t wait_every_step = 1;
  double lr = 1e-5;
  double beta1 = 0.9;
  double beta2 = 0.95;
  double weight_decay = 1e-1;
};

struct DataConfig {
  const std::string path = "/home/ly/main/dataset/example_llm_data/train.arrow";
};

// Function to display the progress bar
void display_progress(int completed, int total, const std::string &prefix = "",
                      int rank = 0, bool only_rank0 = true) {
  if (rank == 0 && only_rank0) {  // Check if this is rank 0
    int width = 50;               // Width of the progress bar
    float progress = total > 1 ? (float)completed / (total - 1)
                               : 1.0;  // Adjusted to avoid division by zero
    int pos = width * progress;

    if (completed == 0) {
      std::cout << "";
    }
    std::cout << prefix;
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
      if (i < pos)
        std::cout << "=";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << completed + 1 << "/" << total << " | "
              << int(progress * 100.0) << " %\r";
    std::cout.flush();
    if (completed == total - 1) {
      std::cout << "" << std::endl;
    }
  }
}

struct Attn : cs::module::Module {
  cs::DynamicScheduler scheduler;
  const ModelConfig &config;
  cs::module::Linear c_attn{nullptr}, c_proj{nullptr};
  std::shared_ptr<cs::compute::ScaledDotProductFlashAttention::State>
      attn_state;

  Attn(cs::DynamicScheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    c_attn = register_module(
        "c_attn", cs::module::Linear(
                      scheduler, cs::compute::Linear::Options{config.n_embd,
                                                              3 * config.n_embd}
                                     .bias(config.use_bias)
                                     .device(config.device)
                                     .dtype(config.dtype)));
    // we don't need to register flash attention to module because it does not
    // have parameters.
    cs::compute::ScaledDotProductFlashAttention::init(
        scheduler, attn_state,
        cs::compute::ScaledDotProductFlashAttention::Options{}.is_causal(true));
    c_proj = register_module(
        "c_proj", cs::module::Linear(
                      scheduler,
                      cs::compute::Linear::Options{config.n_embd, config.n_embd}
                          .bias(config.use_bias)
                          .device(config.device)
                          .dtype(config.dtype)));
  }

  void forward(cs::DynamicScheduler scheduler, cs::Tensor &Output,
               cs::Tensor &Input) {
    // Fc Attn
    cs::Tensor FcAttnOut;
    c_attn->forward(scheduler, FcAttnOut, Input);

    // Attn
    cs::Tensor AttnOutView;
    {
      std::vector<cs::Tensor> qkvSplit;
      cs::compute::Utils::split(scheduler, qkvSplit, FcAttnOut, config.n_embd,
                                -1);

      auto &q = qkvSplit[0];
      auto &k = qkvSplit[1];
      auto &v = qkvSplit[2];

      cs::Tensor qview;
      cs::compute::Utils::view(scheduler, qview, q,
                               {config.batch_size, config.block_size,
                                config.n_head, config.n_embd / config.n_head});
      cs::Tensor kview;
      cs::compute::Utils::view(scheduler, kview, k,
                               {config.batch_size, config.block_size,
                                config.n_head, config.n_embd / config.n_head});
      cs::Tensor vview;
      cs::compute::Utils::view(scheduler, vview, v,
                               {config.batch_size, config.block_size,
                                config.n_head, config.n_embd / config.n_head});

      cs::Tensor AttnOut;
      cs::compute::ScaledDotProductFlashAttention::forward(
          scheduler, attn_state, AttnOut, qview, kview, vview);

      cs::compute::Utils::view(
          scheduler, AttnOutView, AttnOut,
          {config.batch_size, config.block_size, config.n_embd});
    }

    // Fc Proj
    c_proj->forward(scheduler, Output, AttnOutView);
  }

  void backward(cs::DynamicScheduler scheduler, cs::Tensor &DInput,
                cs::Tensor &DOutput) {
    // Fc Proj, we first run backward for input and then backward for parameter.
    cs::Tensor DAttnOut;
    c_proj->backwardInput(scheduler, DAttnOut, DOutput);
    c_proj->backwardParameter(scheduler, DOutput);

    // Attn
    cs::Tensor DFcAttnOut;
    {
      cs::compute::Utils::view(scheduler, DAttnOut, DAttnOut,
                               {config.batch_size, config.block_size,
                                config.n_head, config.n_embd / config.n_head});

      cs::Tensor dq, dk, dv;
      cs::compute::ScaledDotProductFlashAttention::backward(
          scheduler, attn_state, dq, dk, dv, DAttnOut);

      cs::compute::Utils::view(
          scheduler, dq, dq,
          {config.batch_size, config.block_size, config.n_embd});
      cs::compute::Utils::view(
          scheduler, dk, dk,
          {config.batch_size, config.block_size, config.n_embd});
      cs::compute::Utils::view(
          scheduler, dv, dv,
          {config.batch_size, config.block_size, config.n_embd});
      cs::compute::Utils::cat(scheduler, DFcAttnOut, {dq, dk, dv}, -1);
    }

    // Fc Attn, we first run backward for input and then backward for parameter.
    c_attn->backwardInput(scheduler, DInput, DFcAttnOut);
    c_attn->backwardParameter(scheduler, DFcAttnOut);
  }
};

struct MLP : cs::module::Module {
  cs::DynamicScheduler scheduler;
  const ModelConfig &config;
  cs::module::Linear fc1{nullptr}, fc2{nullptr};
  std::shared_ptr<cs::compute::GeLU::State> gelu_state;

  MLP(cs::DynamicScheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    fc1 = register_module(
        "fc1", cs::module::Linear(
                   scheduler, cs::compute::Linear::Options{config.n_embd,
                                                           4 * config.n_embd}
                                  .bias(config.use_bias)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    cs::compute::GeLU::init(scheduler, gelu_state);
    fc2 = register_module(
        "fc2", cs::module::Linear(
                   scheduler, cs::compute::Linear::Options{4 * config.n_embd,
                                                           config.n_embd}
                                  .bias(config.use_bias)
                                  .device(config.device)
                                  .dtype(config.dtype)));
  }

  void forward(cs::DynamicScheduler scheduler, cs::Tensor &Output,
               cs::Tensor &Input) {
    cs::Tensor Fc1Out;
    fc1->forward(scheduler, Fc1Out, Input);
    cs::Tensor GeluOut;
    cs::compute::GeLU::forward(scheduler, gelu_state, GeluOut, Fc1Out);
    fc2->forward(scheduler, Output, GeluOut);
  }

  void backward(cs::DynamicScheduler scheduler, cs::Tensor &DInput,
                cs::Tensor &DOutput) {
    // Fc2
    cs::Tensor DGeluOut;
    fc2->backwardInput(scheduler, DGeluOut, DOutput);
    fc2->backwardParameter(scheduler, DOutput);
    // GeLU
    cs::Tensor DFc1Out;
    cs::compute::GeLU::backward(scheduler, gelu_state, DFc1Out, DGeluOut);
    // Fc1
    fc1->backwardInput(scheduler, DInput, DFc1Out);
    fc1->backwardParameter(scheduler, DFc1Out);
  }
};

struct Block : cs::module::Module {
  const ModelConfig &config;
  cs::DynamicScheduler scheduler;
  cs::module::LayerNorm ln1{nullptr}, ln2{nullptr};
  std::shared_ptr<Attn> attn;
  std::shared_ptr<MLP> mlp;

  Block(cs::DynamicScheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    ln1 = register_module(
        "ln1", cs::module::LayerNorm(
                   scheduler, cs::compute::LayerNorm::Options(config.n_embd)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    attn = register_module("attn", std::make_shared<Attn>(scheduler, config));
    ln2 = register_module(
        "ln2", cs::module::LayerNorm(
                   scheduler, cs::compute::LayerNorm::Options(config.n_embd)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    mlp = register_module("mlp", std::make_shared<MLP>(scheduler, config));
  }

  void forward(cs::DynamicScheduler scheduler, cs::Tensor &Output,
               cs::Tensor &Input) {
    cs::Tensor LN1Out;
    ln1->forward(scheduler, LN1Out, Input);
    cs::Tensor AttnOut;
    attn->forward(scheduler, AttnOut, LN1Out);
    cs::Tensor ResAttnOut;
    cs::compute::Add::forward(scheduler, ResAttnOut, Input, AttnOut);
    cs::Tensor LN2Out;
    ln2->forward(scheduler, LN2Out, ResAttnOut);
    cs::Tensor MLPOut;
    mlp->forward(scheduler, MLPOut, LN2Out);
    cs::compute::Add::forward(scheduler, Output, ResAttnOut, MLPOut);
  }

  void backward(cs::DynamicScheduler scheduler, cs::Tensor &DInput,
                cs::Tensor &DOutput) {
    auto DMLPOut = DOutput;
    cs::Tensor DLN2Out;
    mlp->backward(scheduler, DLN2Out, DMLPOut);
    cs::Tensor DResAttnOut;
    ln2->backward(scheduler, DResAttnOut, DLN2Out);
    cs::Tensor DAttnOut;
    cs::compute::Add::forward(scheduler, DAttnOut, DResAttnOut, DMLPOut);
    cs::Tensor DLN1Out;
    attn->backward(scheduler, DLN1Out, DAttnOut);
    cs::Tensor DInput_;
    ln1->backward(scheduler, DInput_, DLN1Out);
    cs::compute::Add::forward(scheduler, DInput, DInput_, DAttnOut);
  }
};

struct GPT2 : cs::module::Module {
  const ModelConfig &config;
  cs::DynamicScheduler scheduler;
  cs::module::Embedding wte{nullptr}, wpe{nullptr};
  std::vector<std::shared_ptr<Block>> h;
  cs::module::LayerNorm lnf{nullptr};
  cs::module::Linear lm_head{nullptr};
  cs::Tensor Pos;

  GPT2(cs::DynamicScheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    cs::compute::Utils::arange(
        scheduler, Pos, 0, config.block_size,
        torch::TensorOptions().dtype(torch::kInt64).device(config.device));
    wte = register_module(
        "wte",
        cs::module::Embedding(scheduler, cs::compute::Embedding::Options(
                                             config.pad_size, config.n_embd)
                                             .device(config.device)
                                             .dtype(config.dtype)));
    wpe = register_module(
        "wpe",
        cs::module::Embedding(scheduler, cs::compute::Embedding::Options(
                                             config.block_size, config.n_embd)
                                             .device(config.device)
                                             .dtype(config.dtype)));
    // dropout
    for (int layer = 0; layer < config.n_layer; ++layer) {
      // Adding layers to the module list
      h.push_back(register_module(std::to_string(layer),
                                  std::make_shared<Block>(scheduler, config)));
    }
    lnf = register_module(
        "lnf", cs::module::LayerNorm(
                   scheduler, cs::compute::LayerNorm::Options(config.n_embd)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    lm_head = register_module(
        "lm_head",
        cs::module::Linear(scheduler, cs::compute::Linear::Options(
                                          config.n_embd, config.pad_size)
                                          .bias(config.use_bias)
                                          .device(config.device)
                                          .dtype(config.dtype)));
  }

  void forward(cs::DynamicScheduler scheduler, cs::Tensor &Output,
               cs::Tensor &Input) {
    cs::Tensor EmbOut;
    {
      cs::Tensor WteOut, WpeOut;
      wte->forward(scheduler, WteOut, Input);
      wpe->forward(scheduler, WpeOut, Pos);
      cs::Tensor PeOutBroadcast;
      cs::compute::Utils::broadcast_to(scheduler, PeOutBroadcast, WpeOut,
                                       WteOut.sizes());
      cs::compute::Add::forward(scheduler, EmbOut, WteOut, PeOutBroadcast);
    }
    auto BlockOut = EmbOut;
    for (auto &block : h) {
      block->forward(scheduler, BlockOut, BlockOut);
    }
    cs::Tensor LNfOut;
    lnf->forward(scheduler, LNfOut, BlockOut);
    lm_head->forward(scheduler, Output, LNfOut);
  }

  void backward(cs::DynamicScheduler scheduler, cs::Tensor &DOutput) {
    cs::Tensor DLNfOut;
    lm_head->backwardInput(scheduler, DLNfOut, DOutput);
    lm_head->backwardParameter(scheduler, DOutput);
    cs::Tensor DBlockOut;
    lnf->backward(scheduler, DBlockOut, DLNfOut);
    for (auto &block : h) {
      block->backward(scheduler, DBlockOut, DBlockOut);
    }
    auto DEmbOut = DBlockOut;
    {
      cs::Tensor DEmbOutSum0;
      cs::compute::Utils::sum(scheduler, DEmbOutSum0, DEmbOut, 0);
      wpe->backward(scheduler, DEmbOutSum0);
      wte->backward(scheduler, DEmbOut);
    }
  }
};

void train() {
  // torch::manual_seed(42);
  cs::DynamicScheduler scheduler{0};
  ModelConfig modelConfig;
  TrainConfig trainConfig;
  DataConfig dataConfig;
  std::unique_ptr<GPT2> model;
  cs::Tensor input, target;
  cs::Tensor loss;
  const torch::TensorOptions option =
      torch::TensorOptions().dtype(torch::kInt64).device(modelConfig.device);
  const torch::TensorOptions act_option = torch::TensorOptions()
                                              .dtype(modelConfig.dtype)
                                              .device(modelConfig.device);

  std::cout << "Prepare Dataset" << std::endl;

  cs::data::LlmDataset dataset{{dataConfig.path}};
  cs::data::LlmDataLoader dataloader{
      dataset, modelConfig.batch_size, 4, false, 0, 1};

  std::cout << "Init" << std::endl;
  model = std::make_unique<GPT2>(scheduler, modelConfig);

  std::shared_ptr<cs::compute::CrossEntropy::State> loss_state;
  cs::compute::CrossEntropy::init(scheduler, loss_state);

  cs::optimizer::AdamW::init(scheduler, model,
                             cs::optimizer::AdamW::Options{}
                                 .lr(trainConfig.lr)
                                 .beta1(trainConfig.beta1)
                                 .beta2(trainConfig.beta2)
                                 .weight_decay(trainConfig.weight_decay));

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < trainConfig.epoch; ++i) {
    std::cout << "Epoch " << i << std::endl;
    for (int j = 0; j < dataloader.iterationsPerEpoch(); ++j) {
      display_progress(j, dataloader.iterationsPerEpoch(),
                       " Training: ");  // Display progress for the inner loop

      dataloader.load(scheduler, input, target);
      // Forward
      cs::Tensor output;
      model->forward(scheduler, output, input);

      // compute loss
      cs::compute::Utils::view(scheduler, output, output,
                               {modelConfig.batch_size * modelConfig.block_size,
                                modelConfig.pad_size});
      cs::compute::Utils::view(
          scheduler, target, target,
          {modelConfig.batch_size * modelConfig.block_size});
      cs::compute::CrossEntropy::forward(scheduler, loss_state, loss, output,
                                         target);

      // Backward
      cs::Tensor grad_output;
      cs::compute::CrossEntropy::backward(scheduler, loss_state, grad_output);

      model->backward(scheduler, grad_output);

      // Optimizer step
      cs::optimizer::AdamW::step(scheduler, model);

      if ((i + 1) % trainConfig.wait_every_step == 0 or
          i + 1 == trainConfig.epoch) {
        model->lm_head->state()->forward.weight.wait();
      }
    }
    std::cout << "loss: " << loss << std::endl;
  }
  auto time_stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      time_stop - time_start);
  std::cout << "Time taken by loop: " << duration.count() / 1000 << " s"
            << std::endl;
}

int main() {
  train();
  return 0;
}
