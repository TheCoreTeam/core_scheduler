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
#include <mpi.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "communication/all_reduce.h"
#include "communication/communication.h"
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
  const int64_t n_embd = 2048;
  const int64_t n_head = 32;
  const int64_t n_layer = 22;
  const bool use_bias = false;
  const float dropout = 0.0;
  const float epsilon = 1e-5;
  torch::Device device{
      torch::kCUDA,
      static_cast<torch::DeviceIndex>(
          cs::communication::getCommWorld(cs::communication::NCCL).getRank())};
  torch::Dtype dtype = torch::kBFloat16;
};

struct TrainConfig {
  const int64_t epoch = 50;
  const int64_t wait_every_step = 1;
  const int64_t bucket_size = 30 * 1024 * 1024;  // 30M
  double lr = 1e-5;
  double beta1 = 0.9;
  double beta2 = 0.95;
  double weight_decay = 1e-1;
};

// Function to display the progress bar
void display_progress(int completed, int total, const std::string &prefix = "",
                      int rank = 0) {
  int width = 50;  // Width of the progress bar
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
  if (rank == 0) {
    std::cout.flush();
  } else {
    std::cout << std::endl;
    std::cout.flush();
  }
  if (completed == total - 1) {
    std::cout << "" << std::endl;
  }
}

struct Attn : cs::module::Module {
  const cs::Scheduler &scheduler;
  const ModelConfig &config;
  cs::module::Linear c_attn{nullptr}, c_proj{nullptr};
  std::shared_ptr<cs::compute::ScaledDotProductFlashAttention::State>
      attn_state;

  Attn(const cs::Scheduler &scheduler, const ModelConfig &config)
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
    attn_state = cs::compute::ScaledDotProductFlashAttention::init(
        scheduler,
        cs::compute::ScaledDotProductFlashAttention::Options{}.is_causal(true));
    c_proj = register_module(
        "c_proj", cs::module::Linear(
                      scheduler,
                      cs::compute::Linear::Options{config.n_embd, config.n_embd}
                          .bias(config.use_bias)
                          .device(config.device)
                          .dtype(config.dtype)));
  }

  cs::Tensor forward(const cs::Scheduler &scheduler, cs::Tensor &Input) {
    // Fc Attn
    auto FcAttnOut = c_attn->forward(scheduler, Input);

    // Attn
    cs::Tensor AttnOutView;
    {
      auto qkvSplit =
          cs::compute::Utils::split(scheduler, FcAttnOut, config.n_embd, -1);

      auto &q = qkvSplit[0];
      auto &k = qkvSplit[1];
      auto &v = qkvSplit[2];

      auto qview = cs::compute::Utils::view(
          scheduler, q,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});
      auto kview = cs::compute::Utils::view(
          scheduler, k,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});
      auto vview = cs::compute::Utils::view(
          scheduler, v,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});

      auto AttnOut = cs::compute::ScaledDotProductFlashAttention::forward(
          scheduler, attn_state, qview, kview, vview);

      AttnOutView = cs::compute::Utils::view(
          scheduler, AttnOut,
          {config.batch_size, config.block_size, config.n_embd});
    }

    // Fc Proj
    return c_proj->forward(scheduler, AttnOutView);
  }

  cs::Tensor backward(
      const cs::Scheduler &scheduler, const cs::communication::Comm &comm,
      const cs::communication::AllReduceBucket &allreduce_bucket,
      cs::Tensor &DOutput, bool require_backward_grad_sync = false) {
    // Fc Proj, we first run backward for input and then backward for parameter.
    auto DAttnOut = c_proj->backwardInput(scheduler, DOutput);
    c_proj->backwardParameter(scheduler, DOutput);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 c_proj->state()->forward.grad_weight);
    }

    // Attn
    cs::Tensor DFcAttnOut;
    {
      DAttnOut = cs::compute::Utils::view(
          scheduler, DAttnOut,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});

      auto [dq, dk, dv] = cs::compute::ScaledDotProductFlashAttention::backward(
          scheduler, attn_state, DAttnOut);

      dq = cs::compute::Utils::view(
          scheduler, dq, {config.batch_size, config.block_size, config.n_embd});
      dk = cs::compute::Utils::view(
          scheduler, dk, {config.batch_size, config.block_size, config.n_embd});
      dv = cs::compute::Utils::view(
          scheduler, dv, {config.batch_size, config.block_size, config.n_embd});
      DFcAttnOut = cs::compute::Utils::cat(scheduler, {dq, dk, dv}, -1);
    }

    // Fc Attn, we first run backward for input and then backward for parameter.
    auto DInput = c_attn->backwardInput(scheduler, DFcAttnOut);
    c_attn->backwardParameter(scheduler, DFcAttnOut);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 c_attn->state()->forward.grad_weight);
    }
    return DInput;
  }
};

struct MLP : cs::module::Module {
  const cs::Scheduler &scheduler;
  const ModelConfig &config;
  cs::module::Linear fc1{nullptr}, fc2{nullptr};
  std::shared_ptr<cs::compute::GeLU::State> gelu_state;

  MLP(const cs::Scheduler &scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    fc1 = register_module(
        "fc1", cs::module::Linear(
                   scheduler, cs::compute::Linear::Options{config.n_embd,
                                                           4 * config.n_embd}
                                  .bias(config.use_bias)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    gelu_state = cs::compute::GeLU::init(scheduler);
    fc2 = register_module(
        "fc2", cs::module::Linear(
                   scheduler, cs::compute::Linear::Options{4 * config.n_embd,
                                                           config.n_embd}
                                  .bias(config.use_bias)
                                  .device(config.device)
                                  .dtype(config.dtype)));
  }

  cs::Tensor forward(const cs::Scheduler &scheduler, const cs::Tensor &Input) {
    auto Fc1Out = fc1->forward(scheduler, Input);
    auto GeluOut = cs::compute::GeLU::forward(scheduler, gelu_state, Fc1Out);
    return fc2->forward(scheduler, GeluOut);
  }

  cs::Tensor backward(
      const cs::Scheduler &scheduler, const cs::communication::Comm &comm,
      const cs::communication::AllReduceBucket &allreduce_bucket,
      cs::Tensor &DOutput, bool require_backward_grad_sync = false) {
    // Fc2
    auto DGeluOut = fc2->backwardInput(scheduler, DOutput);
    fc2->backwardParameter(scheduler, DOutput);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 fc2->state()->forward.grad_weight);
    }
    // GeLU
    auto DFc1Out = cs::compute::GeLU::backward(scheduler, gelu_state, DGeluOut);
    // Fc1
    auto DInput = fc1->backwardInput(scheduler, DFc1Out);
    fc1->backwardParameter(scheduler, DFc1Out);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 fc1->state()->forward.grad_weight);
    }
    return DInput;
  }
};

struct Block : cs::module::Module {
  const ModelConfig &config;
  const cs::Scheduler &scheduler;
  cs::module::LayerNorm ln1{nullptr}, ln2{nullptr};
  std::shared_ptr<Attn> attn;
  std::shared_ptr<MLP> mlp;

  Block(const cs::Scheduler &scheduler, const ModelConfig &config)
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

  cs::Tensor forward(const cs::Scheduler &scheduler, cs::Tensor &Input) {
    auto LN1Out = ln1->forward(scheduler, Input);
    auto AttnOut = attn->forward(scheduler, LN1Out);
    auto ResAttnOut = cs::compute::Utils::add(scheduler, Input, AttnOut);
    auto LN2Out = ln2->forward(scheduler, ResAttnOut);
    auto MLPOut = mlp->forward(scheduler, LN2Out);
    return cs::compute::Utils::add(scheduler, ResAttnOut, MLPOut);
  }

  cs::Tensor backward(
      const cs::Scheduler &scheduler, const cs::communication::Comm &comm,
      const cs::communication::AllReduceBucket &allreduce_bucket,
      cs::Tensor &DOutput, bool require_backward_grad_sync = false) {
    auto DMLPOut = DOutput;
    auto DLN2Out = mlp->backward(scheduler, comm, allreduce_bucket, DMLPOut,
                                 require_backward_grad_sync);
    auto DResAttnOut = ln2->backward(scheduler, DLN2Out);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 ln2->state()->forward.grad_weight);
      allreduce_bucket.push_back(scheduler, comm,
                                 ln2->state()->forward.grad_bias);
    }
    auto DAttnOut = cs::compute::Utils::add(scheduler, DResAttnOut, DMLPOut);
    auto DLN1Out = attn->backward(scheduler, comm, allreduce_bucket, DAttnOut,
                                  require_backward_grad_sync);
    auto DInput = ln1->backward(scheduler, DLN1Out);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 ln1->state()->forward.grad_weight);
      allreduce_bucket.push_back(scheduler, comm,
                                 ln1->state()->forward.grad_bias);
    }
    return cs::compute::Utils::add(scheduler, DInput, DAttnOut);
  }
};

struct GPT2 : cs::module::Module {
  const ModelConfig &config;
  const cs::Scheduler &scheduler;
  cs::module::Embedding wte{nullptr}, wpe{nullptr};
  std::vector<std::shared_ptr<Block>> h;
  cs::module::LayerNorm lnf{nullptr};
  cs::module::Linear lm_head{nullptr};
  cs::Tensor Pos;

  GPT2(const cs::Scheduler &scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    Pos = cs::compute::Utils::arange(
        scheduler, 0, config.block_size,
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

  cs::Tensor forward(const cs::Scheduler &scheduler, cs::Tensor &Input) {
    cs::Tensor EmbOut;
    {
      auto WteOut = wte->forward(scheduler, Input);
      auto WpeOut = wpe->forward(scheduler, Pos);
      auto PeOutBroadcast =
          cs::compute::Utils::broadcast_to(scheduler, WpeOut, WteOut.sizes());
      EmbOut = cs::compute::Utils::add(scheduler, WteOut, PeOutBroadcast);
    }
    auto BlockOut = EmbOut;
    for (auto &block : h) {
      BlockOut = block->forward(scheduler, BlockOut);
    }
    auto LNfOut = lnf->forward(scheduler, BlockOut);
    return lm_head->forward(scheduler, LNfOut);
  }

  void backward(const cs::Scheduler &scheduler,
                const cs::communication::Comm &comm,
                const cs::communication::AllReduceBucket &allreduce_bucket,
                cs::Tensor &DOutput, bool require_backward_grad_sync = false) {
    auto DLNfOut = lm_head->backwardInput(scheduler, DOutput);
    lm_head->backwardParameter(scheduler, DOutput);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 lm_head->state()->forward.grad_weight);
    }
    auto DBlockOut = lnf->backward(scheduler, DLNfOut);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 lnf->state()->forward.grad_weight);
      allreduce_bucket.push_back(scheduler, comm,
                                 lnf->state()->forward.grad_bias);
    }
    for (auto &block : h) {
      DBlockOut = block->backward(scheduler, comm, allreduce_bucket, DBlockOut,
                                  require_backward_grad_sync);
    }
    auto DEmbOut = DBlockOut;
    {
      auto DEmbOutSum0 = cs::compute::Utils::sum(scheduler, DEmbOut, 0);
      wpe->backward(scheduler, DEmbOutSum0);
      if (require_backward_grad_sync) {
        allreduce_bucket.push_back(scheduler, comm,
                                   wpe->state()->forward.grad_weight);
      }
      wte->backward(scheduler, DEmbOut);
      if (require_backward_grad_sync) {
        allreduce_bucket.push_back(scheduler, comm,
                                   wte->state()->forward.grad_weight);
      }
    }
  }
};

void train(const std::string path) {
  torch::manual_seed(42);
  ModelConfig modelConfig;
  TrainConfig trainConfig;
  std::unique_ptr<GPT2> model;
  cs::Tensor loss;
  cs::Tensor output, grad_output;
  const cs::communication::Comm comm =
      cs::communication::getCommWorld(cs::communication::NCCL);
  cs::DynamicScheduler scheduler{(int)comm.getRank()};
  const torch::TensorOptions option =
      torch::TensorOptions().dtype(torch::kInt64).device(modelConfig.device);
  const torch::TensorOptions act_option = torch::TensorOptions()
                                              .dtype(modelConfig.dtype)
                                              .device(modelConfig.device);

  std::cout << "Prepare Dataset" << std::endl;

  cs::data::LlmDataset dataset{{path}};
  cs::data::LlmDataLoader dataloader{dataset, comm, modelConfig.batch_size, 4,
                                     false};

  std::cout << "Init" << std::endl;
  model = std::make_unique<GPT2>(scheduler, modelConfig);
  auto loss_state = cs::compute::CrossEntropy::init(scheduler);

  bool require_backward_grad_sync = true;
  cs::communication::AllReduceBucket allreduce_bucket(trainConfig.bucket_size,
                                                      cs::communication::SUM);

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
      display_progress(j, dataloader.iterationsPerEpoch(), " Training: ",
                       comm.getRank());  // Display progress for the inner loop

      auto data = dataloader.load(scheduler);
      auto input = data["input_ids"];
      auto target = data["labels"];
      // Forward
      output = model->forward(scheduler, input);

      // compute loss
      output = cs::compute::Utils::view(
          scheduler, output,
          {modelConfig.batch_size * modelConfig.block_size,
           modelConfig.pad_size});
      target = cs::compute::Utils::view(
          scheduler, target, {modelConfig.batch_size * modelConfig.block_size});
      loss = cs::compute::CrossEntropy::forward(scheduler, loss_state, output,
                                                target);

      // Backward
      grad_output = cs::compute::CrossEntropy::backward(scheduler, loss_state);

      model->backward(scheduler, comm, allreduce_bucket, grad_output,
                      require_backward_grad_sync);

      // Optimizer step
      cs::optimizer::AdamW::step(scheduler, model);

      if ((i + 1) % trainConfig.wait_every_step == 0 or
          i + 1 == trainConfig.epoch) {
        model->lm_head->state()->forward.weight.wait();
      }
    }
    cs::communication::AllReduce::runInplace(scheduler, comm, {loss},
                                             cs::communication::SUM);
    if (comm.getRank() == 0) {
      std::cout << "loss: " << loss << std::endl;
    }
  }
  auto time_stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      time_stop - time_start);
  if (comm.getRank() == 0) {
    std::cout << "Time taken by loop: " << duration.count() / 1000 << " s"
              << std::endl;
  }
}

int main(int argc, char **argv) {
  cxxopts::Options options("GPT2", "An GPT2 example");

  options.add_options()("path", "Dataset Path", cxxopts::value<std::string>());

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  const auto path = result["path"].as<std::string>();
  MPI_Init(&argc, &argv);
  train(path);
  MPI_Finalize();
  return 0;
}
