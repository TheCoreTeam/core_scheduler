#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "communication/all_reduce.h"
#include "communication/communication.h"
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
  const int64_t n_embd = 2048;
  const int64_t n_head = 32;
  const int64_t n_layer = 22;
  const bool use_bias = false;
  const float dropout = 0.0;
  const float epsilon = 1e-5;
  torch::Device device{torch::kCUDA,
                       (torch::DeviceIndex)dllm::communication::getCommWorld(
                           dllm::communication::NCCL)
                           .getRank()};
  torch::Dtype dtype = torch::kBFloat16;
};

struct TrainConfig {
  const int64_t epoch = 5;
  const int64_t wait_every_step = 1;
  const int64_t bucket_size = 30 * 1024 * 1024;  // 60M
  double lr = 5e-4;
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

struct Attn : dllm::module::Module {
  dllm::Scheduler scheduler;
  const ModelConfig &config;
  dllm::module::Linear c_attn{nullptr}, c_proj{nullptr};
  std::shared_ptr<dllm::compute::ScaledDotProductFlashAttention::State>
      attn_state;

  Attn(dllm::Scheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    c_attn = register_module(
        "c_attn", dllm::module::Linear(scheduler,
                                       dllm::compute::Linear::Options{
                                           config.n_embd, 3 * config.n_embd}
                                           .bias(config.use_bias)
                                           .device(config.device)
                                           .dtype(config.dtype)));
    // we don't need to register flash attention to module because it does not
    // have parameters.
    dllm::compute::ScaledDotProductFlashAttention::init(
        scheduler, attn_state,
        dllm::compute::ScaledDotProductFlashAttention::Options{}.is_causal(
            true));
    c_proj = register_module(
        "c_proj", dllm::module::Linear(
                      scheduler, dllm::compute::Linear::Options{config.n_embd,
                                                                config.n_embd}
                                     .bias(config.use_bias)
                                     .device(config.device)
                                     .dtype(config.dtype)));
  }

  void forward(dllm::Scheduler scheduler, dllm::Tensor &Output,
               dllm::Tensor &Input) {
    // Fc Attn
    dllm::Tensor FcAttnOut;
    c_attn->forward(scheduler, FcAttnOut, Input);

    // Attn
    dllm::Tensor AttnOutView;
    {
      std::vector<dllm::Tensor> qkvSplit;
      dllm::compute::Utils::split(scheduler, qkvSplit, FcAttnOut, config.n_embd,
                                  -1);

      auto &q = qkvSplit[0];
      auto &k = qkvSplit[1];
      auto &v = qkvSplit[2];

      dllm::Tensor qview;
      dllm::compute::Utils::view(
          scheduler, qview, q,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});
      dllm::Tensor kview;
      dllm::compute::Utils::view(
          scheduler, kview, k,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});
      dllm::Tensor vview;
      dllm::compute::Utils::view(
          scheduler, vview, v,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});

      dllm::Tensor AttnOut;
      dllm::compute::ScaledDotProductFlashAttention::forward(
          scheduler, attn_state, AttnOut, qview, kview, vview);

      dllm::compute::Utils::view(
          scheduler, AttnOutView, AttnOut,
          {config.batch_size, config.block_size, config.n_embd});
    }

    // Fc Proj
    c_proj->forward(scheduler, Output, AttnOutView);
  }

  void backward(dllm::Scheduler scheduler,
                const dllm::communication::Comm &comm,
                dllm::communication::AllReduceBucket allreduce_bucket,
                dllm::Tensor &DInput, dllm::Tensor &DOutput,
                bool require_backward_grad_sync = false) {
    // Fc Proj, we first run backward for input and then backward for parameter.
    dllm::Tensor DAttnOut;
    c_proj->backwardInput(scheduler, DAttnOut, DOutput);
    c_proj->backwardParameter(scheduler, DOutput);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 c_proj->state()->forward.grad_weight);
    }

    // Attn
    dllm::Tensor DFcAttnOut;
    {
      dllm::compute::Utils::view(
          scheduler, DAttnOut, DAttnOut,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});

      dllm::Tensor dq, dk, dv;
      dllm::compute::ScaledDotProductFlashAttention::backward(
          scheduler, attn_state, dq, dk, dv, DAttnOut);

      dllm::compute::Utils::view(
          scheduler, dq, dq,
          {config.batch_size, config.block_size, config.n_embd});
      dllm::compute::Utils::view(
          scheduler, dk, dk,
          {config.batch_size, config.block_size, config.n_embd});
      dllm::compute::Utils::view(
          scheduler, dv, dv,
          {config.batch_size, config.block_size, config.n_embd});
      dllm::compute::Utils::cat(scheduler, DFcAttnOut, {dq, dk, dv}, -1);
    }

    // Fc Attn, we first run backward for input and then backward for parameter.
    c_attn->backwardInput(scheduler, DInput, DFcAttnOut);
    c_attn->backwardParameter(scheduler, DFcAttnOut);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 c_attn->state()->forward.grad_weight);
    }
  }
};

struct MLP : dllm::module::Module {
  dllm::Scheduler scheduler;
  const ModelConfig &config;
  dllm::module::Linear fc1{nullptr}, fc2{nullptr};
  std::shared_ptr<dllm::compute::GeLU::State> gelu_state;

  MLP(dllm::Scheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    fc1 = register_module(
        "fc1", dllm::module::Linear(
                   scheduler, dllm::compute::Linear::Options{config.n_embd,
                                                             4 * config.n_embd}
                                  .bias(config.use_bias)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    dllm::compute::GeLU::init(scheduler, gelu_state);
    fc2 = register_module(
        "fc2", dllm::module::Linear(
                   scheduler, dllm::compute::Linear::Options{4 * config.n_embd,
                                                             config.n_embd}
                                  .bias(config.use_bias)
                                  .device(config.device)
                                  .dtype(config.dtype)));
  }

  void forward(dllm::Scheduler scheduler, dllm::Tensor &Output,
               dllm::Tensor &Input) {
    dllm::Tensor Fc1Out;
    fc1->forward(scheduler, Fc1Out, Input);
    dllm::Tensor GeluOut;
    dllm::compute::GeLU::forward(scheduler, gelu_state, GeluOut, Fc1Out);
    fc2->forward(scheduler, Output, GeluOut);
  }

  void backward(const dllm::Scheduler &scheduler,
                const dllm::communication::Comm &comm,
                dllm::communication::AllReduceBucket allreduce_bucket,
                dllm::Tensor &DInput, dllm::Tensor &DOutput,
                bool require_backward_grad_sync = false) {
    // Fc2
    dllm::Tensor DGeluOut;
    fc2->backwardInput(scheduler, DGeluOut, DOutput);
    fc2->backwardParameter(scheduler, DOutput);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 fc2->state()->forward.grad_weight);
    }
    // GeLU
    dllm::Tensor DFc1Out;
    dllm::compute::GeLU::backward(scheduler, gelu_state, DFc1Out, DGeluOut);
    // Fc1
    fc1->backwardInput(scheduler, DInput, DFc1Out);
    fc1->backwardParameter(scheduler, DFc1Out);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 fc1->state()->forward.grad_weight);
    }
  }
};

struct Block : dllm::module::Module {
  const ModelConfig &config;
  dllm::Scheduler scheduler;
  dllm::module::LayerNorm ln1{nullptr}, ln2{nullptr};
  std::shared_ptr<Attn> attn;
  std::shared_ptr<MLP> mlp;

  Block(dllm::Scheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    ln1 = register_module(
        "ln1", dllm::module::LayerNorm(
                   scheduler, dllm::compute::LayerNorm::Options(config.n_embd)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    attn = register_module("attn", std::make_shared<Attn>(scheduler, config));
    ln2 = register_module(
        "ln2", dllm::module::LayerNorm(
                   scheduler, dllm::compute::LayerNorm::Options(config.n_embd)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    mlp = register_module("mlp", std::make_shared<MLP>(scheduler, config));
  }

  void forward(dllm::Scheduler scheduler, dllm::Tensor &Output,
               dllm::Tensor &Input) {
    dllm::Tensor LN1Out;
    ln1->forward(scheduler, LN1Out, Input);
    dllm::Tensor AttnOut;
    attn->forward(scheduler, AttnOut, LN1Out);
    dllm::Tensor ResAttnOut;
    dllm::compute::Add::forward(scheduler, ResAttnOut, Input, AttnOut);
    dllm::Tensor LN2Out;
    ln2->forward(scheduler, LN2Out, ResAttnOut);
    dllm::Tensor MLPOut;
    mlp->forward(scheduler, MLPOut, LN2Out);
    dllm::compute::Add::forward(scheduler, Output, ResAttnOut, MLPOut);
  }

  void backward(const dllm::Scheduler &scheduler,
                const dllm::communication::Comm &comm,
                dllm::communication::AllReduceBucket allreduce_bucket,
                dllm::Tensor &DInput, dllm::Tensor &DOutput,
                bool require_backward_grad_sync = false) {
    auto DMLPOut = DOutput;
    dllm::Tensor DLN2Out;
    mlp->backward(scheduler, comm, allreduce_bucket, DLN2Out, DMLPOut,
                  require_backward_grad_sync);
    dllm::Tensor DResAttnOut;
    ln2->backward(scheduler, DResAttnOut, DLN2Out);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 ln2->state()->forward.grad_weight);
      allreduce_bucket.push_back(scheduler, comm,
                                 ln2->state()->forward.grad_bias);
    }
    dllm::Tensor DAttnOut;
    dllm::compute::Add::forward(scheduler, DAttnOut, DResAttnOut, DMLPOut);
    dllm::Tensor DLN1Out;
    attn->backward(scheduler, comm, allreduce_bucket, DLN1Out, DAttnOut,
                   require_backward_grad_sync);
    dllm::Tensor DInput_;
    ln1->backward(scheduler, DInput_, DLN1Out);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 ln1->state()->forward.grad_weight);
      allreduce_bucket.push_back(scheduler, comm,
                                 ln1->state()->forward.grad_bias);
    }
    dllm::compute::Add::forward(scheduler, DInput, DInput_, DAttnOut);
  }
};

struct GPT2 : dllm::module::Module {
  const ModelConfig &config;
  dllm::Scheduler scheduler;
  dllm::module::Embedding wte{nullptr}, wpe{nullptr};
  std::vector<std::shared_ptr<Block>> h;
  dllm::module::LayerNorm lnf{nullptr};
  dllm::module::Linear lm_head{nullptr};
  dllm::Tensor Pos;

  GPT2(dllm::Scheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    dllm::compute::Utils::arange(
        scheduler, Pos, 0, config.block_size,
        torch::TensorOptions().dtype(torch::kInt64).device(config.device));
    wte = register_module(
        "wte",
        dllm::module::Embedding(scheduler, dllm::compute::Embedding::Options(
                                               config.pad_size, config.n_embd)
                                               .device(config.device)
                                               .dtype(config.dtype)));
    wpe = register_module(
        "wpe",
        dllm::module::Embedding(scheduler, dllm::compute::Embedding::Options(
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
        "lnf", dllm::module::LayerNorm(
                   scheduler, dllm::compute::LayerNorm::Options(config.n_embd)
                                  .device(config.device)
                                  .dtype(config.dtype)));
    lm_head = register_module(
        "lm_head",
        dllm::module::Linear(scheduler, dllm::compute::Linear::Options(
                                            config.n_embd, config.pad_size)
                                            .bias(config.use_bias)
                                            .device(config.device)
                                            .dtype(config.dtype)));
  }

  void forward(dllm::Scheduler scheduler, dllm::Tensor &Output,
               dllm::Tensor &Input) {
    dllm::Tensor EmbOut;
    {
      dllm::Tensor WteOut, WpeOut;
      wte->forward(scheduler, WteOut, Input);
      wpe->forward(scheduler, WpeOut, Pos);
      dllm::Tensor PeOutBroadcast;
      dllm::compute::Utils::broadcast_to(scheduler, PeOutBroadcast, WpeOut,
                                         WteOut.sizes());
      dllm::compute::Add::forward(scheduler, EmbOut, WteOut, PeOutBroadcast);
    }
    auto BlockOut = EmbOut;
    for (auto &block : h) {
      block->forward(scheduler, BlockOut, BlockOut);
    }
    dllm::Tensor LNfOut;
    lnf->forward(scheduler, LNfOut, BlockOut);
    lm_head->forward(scheduler, Output, LNfOut);
  }

  void backward(const dllm::Scheduler &scheduler,
                const dllm::communication::Comm &comm,
                dllm::communication::AllReduceBucket allreduce_bucket,
                dllm::Tensor &DOutput,
                bool require_backward_grad_sync = false) {
    dllm::Tensor DLNfOut;
    lm_head->backwardInput(scheduler, DLNfOut, DOutput);
    lm_head->backwardParameter(scheduler, DOutput);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 lm_head->state()->forward.grad_weight);
    }
    dllm::Tensor DBlockOut;
    lnf->backward(scheduler, DBlockOut, DLNfOut);
    if (require_backward_grad_sync) {
      allreduce_bucket.push_back(scheduler, comm,
                                 lnf->state()->forward.grad_weight);
      allreduce_bucket.push_back(scheduler, comm,
                                 lnf->state()->forward.grad_bias);
    }
    for (auto &block : h) {
      block->backward(scheduler, comm, allreduce_bucket, DBlockOut, DBlockOut,
                      require_backward_grad_sync);
    }
    auto DEmbOut = DBlockOut;
    {
      dllm::Tensor DEmbOutSum0;
      dllm::compute::Utils::sum(scheduler, DEmbOutSum0, DEmbOut, 0);
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

void train() {
  // torch::manual_seed(42);
  ModelConfig modelConfig;
  TrainConfig trainConfig;
  DataConfig dataConfig;
  std::unique_ptr<GPT2> model;
  dllm::Tensor input, target;
  dllm::Tensor loss;
  dllm::Tensor output, grad_output;
  const dllm::communication::Comm comm =
      dllm::communication::getCommWorld(dllm::communication::NCCL);
  dllm::DynamicScheduler scheduler{(int)comm.getRank()};
  const torch::TensorOptions option =
      torch::TensorOptions().dtype(torch::kInt64).device(modelConfig.device);
  const torch::TensorOptions act_option = torch::TensorOptions()
                                              .dtype(modelConfig.dtype)
                                              .device(modelConfig.device);

  std::cout << "Prepare Dataset" << std::endl;

  dllm::data::LlmDataset dataset{{dataConfig.path}};
  dllm::data::LlmDataLoader dataloader{
      dataset, modelConfig.batch_size, 4,
      false,   comm.getRank(),         comm.getSize()};

  std::cout << "Init" << std::endl;
  model = std::make_unique<GPT2>(scheduler, modelConfig);
  std::shared_ptr<dllm::compute::CrossEntropy::State> loss_state;
  dllm::compute::CrossEntropy::init(scheduler, loss_state);

  bool require_backward_grad_sync = true;
  dllm::communication::AllReduceBucket allreduce_bucket(
      trainConfig.bucket_size, dllm::communication::SUM);

  dllm::optimizer::AdamW::init(scheduler, model,
                               dllm::optimizer::AdamW::Options{}
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

      dataloader.load(scheduler, input, target);
      // Forward
      model->forward(scheduler, output, input);

      // compute loss
      dllm::compute::Utils::view(
          scheduler, output, output,
          {modelConfig.batch_size * modelConfig.block_size,
           modelConfig.pad_size});
      dllm::compute::Utils::view(
          scheduler, target, target,
          {modelConfig.batch_size * modelConfig.block_size});
      dllm::compute::CrossEntropy::forward(scheduler, loss_state, loss, output,
                                           target);

      // Backward
      dllm::compute::CrossEntropy::backward(scheduler, loss_state, grad_output);

      model->backward(scheduler, comm, allreduce_bucket, grad_output,
                      require_backward_grad_sync);

      // Optimizer step
      dllm::optimizer::AdamW::step(scheduler, model);

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

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  train();
  MPI_Finalize();
  return 0;
}
