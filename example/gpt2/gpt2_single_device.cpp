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

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/nn/init.h>
#include <torch/torch.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "compute/cross_entropy.h"
#include "compute/embedding.h"
#include "compute/gelu.h"
#include "compute/init.h"
#include "compute/layer_norm.h"
#include "compute/linear.h"
#include "compute/scaled_dot_product_attention.h"
#include "compute/utils.h"
#include "data/dataloader.h"
#include "data/dataset.h"
#include "logger.h"
#include "module/adamw.h"
#include "autocast.h"
#include "module/embedding.h"
#include "module/layer_norm.h"
#include "module/linear.h"
#include "module/module.h"
#include "optimizer/adamw.h"
#include "tensor.h"
#include "threading/dynamic_scheduler.h"

struct ModelConfig {
  int64_t batch_size = 16;
  const int64_t block_size = 1024;
  const int64_t vocab_size = 50257;
  const int64_t pad_size = 50304;  // pad vocab_size to be more efficient
  const int64_t n_embd = 768;      // 2048
  const int64_t n_head = 12;       // 32
  const int64_t n_layer = 12;      // 22
  const bool use_bias = false;
  const double dropout = 0.0;
  const double epsilon = 1e-5;
  torch::Device device = torch::Device(torch::kCUDA, 0);
  torch::Dtype dtype = torch::kFloat32;  // model percision
  bool share_emb = false;  // share embedding weight or not. See https://arxiv.org/abs/1706.03762
};

struct TrainConfig {
  const std::string log_path = "../training_log.csv";
  int64_t epoch = 1;
  int64_t total_token_batch_size =
      1024 * 512;  // 524288, 2**19, about 0.5 tokens per batch
  int64_t warmup_steps = 715;
  int64_t max_steps = -1;
  int64_t check_every_steps = 1;
  int64_t val_every_steps = 250;
  int64_t save_every_steps = 5000;
  double max_lr = 6e-4;
  double min_lr = 0.1 * max_lr;
  double beta1 = 0.9;
  double beta2 = 0.95;
  double weight_decay = 1e-1;
  double eps = 1e-8;
  double grad_clip_value = 1.0;
  bool use_amp = true;
  torch::Dtype amp_dtype = torch::kBFloat16;  // amp percision: torch::kBFloat16, torch::kFloat16 
                                              // (Now we only support bf16)
  int64_t seed = 1337;
  int64_t wait_every_step = 1;
};

struct DataConfig {
  const std::string path = "/home/ly/main/dataset/fineweb-edu-10BT/train/";
  int64_t num_workers = 1;
  bool shuffle = false;
};

// Helper function to calculate the training arguments
std::unordered_map<std::string, double> getTrainArgs(
    int64_t num_samples, int64_t tokens_per_sample,
    int64_t global_token_batch_size, int64_t batch_size_per_dprank_per_micro_step,
    int64_t num_dprank, int64_t max_steps = -1, int64_t max_epochs = -1) {
  if (max_steps == -1 && max_epochs == -1) {
    throw std::invalid_argument(
        "At least one of max_steps or max_epochs must be provided.");
  }
  if (global_token_batch_size %
          (batch_size_per_dprank_per_micro_step * tokens_per_sample * num_dprank) !=
      0) {
    throw std::invalid_argument(
        "global_token_batch_size must be divisible by "
        "batch_size_per_dprank_per_micro_step * tokens_per_sample * num_dprank.");
  }

  int64_t tokens_per_dprank_per_step =
      tokens_per_sample * batch_size_per_dprank_per_micro_step;
  int64_t total_tokens_per_micro_step = tokens_per_dprank_per_step * num_dprank;
  int64_t grad_accum_steps =
      global_token_batch_size / total_tokens_per_micro_step;
  int64_t total_tokens_per_step =
      total_tokens_per_micro_step * grad_accum_steps;
  int64_t total_tokens_in_dataset = num_samples * tokens_per_sample;

  if (max_steps != -1 && max_epochs != -1) {
    int64_t calculated_max_steps =
        (max_epochs * total_tokens_in_dataset) / global_token_batch_size;
    double calculated_max_epochs = (max_steps * global_token_batch_size) /
                                   static_cast<double>(total_tokens_in_dataset);

    if (!(calculated_max_steps == max_steps &&
          static_cast<int>(calculated_max_epochs) == max_epochs)) {
      throw std::invalid_argument(
          "Inconsistent max_steps and max_epochs based on the dataset and "
          "configuration. "
          "Calculated max_steps from max_epochs: " +
          std::to_string(calculated_max_steps) +
          ", provided max_steps: " + std::to_string(max_steps) +
          ". "
          "Calculated max_epochs from max_steps: " +
          std::to_string(static_cast<int64_t>(calculated_max_epochs)) +
          ", provided max_epochs: " + std::to_string(max_epochs) + ".");
    }
  } else if (max_steps == -1) {
    max_steps =
        (max_epochs * total_tokens_in_dataset) / global_token_batch_size;
  } else if (max_epochs == -1) {
    max_epochs =
        (max_steps * global_token_batch_size) / total_tokens_in_dataset;
  }

  std::unordered_map<std::string, double> result;
  result["epochs"] = max_epochs;
  result["max_steps"] = max_steps;
  result["grad_accum_steps"] = grad_accum_steps;
  result["total_tokens_per_step"] = total_tokens_per_step;
  result["total_tokens_per_micro_step"] = total_tokens_per_micro_step;

  return result;
}

// Class to display the progress bar
struct ProgressBar {
  int total;
  int width;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> last_time;

  ProgressBar(int total_steps, int bar_width = 50)
      : total(total_steps), width(bar_width) {
    start_time = std::chrono::high_resolution_clock::now();
    last_time = start_time;
  }

  std::string format_time(std::chrono::seconds time) const {
    int minutes =
        std::chrono::duration_cast<std::chrono::minutes>(time).count();
    int seconds = (time - std::chrono::minutes(minutes)).count();
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << minutes << ":" << std::setw(2)
        << std::setfill('0') << seconds;
    return oss.str();
  }

  void display(int completed, const std::string &prefix = "", int rank = 0) {
    float progress = total > 1 ? static_cast<float>(completed) / (total - 1)
                               : 1.0;  // Adjusted to avoid division by zero
    int pos = width * progress;

    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::seconds elapsed_time =
        std::chrono::duration_cast<std::chrono::seconds>(current_time -
                                                         start_time);
    std::chrono::seconds step_time =
        std::chrono::duration_cast<std::chrono::seconds>(current_time -
                                                         last_time);
    std::chrono::seconds remaining_time(0);
    if (completed > 0) {
      double avg_step_time =
          elapsed_time.count() / static_cast<double>(completed);
      remaining_time = std::chrono::seconds(
          static_cast<long long>(avg_step_time * (total - completed - 1)));
    }

    // Clear the line before printing progress
    std::cout << "\r" << std::string(width + 50, ' ')
              << "\r";  // Clear the line

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
              << format_time(elapsed_time) << "<" << format_time(remaining_time)
              << " | " << int(progress * 100.0) << " %";

    // Ensure the progress bar always ends with a newline
    if (rank == 0) {
      std::cout.flush();
    } else {
      std::cout << std::endl;  // Force a newline if not the main rank or
                               // completing the bar
    }

    last_time = current_time;  // Update last_time after displaying progress
  }
};

struct LRScheduler {
  int warmup_steps;
  double max_lr;
  int max_steps;
  double min_lr;

  LRScheduler(int warmupSteps, double maxLr, int maxSteps, double minLr)
      : warmup_steps(warmupSteps),
        max_lr(maxLr),
        max_steps(maxSteps),
        min_lr(minLr) {}

  double get_lr(int step) const {
    // 1) Linear warmup for warmup_iters steps
    if (step < warmup_steps) {
      return max_lr * (step + 1) / warmup_steps;
    }
    // 2) If it > max_steps, return minimum learning rate
    if (step > max_steps) {
      return min_lr;
    }
    // 3) In between, use cosine decay down to minimum learning rate
    double decay_ratio =
        static_cast<double>(step - warmup_steps) / (max_steps - warmup_steps);
    assert(0 <= decay_ratio && decay_ratio <= 1);
    double coeff = 0.5 * (1.0 + cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
  }
};

struct Attn : cs::module::Module {
  cs::Scheduler scheduler;
  const ModelConfig &config;
  cs::module::Linear c_attn{nullptr}, c_proj{nullptr};
  std::shared_ptr<cs::compute::ScaledDotProductFlashAttention::State>
      attn_state;

  Attn(cs::Scheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    c_attn = register_module(
        "c_attn", cs::module::Linear(scheduler,
                                        cs::compute::Linear::Options{
                                            config.n_embd, 3 * config.n_embd}
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
                      scheduler, cs::compute::Linear::Options{config.n_embd,
                                                                 config.n_embd}
                                     .bias(config.use_bias)
                                     .device(config.device)
                                     .dtype(config.dtype)));

    _init_weights(scheduler, config);
  }

  void _init_weights(cs::Scheduler scheduler,
                     const ModelConfig &config) {
    cs::compute::Utils::normal_(scheduler, c_proj->state()->forward.weight, 0.0,
                                0.02 / sqrt(2 * config.n_layer));
  }

  cs::Tensor forward(const cs::Scheduler scheduler, cs::Tensor x) {
    // Fc Attn
    auto c_attn_out = c_attn->forward(scheduler, x);

    // Attn
    cs::Tensor attn_out;
    {
      auto qkv =
          cs::compute::Utils::split(scheduler, c_attn_out, config.n_embd, -1);

      auto &q = qkv[0];
      auto &k = qkv[1];
      auto &v = qkv[2];

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

      attn_out = cs::compute::ScaledDotProductFlashAttention::forward(
          scheduler, attn_state, qview, kview, vview);

      attn_out = cs::compute::Utils::view(
          scheduler, attn_out,
          {config.batch_size, config.block_size, config.n_embd});
    }

    // Fc Proj
    return c_proj->forward(scheduler, attn_out);
  }

  cs::Tensor backward(const cs::Scheduler scheduler,
                      cs::Tensor dx) {
    // Fc Proj, we first run backward for input and then backward for parameter.
    auto dx_c_proj = c_proj->backward_input(scheduler, dx);
    c_proj->backward_parameter(scheduler, dx);

    cs::Tensor dx_attn;
    // Attn
    {
      dx_c_proj = cs::compute::Utils::view(
          scheduler, dx_c_proj,
          {config.batch_size, config.block_size, config.n_head,
           config.n_embd / config.n_head});

      auto [dq, dk, dv] = cs::compute::ScaledDotProductFlashAttention::backward(
          scheduler, attn_state, dx_c_proj);

      dq = cs::compute::Utils::view(
          scheduler, dq, {config.batch_size, config.block_size, config.n_embd});
      dk = cs::compute::Utils::view(
          scheduler, dk, {config.batch_size, config.block_size, config.n_embd});
      dv = cs::compute::Utils::view(
          scheduler, dv, {config.batch_size, config.block_size, config.n_embd});
      dx_attn = cs::compute::Utils::cat(scheduler, {dq, dk, dv}, -1);
    }

    // Fc Attn, we first run backward for input and then backward for parameter.
    auto dx_c_attn = c_attn->backward_input(scheduler, dx_attn);
    c_attn->backward_parameter(scheduler, dx_attn);
    return dx_c_attn;
  }
};

struct MLP : cs::module::Module {
  cs::Scheduler scheduler;
  const ModelConfig &config;
  cs::module::Linear fc1{nullptr}, fc2{nullptr};
  std::shared_ptr<cs::compute::GeLU::State> gelu_state;

  MLP(cs::Scheduler scheduler, const ModelConfig &config)
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

    _init_weights(scheduler, config);
  }

  void _init_weights(cs::Scheduler scheduler,
                     const ModelConfig &config) {
    cs::compute::Utils::normal_(scheduler, fc2->state()->forward.weight, 0.0,
                                0.02 / sqrt(2 * config.n_layer));
  }

  cs::Tensor forward(const cs::Scheduler &scheduler, cs::Tensor x) {
    auto fc1_out = fc1->forward(scheduler, x);
    auto gelu_out = cs::compute::GeLU::forward(scheduler, gelu_state, fc1_out);
    return fc2->forward(scheduler, gelu_out);
  }

  cs::Tensor backward(const cs::Scheduler scheduler, cs::Tensor dx) {
    // Fc2
    auto dx_fc2 = fc2->backward_input(scheduler, dx);
    fc2->backward_parameter(scheduler, dx);
    // GeLU
    auto dx_gelu = cs::compute::GeLU::backward(scheduler, gelu_state, dx_fc2);
    // Fc1
    auto dx_fc1 = fc1->backward_input(scheduler, dx_gelu);
    fc1->backward_parameter(scheduler, dx_gelu);
    return dx_fc1;
  }
};

struct Block : cs::module::Module {
  const ModelConfig &config;
  cs::Scheduler scheduler;
  cs::module::LayerNorm ln1{nullptr}, ln2{nullptr};
  std::shared_ptr<Attn> attn;
  std::shared_ptr<MLP> mlp;

  Block(cs::Scheduler scheduler, const ModelConfig &config)
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

    _init_weights(scheduler, config);
  }

  void _init_weights(cs::Scheduler scheduler,
                     const ModelConfig &config) {
  }

  cs::Tensor forward(const cs::Scheduler scheduler, cs::Tensor x) {
    auto ln1_out = ln1->forward(scheduler, x);
    auto attn_out = attn->forward(scheduler, ln1_out);
    auto res1_out = cs::compute::Utils::add(scheduler, x, attn_out);
    auto ln2_out = ln2->forward(scheduler, res1_out);
    auto mlp_out = mlp->forward(scheduler, ln2_out);
    return cs::compute::Utils::add(scheduler, res1_out, mlp_out);
  }

  cs::Tensor backward(const cs::Scheduler scheduler,
                      cs::Tensor dx) {
    auto dx_mlp = mlp->backward(scheduler, dx);
    auto dx_ln2 = ln2->backward(scheduler, dx_mlp);
    auto dx_res1 = cs::compute::Utils::add(scheduler, dx_ln2, dx);
    auto dx_attn = attn->backward(scheduler, dx_res1);
    auto dx_ln1 = ln1->backward(scheduler, dx_attn);
    auto dx_res2 = cs::compute::Utils::add(scheduler, dx_ln1, dx_res1);
    return dx_res2;
  }
};

struct GPT2 : cs::module::Module {
  const ModelConfig &config;
  cs::Scheduler scheduler;
  cs::module::Embedding wte{nullptr}, wpe{nullptr};
  std::vector<std::shared_ptr<Block>> h;
  cs::module::LayerNorm lnf{nullptr};
  cs::module::Linear lm_head{nullptr};
  cs::Tensor pos;

  GPT2(cs::Scheduler scheduler, const ModelConfig &config)
      : scheduler(scheduler), config{config} {
    pos = cs::compute::Utils::arange(
        scheduler, 0, config.block_size,
        torch::TensorOptions().dtype(torch::kInt64).device(config.device));
    wte = register_module(
        "wte",
        cs::module::Embedding(scheduler, cs::compute::Embedding::Options(
                                                config.pad_size, config.n_embd)
                                                .device(config.device)
                                                .dtype(config.dtype)));
    wpe = register_module(
        "wpe", cs::module::Embedding(scheduler,
                                        cs::compute::Embedding::Options(
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

    _init_weights(scheduler, config);
  }

  void _init_weights(cs::Scheduler scheduler,
                     const ModelConfig &config) {
    cs::compute::Utils::normal_(scheduler, wte->state()->forward.weight, 0.0,
                                0.02);
    cs::compute::Utils::normal_(scheduler, wpe->state()->forward.weight, 0.0,
                                0.02);
    cs::compute::Utils::normal_(scheduler, lm_head->state()->forward.weight,
                                0.0, 0.02);
    if (config.share_emb) { // share weight
      lm_head->state()->forward.weight = wte->state()->forward.weight;
    }
  }

  cs::Tensor forward(cs::Scheduler scheduler, const cs::Tensor input) {
    cs::Tensor emb_out;
    {
      auto wte_out = wte->forward(scheduler, input);
      auto wpe_out = wpe->forward(scheduler, pos);
      wpe_out =
          cs::compute::Utils::broadcast_to(scheduler, wpe_out, wte_out.sizes());
      emb_out = cs::compute::Utils::add(scheduler, wte_out, wpe_out);
    }
    auto block_out = emb_out;
    for (auto &block : h) {
      block_out = block->forward(scheduler, block_out);
    }
    auto lnf_out = lnf->forward(scheduler, block_out);
    auto lm_head_out = lm_head->forward(scheduler, lnf_out);
    return lm_head_out;
  }

  void backward(const cs::Scheduler scheduler, cs::Tensor dinput) {
    auto dx_lnf = lm_head->backward_input(scheduler, dinput);
    lm_head->backward_parameter(scheduler, dinput);
    auto dx_block = lnf->backward(scheduler, dx_lnf);
    for (int i = h.size() - 1; i >= 0; --i) { // reverse h in backward
        dx_block = h[i]->backward(scheduler, dx_block);
    }
    auto dx_emb = dx_block;
    {
      auto dx_wpe = cs::compute::Utils::sum(scheduler, dx_emb, 0);
      wpe->backward(scheduler, dx_wpe);
      wte->backward(scheduler, dx_emb);
    }
    if (config.share_emb) { // share weight
      lm_head->state()->forward.grad_weight = wte->state()->forward.grad_weight = 
        cs::compute::Utils::add(scheduler, lm_head->state()->forward.grad_weight, 
          wte->state()->forward.grad_weight);
    }
  }
};

void train() {
  ModelConfig modelConfig;
  TrainConfig trainConfig;
  DataConfig dataConfig;
  torch::manual_seed(trainConfig.seed);
  cs::DynamicScheduler scheduler{0};
  bool master_process = true;
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start,
      time_stop;
  std::chrono::duration<double> duration;
  std::ofstream logFile;
  std::shared_ptr<GPT2> model;
  cs::Tensor output, loss;
  const torch::TensorOptions data_option =
      torch::TensorOptions().dtype(torch::kInt64).device(modelConfig.device);
  const torch::TensorOptions act_option = torch::TensorOptions()
                                              .dtype(modelConfig.dtype)
                                              .device(modelConfig.device);
  if (master_process) {
    std::cout << "Prepare Dataset" << std::endl;
  }
  cs::data::LlmDataset dataset{{dataConfig.path}};
  cs::data::LlmDataLoader dataloader{dataset, modelConfig.batch_size,
                                     dataConfig.num_workers,
                                     dataConfig.shuffle};

  if (master_process) {
    std::cout << "Init" << std::endl;
    logFile.open(trainConfig.log_path);
    logFile << "Step,Loss\n";
  }
  model = std::make_shared<GPT2>(scheduler, modelConfig);

  auto loss_state = cs::compute::CrossEntropy::init(scheduler);

  cs::module::AdamW opt{scheduler, model,
                        cs::optimizer::AdamW::Options{}
                            .lr(trainConfig.max_lr)
                            .beta1(trainConfig.beta1)
                            .beta2(trainConfig.beta2)
                            .weight_decay(trainConfig.weight_decay)
                            .fused(false)};

  std::unordered_map<std::string, double> training_args =
      getTrainArgs(dataset.size(), modelConfig.block_size,
                   trainConfig.total_token_batch_size, modelConfig.batch_size,
                   1, trainConfig.max_steps, trainConfig.epoch);
  double epochs = training_args["epochs"];
  double max_steps = training_args["max_steps"];
  double grad_accum_steps = training_args["grad_accum_steps"];
  double total_tokens_per_step = training_args["total_tokens_per_step"];

  if (master_process) {
    std::cout << "The training process will train " << epochs << " epochs, "
              << max_steps << " steps." << std::endl;
    std::cout << "=> calculated gradient accumulation steps: "
              << grad_accum_steps << std::endl;
    std::cout << "=> calculated tokens per step: " << total_tokens_per_step
              << std::endl;
  }

  LRScheduler lr_scheduler(trainConfig.warmup_steps, trainConfig.max_lr,
                           (int)max_steps, trainConfig.min_lr);

  ProgressBar progressBar(max_steps);

  for (int step = 0; step < max_steps; ++step) {
    time_start = std::chrono::high_resolution_clock::now();
    progressBar.display(step, "Training: ", 0);

    // TODO: Add validation
    if ((step + 1) % trainConfig.val_every_steps == 0 && master_process) {
      // validation
    }

    // Training micro steps
    for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
      auto data = dataloader.load(scheduler);
      auto input = data["input_ids"];
      auto target = data["labels"];

      if (trainConfig.use_amp) {  // AMP: the forward parts should be written into this {}
        cs::autocast::ContextGuard guard{scheduler, trainConfig.amp_dtype};

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
      } else {
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
      }

      // Backward
      auto grad_output =
          cs::compute::CrossEntropy::backward(scheduler, loss_state);
      model->backward(scheduler, grad_output);

      // Wait in micro steps
      auto wait_step = step * (int)grad_accum_steps + micro_step;
      if (trainConfig.wait_every_step != -1 &&
          ((wait_step + 1) % trainConfig.wait_every_step == 0 ||
           (wait_step + 1) == max_steps * grad_accum_steps)) {
        model->wte->state()->forward.grad_weight.wait();
      }
    }

    // TODO: Add gradient clipping

    // Optimizer step
    opt->step(scheduler);
    opt->set_lr(lr_scheduler.get_lr(step));
    opt->zero_grad(scheduler);

    // Wait in steps
    if (trainConfig.wait_every_step != -1 &&
        ((step + 1) % trainConfig.wait_every_step == 0 ||
         step == max_steps - 1)) {
      model->lm_head->state()->forward.weight.wait();
    }

    // Check
    time_stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(
        time_stop - time_start);
    if (trainConfig.check_every_steps != -1 &&
        (step + 1) % trainConfig.check_every_steps == 0 && master_process) {
      printf(  // Noting: add \n to the beginning of the string for single
               // device
          "\nstep %5d | lr %.4e | grad norm: %.4f | dt: %.2fs | tok/sec: "
          "%.2f\n",
          step + 1, opt->get_lr(), 1.0, duration.count(),
          total_tokens_per_step / (duration.count()));
      std::cout << "loss: " << loss << std::endl;
      logFile << step + 1 << "," << loss << "\n";
    }
  }
  if (master_process) {
    logFile.close();
  }
}

int main() {
  train();
  return 0;
}
