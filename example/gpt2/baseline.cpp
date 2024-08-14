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


struct ModelConfig {
  int64_t batch_size = 8;
  const int64_t block_size = 1024;
  const int64_t vocab_size = 50257;
  const int64_t pad_size = 50304;  // pad vocab_size to be more efficient
  const int64_t n_embd = 768;      // 2048
  const int64_t n_head = 12;       // 32
  const int64_t n_layer = 12;      // 22
  const bool use_bias = false;
  const double dropout = 0.0;
  const double epsilon = 1e-5;
  torch::Device device = torch::kCUDA;
  torch::Dtype dtype = torch::kBFloat16;  // model percision
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
  bool amp = true;
  torch::Dtype amp_dtype = torch::kBFloat16;  // amp percision: torch::kBFloat16, torch::kFloat16 
                                              // (Now we only support bf16)
  int64_t seed = 1337;
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


// ------------------------- Model ------------------------- //

torch::Tensor standard_attention(const torch::Tensor q, const torch::Tensor k,
                                 const torch::Tensor v,
                                 const torch::Tensor bias, const int B,
                                 const int T, const int C,
                                 const float dropout_prob,
                                 const bool is_training) {
  auto bias_device = bias.to(q.device());
  auto att =
      torch::matmul(q, k.transpose(-2, -1)) * (1.0 / std::sqrt(k.size(-1)));
  att = att.masked_fill(bias_device.slice(/*dim=*/2, /*start=*/0, /*end=*/T)
                                .slice(
                                    /*dim=*/3, /*start=*/0, /*end=*/T) == 0,
                        -std::numeric_limits<float>::infinity());
  att = torch::softmax(att, /*dim=*/-1);
  att = torch::nn::functional::dropout(
      att, torch::nn::functional::DropoutFuncOptions()
               .p(dropout_prob)
               .training(is_training));
  auto y = torch::matmul(
      att, v);  // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
  y = y.transpose(1, 2).contiguous().view({B, T, C});
  return y;
}

struct AttnTorch : torch::nn::Module {
  bool training = true;
  torch::Tensor c_attn_out, attn_out, c_proj_out;
  torch::nn::Linear c_attn{nullptr}, c_proj{nullptr};
  torch::Tensor bias;

  AttnTorch(const ModelConfig& config) {
    c_attn = register_module(
        "c_attn", torch::nn::Linear(
                      torch::nn::LinearOptions(config.n_embd, 3*config.n_embd)
                          .bias(config.use_bias)));
    // c_attn = register_module(
    //     "c_attn", torch::nn::Linear(
    //                   torch::nn::LinearOptions(config.n_embd, config.n_embd)
    //                       .bias(config.use_bias)));
    
    c_proj = register_module(
        "c_proj",
        torch::nn::Linear(torch::nn::LinearOptions(config.n_embd, config.n_embd)
                              .bias(config.use_bias)));
    bias = torch::tril(torch::ones({config.block_size, config.block_size}))
               .view({1, 1, config.block_size, config.block_size});
  
    _init_weights(config);
  }

  void _init_weights(const ModelConfig &config) {
    torch::nn::init::normal_(c_attn->weight, 0.0,
                                0.02 / sqrt(2 * config.n_layer));
    torch::nn::init::normal_(c_proj->weight, 0.0,
                                0.02 / sqrt(2 * config.n_layer));
  }

  // Forward pass
  torch::Tensor forward(torch::Tensor x, const ModelConfig& config) {
    x = c_attn->forward(x);
    c_attn_out = x;

    auto B = config.batch_size;  // Batch size
    auto T = config.block_size;  // Sequence length
    auto C = config.n_embd;      // Embedding dimensionality (n_embd)
    auto outputs = x.split(config.n_embd, 2);
    auto& q = outputs[0];
    auto& k = outputs[1];
    auto& v = outputs[2];
    k = k.view({B, T, config.n_head, C / config.n_head})
            .transpose(1, 2);  // (B, nh, T, hs)
    q = q.view({B, T, config.n_head, C / config.n_head})
            .transpose(1, 2);  // (B, nh, T, hs)
    v = v.view({B, T, config.n_head, C / config.n_head})
            .transpose(1, 2);  // (B, nh, T, hs)
    x = standard_attention(q, k, v, bias, B, T, C, config.dropout, training);
    attn_out = x;

    x = c_proj->forward(x);
    c_proj_out = x;
    x = torch::nn::functional::dropout(
        x, torch::nn::functional::DropoutFuncOptions()
               .p(config.dropout)
               .training(training));
    return x;
  }
};

struct MLPTorch : torch::nn::Module {
  torch::Tensor fc1_out, gelu_out, fc2_out;
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};

  MLPTorch(const ModelConfig& config) {
    fc1 = register_module(
        "fc1", torch::nn::Linear(
                   torch::nn::LinearOptions(config.n_embd, 4 * config.n_embd)
                       .bias(config.use_bias)));
    fc2 = register_module(
        "fc2", torch::nn::Linear(
                   torch::nn::LinearOptions(4 * config.n_embd, config.n_embd)
                       .bias(config.use_bias)));
  
    _init_weights(config);
  }

  void _init_weights(const ModelConfig &config) {
    torch::nn::init::normal_(fc1->weight, 0.0,
                                0.02 / sqrt(2 * config.n_layer));
    torch::nn::init::normal_(fc2->weight, 0.0,
                                0.02 / sqrt(2 * config.n_layer));
  }


  // Forward pass
  torch::Tensor forward(torch::Tensor x, const ModelConfig& config) {
    x = fc1->forward(x);
    fc1_out = x;
    x = torch::gelu(x);
    gelu_out = x;
    x = fc2->forward(x);
    fc2_out = x;
    return x;
  }
};

struct BlockTorch : torch::nn::Module {
  torch::Tensor ln1_out, res1_out, ln2_out, res2_out;
  bool training = true;
  ModelConfig config;
  std::shared_ptr<AttnTorch> attn;
  std::shared_ptr<MLPTorch> mlp;
  torch::nn::LayerNorm ln_1{nullptr}, ln_2{nullptr};

  BlockTorch(const ModelConfig& config) {
    ln_1 = register_module(
        "ln_1", torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                         std::vector<int64_t>{config.n_embd})
                                         .eps(config.epsilon)
                                         .elementwise_affine(true)));
    attn = register_module("attn", std::make_shared<AttnTorch>(config));
    ln_2 = register_module(
        "ln_2", torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                         std::vector<int64_t>{config.n_embd})
                                         .eps(config.epsilon)
                                         .elementwise_affine(true)));
    mlp = register_module("mlp", std::make_shared<MLPTorch>(config));
  }

  // Forward pass
  torch::Tensor forward(torch::Tensor x, const ModelConfig& config) {
    auto output1 = ln_1->forward(x);
    ln1_out = output1;
    output1 = attn->forward(output1, config);
    output1 = x + output1;
    res1_out = output1;
    auto output2 = ln_2->forward(output1);
    ln2_out = output2;
    output2 = mlp->forward(output2, config);
    output2 = output1 + output2;
    res2_out = output2;
    return output2;
  }
};

struct GPT2Torch : torch::nn::Module {
  bool training = true;
  torch::Tensor wte_out, wpe_out, emb_out, lnf_out, lm_head_out;
  torch::nn::Embedding wte{nullptr}, wpe{nullptr};
  torch::nn::ModuleList h;
  torch::nn::LayerNorm ln_f{nullptr};
  torch::nn::Linear lm_head{nullptr};

  GPT2Torch(const ModelConfig& config) {
    wte =
        register_module("wte", torch::nn::Embedding(torch::nn::EmbeddingOptions(
                                   config.pad_size, config.n_embd)));
    wpe =
        register_module("wpe", torch::nn::Embedding(torch::nn::EmbeddingOptions(
                                   config.block_size, config.n_embd)));
    for (int layer = 0; layer < config.n_layer; ++layer) {
      // Adding layers to the module list
      h->push_back(register_module("h" + std::to_string(layer),
                                   std::make_shared<BlockTorch>(config)));
    }
    ln_f = register_module(
        "ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions(
                                         std::vector<int64_t>{config.n_embd})
                                         .eps(config.epsilon)
                                         .elementwise_affine(true)));
    lm_head = register_module(
        "lm_head", torch::nn::Linear(
                       torch::nn::LinearOptions(config.n_embd, config.pad_size)
                           .bias(config.use_bias)));
  
    _init_weights(config);
  }

  void _init_weights(const ModelConfig &config) {
    torch::nn::init::normal_(wte->weight, 0.0,
                                0.02);
    torch::nn::init::normal_(wpe->weight, 0.0,
                                0.02);
    torch::nn::init::normal_(lm_head->weight,
                                0.0, 0.02);
    // lm_head->weight = wte->weight; // share weight
  }

  // Forward pass
  torch::Tensor forward(torch::Tensor input_ids, const ModelConfig& config) {
    auto tok_emb = wte->forward(input_ids);
    wte_out = tok_emb;
    auto pos = torch::arange(0, config.block_size, input_ids.options());
    auto pos_emb = wpe->forward(pos);
    wpe_out = pos_emb;
    auto x = tok_emb + pos_emb;
    emb_out = x;
    x = torch::nn::functional::dropout(
        x, torch::nn::functional::DropoutFuncOptions()
               .p(config.dropout)
               .training(training));
    for (auto& block : *h) {
      x = block->as<BlockTorch>()->forward(x, config);
    }
    x = ln_f->forward(x);
    lnf_out = x;
    x = lm_head->forward(x);
    lm_head_out = x;
    return x;
  }
};


void train() {
  ModelConfig modelConfig;
  TrainConfig trainConfig;
  DataConfig dataConfig;
  torch::manual_seed(trainConfig.seed);
  bool master_process = true;
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start,
      time_stop;
  std::chrono::duration<double> duration;
  std::ofstream logFile;
  std::shared_ptr<GPT2Torch> model;
  torch::Tensor loss;
  const torch::TensorOptions data_option =
      torch::TensorOptions().dtype(torch::kInt64).device(modelConfig.device);
  const torch::TensorOptions act_option = torch::TensorOptions()
                                              .dtype(modelConfig.dtype)
                                              .device(modelConfig.device);
  if (master_process) {
    std::cout << "Prepare Dataset" << std::endl;
  }

  if (master_process) {
    std::cout << "Init" << std::endl;
    logFile.open(trainConfig.log_path);
    logFile << "Step,Loss\n";
  }
  model = std::make_shared<GPT2Torch>(modelConfig);
  model->to(modelConfig.device, modelConfig.dtype);

  torch::optim::AdamW opt{model->parameters(),
                            torch::optim::AdamWOptions{}
                                .lr(trainConfig.max_lr)
                                .betas({trainConfig.beta1, trainConfig.beta2})
                                .eps(trainConfig.eps)
                                .weight_decay(trainConfig.weight_decay)};

  std::unordered_map<std::string, double> training_args =
      getTrainArgs(11778*512, modelConfig.block_size,
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

  // Random generate data
  auto input = torch::randint(
      0, modelConfig.vocab_size, 
      {modelConfig.batch_size, modelConfig.block_size}, data_option);
  auto target = torch::randint(
      0, modelConfig.vocab_size, 
      {modelConfig.batch_size, modelConfig.block_size}, data_option);
  for (int step = 0; step < max_steps; ++step) {
    time_start = std::chrono::high_resolution_clock::now();
    progressBar.display(step, "Training: ", 0);

    // TODO: Add validation
    if ((step + 1) % trainConfig.val_every_steps == 0 && master_process) {
      // validation
    }

    // Training micro steps
    for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
      // auto data = dataloader.load(scheduler);
      // auto input = data["input_ids"];
      // auto target = data["labels"];

      { // AMP is not supported in libtorch
        // Forward
        torch::Tensor output;
        output = model->forward(input, modelConfig);

        // compute loss
        {
        output = output.view(
            {modelConfig.batch_size * modelConfig.block_size, modelConfig.pad_size});
        target = target.view(
            {modelConfig.batch_size * modelConfig.block_size});
        loss = at::cross_entropy_loss(output, target);
        }
      }
      // Backward
      loss.backward();
    }

    // TODO: Add gradient clipping

    // Optimizer step
    opt.step();
    auto lr = lr_scheduler.get_lr(step);
    for (auto& group : opt.param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
    }
    opt.zero_grad();

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
          step + 1, lr, 1.0, duration.count(),
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