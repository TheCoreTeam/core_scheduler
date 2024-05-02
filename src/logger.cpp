#include "logger.h"

#include <spdlog/sinks/stdout_color_sinks.h>

#include <memory>

namespace dllm {

spdlog::logger &logger() {
  static spdlog::logger _logger{
      "DLLM.cpp", std::make_shared<spdlog::sinks::stdout_color_sink_mt>()};
  return _logger;
}
}  // namespace dllm
