#include "logger.h"
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace dllm {

spdlog::logger &logger() {
  static spdlog::logger _logger{
      "DLLM.cpp", std::make_shared<spdlog::sinks::stdout_color_sink_mt>()};
  return _logger;
}
} // namespace dllm
