#pragma once
#include <memory>

namespace dllm {
struct Event {
  struct Impl;

  Event();

  explicit Event(std::shared_ptr<Impl> impl);

  ~Event();

  void block() const;

  void record() const;

  [[nodiscard]] bool query() const;

  void synchronize() const;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

 private:
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm