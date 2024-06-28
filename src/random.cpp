/*
 * Copyright (c) 2024 The Core Team
 *
 * Licensed under the Apache License, Version 2.0;
 * You may not use this file except in compliance with the License.
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

#include "random.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

namespace cs {
at::Generator &getCUDAGenerator() {
  static at::Generator generator = at::cuda::detail::getDefaultCUDAGenerator();
  return generator;
}

at::Generator &getCPUGenerator() {
  static at::Generator generator = at::detail::getDefaultCPUGenerator();
  return generator;
}

void manual_seed(const int64_t seed) {
  at::manual_seed(seed);
  getCUDAGenerator().set_current_seed(seed);
  getCUDAGenerator().set_offset(0);
  getCPUGenerator().set_current_seed(seed);
}
}  // namespace cs
