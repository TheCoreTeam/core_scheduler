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

#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>

#include "logger.h"

namespace cs {
cudnnHandle_t getCurrentCuDnnHandle() {
  thread_local struct HandleImpl {
    cudnnHandle_t handle{nullptr};
    HandleImpl() {
      CS_CHECK_CUDNN(cudnnCreate(&handle));
      CS_CHECK_CUDNN(
          cudnnSetStream(handle, c10::cuda::getCurrentCUDAStream().stream()));
    }

    ~HandleImpl() { CS_CHECK_CUDNN(cudnnDestroy(handle)); }
  } handle{};
  return handle.handle;
}
}  // namespace cs
