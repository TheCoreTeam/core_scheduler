# Copyright (c) 2024 The Core Team
#
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CORE_SCHEDULER_COMPUTE_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cross_entropy.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gelu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/embedding.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_embedding.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_flash_attn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sdpa_cudnn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_layer_norm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_gelu_linear_cudnn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gelu_linear_cudnn.cpp
)
set(CORE_SCHEDULER_COMPUTE_SRCS ${CORE_SCHEDULER_COMPUTE_SRCS} PARENT_SCOPE)
