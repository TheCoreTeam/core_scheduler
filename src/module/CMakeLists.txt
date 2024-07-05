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

set(CORE_SCHEDULER_MODULE_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gelu_linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_gelu_linear.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_layer_norm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/embedding.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/amp_embedding.cpp
)
set(CORE_SCHEDULER_MODULE_SRCS ${CORE_SCHEDULER_MODULE_SRCS} PARENT_SCOPE)
