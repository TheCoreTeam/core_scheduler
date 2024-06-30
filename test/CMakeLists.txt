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

add_executable(core_scheduler_tests
        main.cpp
        test_adamw.cpp
        test_amp_adamw.cpp
        test_linear.cpp
        test_gelu.cpp
        test_flash_attention.cpp
        test_embedding.cpp
        test_cross_entropy.cpp
        test_utils.cpp
        test_layer_norm.cpp
        test_load_save.cpp
        test_cudnn_attention.cpp
        test_gelu_linear.cpp
)

target_link_libraries(core_scheduler_tests PRIVATE
        core_scheduler
        GTest::gtest
        fmt::fmt
)

target_compile_options(core_scheduler_tests PRIVATE
        $<$<CONFIG:Release>:-Ofast>
        $<$<CONFIG:Debug>:-O0>
)

add_executable(core_scheduler_communication_tests
        main_communication.cpp
        test_all_reduce.cpp
        test_all_gather.cpp
        test_all_to_all.cpp
        test_reduce_scatter.cpp
)

target_include_directories(core_scheduler_communication_tests PRIVATE
        ${TORCH_INCLUDE_PATH}
)

target_link_libraries(core_scheduler_communication_tests PRIVATE
        core_scheduler
        GTest::gtest
        MPI::MPI_C
        fmt::fmt
)

target_compile_options(core_scheduler_communication_tests PRIVATE
        $<$<CONFIG:Release>:-Ofast>
        $<$<CONFIG:Debug>:-O0>
)
