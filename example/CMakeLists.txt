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

function(create_executable target_name source_files)
    add_executable(${target_name} ${source_files})
    target_link_libraries(${target_name} PRIVATE
            dllm
    )
    target_compile_options(${target_name} PRIVATE
            $<$<CONFIG:Release>:-Ofast>
            $<$<CONFIG:Debug>:-O0>
    )
endfunction()

create_executable(GPT2SingleDevice gpt2_single_device.cpp)

create_executable(GPT2DDP gpt2_ddp.cpp)