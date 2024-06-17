# Copyright (c) 2024 The Core team
#
# Licensed under the Apache License, Version 2.0
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Ensure that clang-format, cmake-format, and shfmt are installed
if ! command -v clang-format &>/dev/null; then
	echo "clang-format could not be found, please install it."
	exit 1
fi

# TODO(Jie): cmake?
#if ! command -v cmake-format &>/dev/null; then
#	echo "cmake-format could not be found, please install it."
#	exit 1
#fi

# 格式化Shell脚本文件，并确保每个文件以恰好一个空行结束
if command -v shfmt &>/dev/null; then
	find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -name '*.sh' -exec shfmt -w {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;
fi

# 格式化C++和CUDA文件，确保跳过指定目录，并确保每个文件以恰好一个空行结束
find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;

# TODO(Jie): cmake?
# 格式化CMake文件，确保跳过指定目录，并确保每个文件以恰好一个空行结束
# find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -name '*.cmake' -exec cmake-format -i {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;
# find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -name 'CMakeLists.txt' -exec cmake-format -i {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;

echo "All files have been formatted and adjusted for trailing newlines."
