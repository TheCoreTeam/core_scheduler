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

if ! command -v shfmt &>/dev/null; then
	echo "shfmt could not be found, please install it."
	exit 1
fi

# 格式化C++和CUDA文件，确保跳过指定目录，并确保每个文件以恰好一个空行结束
find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -i {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;

# TODO(Jie): cmake?
# 格式化CMake文件，确保跳过指定目录，并确保每个文件以恰好一个空行结束
# find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -name '*.cmake' -exec cmake-format -i {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;
# find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -name 'CMakeLists.txt' -exec cmake-format -i {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;

# 格式化Shell脚本文件，并确保每个文件以恰好一个空行结束
find . -type f -not \( -path './third_party/*' -o -path '*/cmake-build*' -o -path '*/build/*' \) -name '*.sh' -exec shfmt -w {} \; -exec sh -c "sed -i -e :a -e '/^\n*$/N;/\n$/ba' -e '\$a\\' {}" \;

echo "All files have been formatted and adjusted for trailing newlines."
