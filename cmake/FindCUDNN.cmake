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

# Distributed under the OSI-approved BSD 3-Clause License.

#.rst:
# FindCUDNN
# --------
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  ``CUDNN_FOUND``
#    True if CUDNN found on the local system
#
#  ``CUDNN_INCLUDE_DIRS``
#    Location of CUDNN header files.
#
#  ``CUDNN_LIBRARIES``
#    The CUDNN libraries.
#
#  ``nvidia::cudnn``
#    The CUDNN target
#

include(FindPackageHandleStandardArgs)

function(system)
    set(options STRIP)
    set(oneValueArgs OUTPUT_VARIABLE ERROR_VARIABLE WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)
    cmake_parse_arguments(
            SYSTEM
            "${options}"
            "${oneValueArgs}"
            "${multiValueArgs}"
            "${ARGN}"
    )

    if (NOT DEFINED SYSTEM_WORKING_DIRECTORY)
        set(SYSTEM_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()

    execute_process(
            COMMAND ${SYSTEM_COMMAND}
            OUTPUT_VARIABLE STDOUT
            ERROR_VARIABLE STDERR
            WORKING_DIRECTORY "${SYSTEM_WORKING_DIRECTORY}"
    )

    if ("${SYSTEM_STRIP}")
        string(STRIP "${STDOUT}" STDOUT)
        string(STRIP "${STDERR}" STDERR)
    endif ()

    set("${SYSTEM_OUTPUT_VARIABLE}" "${STDOUT}" PARENT_SCOPE)

    if (DEFINED SYSTEM_ERROR_VARIABLE)
        set("${SYSTEM_ERROR_VARIABLE}" "${STDERR}" PARENT_SCOPE)
    endif ()
endfunction()

if (NOT DEFINED PYTHON_EXECUTABLE)
    if (WIN32)
        set(PYTHON_EXECUTABLE "python.exe")
    else ()
        set(PYTHON_EXECUTABLE "python")
    endif ()
endif ()

if (UNIX)
    system(
            STRIP OUTPUT_VARIABLE PYTHON_EXECUTABLE
            COMMAND bash -c "type -P '${PYTHON_EXECUTABLE}'"
    )
endif ()

system(
        STRIP OUTPUT_VARIABLE PYTHON_VERSION
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('platform').python_version())"
)

system(
        STRIP OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        COMMAND "${PYTHON_EXECUTABLE}" -c "print(' '.join(__import__('site').getsitepackages()))"
)

message(STATUS "Use Python version: ${PYTHON_VERSION}")
message(STATUS "Use Python executable: \"${PYTHON_EXECUTABLE}\"")
message(STATUS "Python site-packages directory: ${PYTHON_SITE_PACKAGES}")

if (DEFINED ENV{CONDA_PREFIX})
    list(APPEND PIP_CUDNN_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/nvidia/cudnn/include)
    list(APPEND PIP_CUDNN_LIB_DIR ${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib)
    list(APPEND PIP_CUDNN_FRONTEND_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/include)
endif ()

find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h cudnn_v8.h cudnn_v7.h
        HINTS ${PIP_CUDNN_INCLUDE_DIR} ${CUDA_TOOLKIT_ROOT} $ENV{CUDA_PATH} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN} $ENV{CUDNN_ROOT_DIR} $ENV{CUDA_PATH}/../../../NVIDIA/CUDNN/v9.0 /usr/include /usr/include/x86_64-linux-gnu/ /usr/include/aarch64-linux-gnu/
        PATH_SUFFIXES cuda/include include include/12.3)

find_path(CUDNN_FRONTEND_INCLUDE_DIR NAMES cudnn_frontend.h
        HINTS ${PIP_CUDNN_FRONTEND_INCLUDE_DIR} ${CUDA_TOOLKIT_ROOT} $ENV{CUDA_PATH} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN} $ENV{CUDNN_ROOT_DIR} $ENV{CUDA_PATH}/../../../NVIDIA/CUDNN/v9.0 /usr/include /usr/include/x86_64-linux-gnu/ /usr/include/aarch64-linux-gnu/
        PATH_SUFFIXES cuda/include include include/12.3)

# 初始设置查找路径和后缀
set(CUDNN_HINTS ${PIP_CUDNN_LIB_DIR} ${CUDA_TOOLKIT_ROOT} $ENV{CUDA_PATH} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN} $ENV{CUDNN_ROOT_DIR} $ENV{CUDA_PATH}/../../../NVIDIA/CUDNN/v9.0 /usr/lib/x86_64-linux-gnu/ /usr/include/aarch64-linux-gnu/ /usr/)
set(CUDNN_PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64 cuda/lib/x64 lib/12.3/x64)

# 查找各个库并追加到 CUDNN_LIBRARIES 列表中
find_library(CUDNN_MAIN_LIBRARY NAMES libcudnn.so.9
        HINTS ${CUDNN_HINTS}
        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
list(APPEND CUDNN_LIBRARIES ${CUDNN_MAIN_LIBRARY})

#find_library(CUDNN_ADV_LIBRARY NAMES libcudnn_adv.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_ADV_LIBRARY})
#
#find_library(CUDNN_CNN_LIBRARY NAMES libcudnn_cnn.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_CNN_LIBRARY})
#
#find_library(CUDNN_ENGINES_PRECOMPILED_LIBRARY NAMES libcudnn_engines_precompiled.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_ENGINES_PRECOMPILED_LIBRARY})
#
#find_library(CUDNN_ENGINES_RUNTIME_COMPILED_LIBRARY NAMES libcudnn_engines_runtime_compiled.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_ENGINES_RUNTIME_COMPILED_LIBRARY})
#
#find_library(CUDNN_GRAPH_LIBRARY NAMES libcudnn_graph.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_GRAPH_LIBRARY})
#
#find_library(CUDNN_HEURISTIC_LIBRARY NAMES libcudnn_heuristic.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_HEURISTIC_LIBRARY})
#
#find_library(CUDNN_OPS_LIBRARY NAMES libcudnn_ops.so.9
#        HINTS ${CUDNN_HINTS}
#        PATH_SUFFIXES ${CUDNN_PATH_SUFFIXES})
#list(APPEND CUDNN_LIBRARIES ${CUDNN_OPS_LIBRARY})

# 确保所有库都已找到
foreach (lib ${CUDNN_LIBRARIES})
    if (NOT lib)
        message(FATAL_ERROR "A required cudnn library is missing: ${lib}")
    endif ()
endforeach ()

# 输出所有找到的库，用于调试
message(STATUS "Found CUDA DNN Libraries: ${CUDNN_LIBRARIES}")
set(CUDNN_LIBRARY ${CUDNN_LIBRARIES})

if (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_HEADER_CONTENTS)
elseif (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_v8.h")
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn_v8.h CUDNN_HEADER_CONTENTS)
elseif (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_v7.h")
    file(READ ${CUDNN_INCLUDE_DIR}/cudnn_v7.h CUDNN_HEADER_CONTENTS)
endif ()
if (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" CUDNN_VERSION_H_CONTENTS)
    string(APPEND CUDNN_HEADER_CONTENTS "${CUDNN_VERSION_H_CONTENTS}")
    unset(CUDNN_VERSION_H_CONTENTS)
elseif (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version_v8.h")
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version_v8.h" CUDNN_VERSION_H_CONTENTS)
    string(APPEND CUDNN_HEADER_CONTENTS "${CUDNN_VERSION_H_CONTENTS}")
    unset(CUDNN_VERSION_H_CONTENTS)
elseif (EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version_v7.h")
    file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version_v7.h" CUDNN_VERSION_H_CONTENTS)
    string(APPEND CUDNN_HEADER_CONTENTS "${CUDNN_VERSION_H_CONTENTS}")
    unset(CUDNN_VERSION_H_CONTENTS)
endif ()
if (CUDNN_HEADER_CONTENTS)
    string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
            _CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
            _CUDNN_VERSION_MAJOR "${_CUDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
            _CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
            _CUDNN_VERSION_MINOR "${_CUDNN_VERSION_MINOR}")
    string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
            _CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
            _CUDNN_VERSION_PATCH "${_CUDNN_VERSION_PATCH}")
    if (NOT _CUDNN_VERSION_MAJOR)
        set(_CUDNN_VERSION "?")
    else ()
        set(_CUDNN_VERSION "${_CUDNN_VERSION_MAJOR}.${_CUDNN_VERSION_MINOR}.${_CUDNN_VERSION_PATCH}")
    endif ()
endif ()

set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
mark_as_advanced(CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

find_package_handle_standard_args(CUDNN
        REQUIRED_VARS CUDNN_INCLUDE_DIR CUDNN_LIBRARY
        VERSION_VAR CUDNN_VERSION
)

if (WIN32)
    set(CUDNN_DLL_DIR ${CUDNN_INCLUDE_DIR})
    list(TRANSFORM CUDNN_DLL_DIR APPEND "/../bin")
    find_file(CUDNN_LIBRARY_DLL NAMES cudnn64_${CUDNN_VERSION_MAJOR}.dll PATHS ${CUDNN_DLL_DIR})
endif ()

if (CUDNN_FOUND AND NOT TARGET nvidia::cudnn)
    if (EXISTS "${CUDNN_LIBRARY_DLL}")
        add_library(nvidia::cudnn SHARED IMPORTED)
        set_target_properties(nvidia::cudnn PROPERTIES
                IMPORTED_LOCATION "${CUDNN_LIBRARY_DLL}"
                IMPORTED_IMPLIB "${CUDNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "C")
    else ()
        add_library(nvidia::cudnn UNKNOWN IMPORTED)
        set_target_properties(nvidia::cudnn PROPERTIES
                IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "C")
    endif ()
endif ()

if (CUDNN_FOUND AND NOT TARGET nvidia::cudnn_frontend)
        add_library(nvidia::cudnn_frontend UNKNOWN IMPORTED)
        set_target_properties(nvidia::cudnn_frontend PROPERTIES
                IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_FRONTEND_INCLUDE_DIR}"
                IMPORTED_LINK_INTERFACE_LANGUAGES "C")
        target_compile_definitions(nvidia::cudnn_frontend INTERFACE
                $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>)
        target_link_libraries(
                nvidia::cudnn_frontend INTERFACE
                CUDA::cudart
                CUDA::nvrtc)
endif ()
