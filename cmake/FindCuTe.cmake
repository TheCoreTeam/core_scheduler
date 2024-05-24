# Distributed under the OSI-approved BSD 3-Clause License.

#.rst:
# FindCUTE
# --------
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  ``CUTE_FOUND``
#    True if CUTE found on the local system
#
#  ``CUTE_INCLUDE_DIRS``
#    Location of CUTE header files.
#
#  ``CUTE_LIBRARIES``
#    The CUTE libraries.
#
#  ``nvidia::cutlass::cute``
#    The CUTE target
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
    list(APPEND PIP_CUTE_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/cutlass_library/source/include/cute)
    list(APPEND PIP_CUTLASS_INCLUDE_DIR ${PYTHON_SITE_PACKAGES}/cutlass_library/source/include/cutlass)
endif ()

find_path(CUTE_INCLUDE_DIR NAMES tensor.hpp
        HINTS ${PIP_CUTE_INCLUDE_DIR} ${CUDA_TOOLKIT_ROOT} $ENV{CUDA_PATH} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUTE} $ENV{CUTE_ROOT_DIR} $ENV{CUDA_PATH}/../../../NVIDIA/CUTE/v9.0 /usr/include /usr/include/x86_64-linux-gnu/ /usr/include/aarch64-linux-gnu/
        PATH_SUFFIXES cuda/include include include/12.3)

find_path(CUTLASS_INCLUDE_DIR NAMES cutlass.h
        HINTS ${PIP_CUTLASS_INCLUDE_DIR} ${CUDA_TOOLKIT_ROOT} $ENV{CUDA_PATH} $ENV{CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUTE} $ENV{CUTE_ROOT_DIR} $ENV{CUDA_PATH}/../../../NVIDIA/CUTE/v9.0 /usr/include /usr/include/x86_64-linux-gnu/ /usr/include/aarch64-linux-gnu/
        PATH_SUFFIXES cuda/include include include/12.3)

if (EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass.h")
    file(READ ${CUTLASS_INCLUDE_DIR}/cutlass.h CUTLASS_HEADER_CONTENTS)
endif ()
if (EXISTS "${CUTLASS_INCLUDE_DIR}/version.h")
    file(READ "${CUTLASS_INCLUDE_DIR}/version.h" CUTLASS_VERSION_H_CONTENTS)
    string(APPEND CUTLASS_HEADER_CONTENTS "${CUTLASS_VERSION_H_CONTENTS}")
    unset(CUTLASS_VERSION_H_CONTENTS)
endif ()
if (CUTLASS_HEADER_CONTENTS)
    string(REGEX MATCH "define CUTLASS_MAJOR * +([0-9]+)"
            _CUTE_VERSION_MAJOR "${CUTLASS_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUTLASS_MAJOR * +([0-9]+)" "\\1"
            _CUTE_VERSION_MAJOR "${_CUTE_VERSION_MAJOR}")
    string(REGEX MATCH "define CUTLASS_MINOR * +([0-9]+)"
            _CUTE_VERSION_MINOR "${CUTLASS_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUTLASS_MINOR * +([0-9]+)" "\\1"
            _CUTE_VERSION_MINOR "${_CUTE_VERSION_MINOR}")
    string(REGEX MATCH "define CUTLASS_PATCH * +([0-9]+)"
            _CUTE_VERSION_PATCH "${CUTLASS_HEADER_CONTENTS}")
    string(REGEX REPLACE "define CUTLASS_PATCH * +([0-9]+)" "\\1"
            _CUTE_VERSION_PATCH "${_CUTE_VERSION_PATCH}")
    if (NOT _CUTE_VERSION_MAJOR)
        set(_CUTE_VERSION "?")
    else ()
        set(_CUTE_VERSION "${_CUTE_VERSION_MAJOR}.${_CUTE_VERSION_MINOR}.${_CUTE_VERSION_PATCH}")
    endif ()
endif ()

set(CUTE_INCLUDE_DIRS ${CUTE_INCLUDE_DIR})
mark_as_advanced(CUTE_INCLUDE_DIR)

find_package_handle_standard_args(CuTe
        REQUIRED_VARS CUTE_INCLUDE_DIR
        VERSION_VAR CUTE_VERSION
)

if (CUTE_FOUND AND NOT TARGET nvidia::cutlass::cute)
    add_library(nvidia::cutlass::cute INTERFACE IMPORTED)
    target_include_directories(nvidia::cutlass::cute INTERFACE ${CUTE_INCLUDE_DIR}/../)
endif ()
