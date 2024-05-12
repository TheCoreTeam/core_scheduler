if (DLLM_ENABLE_FLASH_ATTENTION)
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

    message(STATUS "Use Python version: ${PYTHON_VERSION}")
    message(STATUS "Use Python executable: \"${PYTHON_EXECUTABLE}\"")

    if (NOT DEFINED PYTHON_INCLUDE_DIR)
        message(STATUS "Auto detecting Python include directory...")
        system(
                STRIP OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('platinclude'))"
        )
    endif ()

    if ("${PYTHON_INCLUDE_DIR}" STREQUAL "")
        message(FATAL_ERROR "Python include directory not found")
    else ()
        message(STATUS "Detected Python include directory: \"${PYTHON_INCLUDE_DIR}\"")
        include_directories("${PYTHON_INCLUDE_DIR}")
    endif ()

    system(
            STRIP OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            COMMAND "${PYTHON_EXECUTABLE}" -c "print(__import__('sysconfig').get_path('purelib'))"
    )
    message(STATUS "Detected Python site packages: \"${PYTHON_SITE_PACKAGES}\"")

    if (NOT DEFINED TORCH_INCLUDE_PATH)
        message(STATUS "Auto detecting Torch include directory...")
        system(
                STRIP OUTPUT_VARIABLE TORCH_INCLUDE_PATH
                COMMAND "${PYTHON_EXECUTABLE}" -c "print('\\\;'.join(__import__('torch.utils.cpp_extension', fromlist=[None]).include_paths()))"
        )

        if ("${TORCH_INCLUDE_PATH}" STREQUAL "")
            set(TORCH_INCLUDE_PATH "${PYTHON_SITE_PACKAGES}/torch/include")
        endif ()
    endif ()

    if ("${TORCH_INCLUDE_PATH}" STREQUAL "")
        message(FATAL_ERROR "Torch include directory not found. Got: \"${TORCH_INCLUDE_PATH}\"")
    else ()
        message(STATUS "Detected Torch include directory: \"${TORCH_INCLUDE_PATH}\"")
        include_directories(${TORCH_INCLUDE_PATH})
    endif ()

    if (NOT DEFINED TORCH_LIBRARY_PATH)
        message(STATUS "Auto detecting Torch library directory...")
        system(
                STRIP OUTPUT_VARIABLE TORCH_LIBRARY_PATH
                COMMAND "${PYTHON_EXECUTABLE}" -c "print('\\\;'.join(__import__('torch.utils.cpp_extension', fromlist=[None]).library_paths()))"
        )

        if ("${TORCH_LIBRARY_PATH}" STREQUAL "")
            set(TORCH_LIBRARY_PATH "${PYTHON_SITE_PACKAGES}/torch/lib")
        endif ()
    endif ()

    if ("${TORCH_LIBRARY_PATH}" STREQUAL "")
        message(FATAL_ERROR "Torch library directory not found. Got: \"${TORCH_LIBRARY_PATH}\"")
    else ()
        message(STATUS "Detected Torch library directory: \"${TORCH_LIBRARY_PATH}\"")
    endif ()

    unset(TORCH_LIBRARIES)

    foreach (VAR_PATH ${TORCH_LIBRARY_PATH})
        file(GLOB TORCH_LIBRARY "${VAR_PATH}/*")
        message(STATUS "Detected Torch libraries: \"${TORCH_LIBRARY}\"")
    endforeach ()

    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    set(DLLM_FLASH_ATTENTION_SRC test_flash_attention.cpp)
else ()
    set(TORCH_INCLUDE_PATH)
    set(TORCH_LIBRARY)
    set(Python_LIBRARIES)
    set(DLLM_FLASH_ATTENTION_SRC)
endif ()

add_executable(dllm_tests
        main.cpp
        test_fc.cpp
        test_relu.cpp
        test_gelu.cpp
        ${DLLM_FLASH_ATTENTION_SRC}
)

target_include_directories(dllm_tests PRIVATE
        ${TORCH_INCLUDE_PATH}
)

target_link_libraries(dllm_tests PRIVATE
        dllm
        spdlog::spdlog
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
        GTest::gtest
        ${TORCH_LIBRARY}
        ${Python_LIBRARIES}
)

target_compile_options(dllm_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)

add_executable(dllm_communication_tests
        main_communication.cpp
        test_all_reduce.cpp
)

target_include_directories(dllm_communication_tests PRIVATE
        ${TORCH_INCLUDE_PATH}
)

target_link_libraries(dllm_communication_tests PRIVATE
        dllm
        spdlog::spdlog
        CUDA::cudart
        CUDA::cublas
        Eigen3::Eigen
        GTest::gtest
        MPI::MPI_CXX
        ${NCCL_LIBRARIES}
        ${TORCH_LIBRARY}
        ${Python_LIBRARIES}
)

target_compile_options(dllm_communication_tests PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0>
)
