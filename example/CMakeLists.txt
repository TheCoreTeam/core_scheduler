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
