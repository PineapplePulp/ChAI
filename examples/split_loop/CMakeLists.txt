add_executable(SplitLoop 
    ${CMAKE_CURRENT_SOURCE_DIR}/split_loop.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(SplitLoop bridge)
add_dependencies(SplitLoop ChAI)
target_link_options(SplitLoop
    PRIVATE
    ${CHAI_LINKER_ARGS}
)

set_target_properties(SplitLoop PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)