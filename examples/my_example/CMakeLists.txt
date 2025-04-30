

add_executable(MyExample 
    ${CMAKE_CURRENT_SOURCE_DIR}/my_example.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(MyExample bridge)
add_dependencies(MyExample ChAI)
target_link_options(MyExample
    PRIVATE
    ${CHAI_LINKER_ARGS}
)

set_target_properties(MyExample PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)


# add_custom_command(TARGET MyMyExampleApp POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory
#           "${CMAKE_CURRENT_SOURCE_DIR}/resources"
#           "$<TARGET_FILE_DIR:MyExample>/resources"
#   COMMENT "Copying runtime resources"
# )