

add_executable(MyExample 
    ${PROJECT_ROOT_DIR}/examples/my_example/my_example.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(MyExample bridge)
add_dependencies(MyExample ChAI)
target_link_options(MyExample
    PRIVATE
    ${CHAI_LINKER_ARGS}
)


# add_custom_command(TARGET MyMyExampleApp POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory
#           "${CMAKE_CURRENT_SOURCE_DIR}/resources"
#           "$<TARGET_FILE_DIR:MyExample>/resources"
#   COMMENT "Copying runtime resources"
# )