




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