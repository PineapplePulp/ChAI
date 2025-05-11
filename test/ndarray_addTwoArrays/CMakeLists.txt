add_executable(AddTwoArraysTest 
    ${CMAKE_CURRENT_SOURCE_DIR}/addTwoArraysTest.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(AddTwoArraysTest bridge)
add_dependencies(AddTwoArraysTest ChAI)
target_link_options(AddTwoArraysTest
    PRIVATE
    ${CHAI_LINKER_ARGS}
)

set_target_properties(AddTwoArraysTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)