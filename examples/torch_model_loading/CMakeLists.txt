




add_executable(TorchLoad 
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_load.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(TorchLoad bridge)
add_dependencies(TorchLoad ChAI)
add_dependencies(TorchLoad bridge_objs)
target_link_options(TorchLoad
    PRIVATE
      --main-module torch_load.chpl
      -M ${PROJECT_ROOT_DIR}/lib
      ${CHAI_LINKER_ARGS}
)

add_custom_command(TARGET TorchLoad POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/models"
          "${CMAKE_BINARY_DIR}/TorchLoad_models"
  COMMENT "Copying model folder"
)

set_target_properties(TorchLoad PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)