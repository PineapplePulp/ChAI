




add_executable(TorchLoad 
    ${CMAKE_CURRENT_SOURCE_DIR}/torch_load.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(TorchLoad bridge)
add_dependencies(TorchLoad ChAI)
target_link_options(TorchLoad
    PRIVATE
    ${CHAI_LINKER_ARGS}
)

add_custom_command(TARGET TorchLoad POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/models"
          "$<TARGET_FILE_DIR:TorchLoad>/models"
  COMMENT "Copying model folder"
)