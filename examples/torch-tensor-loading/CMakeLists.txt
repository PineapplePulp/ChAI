




# add_executable(TorchLoad 
#     ${CMAKE_CURRENT_SOURCE_DIR}/torch_load.chpl
#     ${CHAI_LIB_FILES}
# )
# add_dependencies(TorchLoad bridge)
# add_dependencies(TorchLoad ChAI)
# add_dependencies(TorchLoad bridge_objs)
# target_link_options(TorchLoad
#     PRIVATE
#       --main-module torch_load.chpl
#       -M ${PROJECT_ROOT_DIR}/lib
#       ${CHAI_LINKER_ARGS}
# )

chai_add_executable(TorchLoad
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_load.chpl
  ${PROJECT_ROOT_DIR}/lib
)

add_custom_command(TARGET TorchLoad POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/tensors"
          "${CMAKE_BINARY_DIR}/examples/TorchLoad/tensors"
  COMMENT "Copying tensor folder"
)

set_target_properties(TorchLoad PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)


# # === 1) The helper-function:
# function(chpl_add_executable TARGET MAIN_CHPL)
#   add_executable(${TARGET}
#     ${MAIN_CHPL}
#     ${CHAI_LIB_FILES}
#   )

#   add_dependencies(${TARGET}
#     bridge
#     ChAI
#     bridge_objs
#     TorchBridge
#   )

#   get_filename_component(_main_base ${MAIN_CHPL} NAME)

#   set(_module_dirs "${PROJECT_ROOT_DIR}/lib")
#   if(ARGN)
#     list(APPEND _module_dirs ${ARGN})
#   endif()

#   set(_m_flags "")
#   foreach(_d       IN LISTS _module_dirs)
#     list(APPEND _m_flags "-M" "${_d}")
#   endforeach()

#   target_link_options(${TARGET}
#     PRIVATE
#       --main-module ${_main_base}
#       ${_m_flags}
#       ${CHAI_LINKER_ARGS}
#   )
# endfunction()




#
#  Updated chpl_add_executable:
#




# === 3) Replace TorchLoad:
# Suppose TorchLoad also needs an additional module directory at
# ${PROJECT_ROOT_DIR}/torch_extra_modules:
chai_add_executable(TorchLoad2
  ${CMAKE_CURRENT_SOURCE_DIR}/torch_load.chpl
  ${PROJECT_ROOT_DIR}/lib
)

# Then re‐attach the “copy tensors” step and RUNTIME_OUTPUT_DIRECTORY:
add_custom_command(TARGET TorchLoad2 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/tensors"
          "${CMAKE_BINARY_DIR}/examples/TorchLoad/tensors"
  COMMENT "Copying tensor folder"
)
set_target_properties(TorchLoad2 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
