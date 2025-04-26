




add_executable(WebcamFilter 
    ${CMAKE_CURRENT_SOURCE_DIR}/WebcamFilter.chpl
    ${CHAI_LIB_FILES}
)
add_dependencies(WebcamFilter bridge)
add_dependencies(WebcamFilter ChAI)
target_link_options(WebcamFilter
    PRIVATE
    ${CHAI_LINKER_ARGS}
)

# add_custom_command(TARGET WebcamFilter POST_BUILD
#   COMMAND ${CMAKE_COMMAND} -E copy_directory
#           "${CMAKE_CURRENT_SOURCE_DIR}/models"
#           "$<TARGET_FILE_DIR:WebcamFilter>/models"
#   COMMENT "Copying model folder"
# )