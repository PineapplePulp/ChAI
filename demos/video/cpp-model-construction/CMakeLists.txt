include(CMakePrintHelpers)

# project(MyProject)
# set(CMAKE_CXX_STANDARD 17)
# list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake")
find_package(Torch REQUIRED)  
find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -lm -ldl")

cmake_print_variables(TORCH_LIBRARIES)
cmake_print_variables(TORCH_INCLUDE_DIRS)
cmake_print_variables(TORCH_INSTALL_PREFIX)
cmake_print_variables(TORCH_CXX_FLAGS)
cmake_print_variables(TORCH_LIBRARY)

add_executable(CPPModelConstruction ${CMAKE_CURRENT_SOURCE_DIR}/net.cpp ${CMAKE_CURRENT_SOURCE_DIR}/transformer_net.hpp)

target_link_libraries(CPPModelConstruction ${TORCH_LIBRARIES})
set_property(TARGET CPPModelConstruction PROPERTY CXX_STANDARD 23)

set_property(TARGET CPPModelConstruction PROPERTY
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(CPPModelCam ${CMAKE_CURRENT_SOURCE_DIR}/cam.cpp ${CMAKE_CURRENT_SOURCE_DIR}/transformer_net.hpp)

include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(CPPModelCam ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET CPPModelCam PROPERTY CXX_STANDARD 23)

set_property(TARGET CPPModelCam PROPERTY
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)