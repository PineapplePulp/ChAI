add_library(BridgeUtil INTERFACE)

# Tell consumers where to find the headers
target_include_directories(BridgeUtil
    INTERFACE               # usage requirement
        ${CMAKE_CURRENT_SOURCE_DIR}/util
)

# (optional) expose compile options / definitions
# target_compile_features(myheaders INTERFACE cxx_std_20)
# target_compile_definitions(myheaders INTERFACE MYLIB_USE_SOMETHING=1)

# (optional) make an alias so callers can use myproject::myheaders
# add_library(BridgeUtil::myheaders ALIAS myheaders)
