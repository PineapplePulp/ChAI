cmake_minimum_required(VERSION 3.31 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

include(dlcache)

include(CMakePrintHelpers)

set(TORCH_URL_PREFIX "https://download.pytorch.org/libtorch/nightly/cpu")




if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
elseif(WIN32)
    message(FATAL_ERROR "Windows is not supported yet.")
endif()

if(APPLE)
    set(TORCH_DISTRIBUTION "${TORCH_URL_PREFIX}/libtorch-macos-arm64-latest.zip")
elseif(LINUX)
    set(TORCH_DISTRIBUTION "$${TORCH_URL_PREFIX}/libtorch-cxx11-abi-static-with-deps-latest.zip")
endif()

function(download_libtorch)
    cmake_parse_arguments(
        DL_TORCH
        "CLEAR_CACHE"
        "CACHE_DIR"
        "DESTINATION"
        ${ARGV}
    )

    cmake_print_variables(DL_TORCH_CLEAR_CACHE)
    cmake_print_variables(DL_TORCH_CACHE_DIR)
    cmake_print_variables(DL_TORCH_DESTINATION)



    if(DL_TORCH_CACHE_DIR)
        set(TORCH_CACHE_DIR ${DL_TORCH_CACHE_DIR})
    else()
        set(TORCH_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache")
    endif()

    set(TORCH_CACHE_NAME "libtorch_cache.zip")
    set(TORCH_CACHE "${TORCH_CACHE_DIR}/${TORCH_CACHE_NAME}")
    # set(TORCH_DIR "${TORCH_CACHE_DIR}/libtorch")


    if(DL_TORCH_DESTINATION)
        set(TORCH_DIR ${DL_TORCH_DESTINATION})
    else()
        set(TORCH_DIR "${TORCH_CACHE_DIR}/torchlib")
    endif()
    
    if(EXISTS ${TORCH_CACHE})
        if(DL_TORCH_CLEAR_CACHE)
            message(STATUS "Clearing cache: ${TORCH_CACHE}")
            file(REMOVE "${TORCH_CACHE}")
            set(SHOULD_DOWNLOAD TRUE)
        else()
            message(STATUS "Cache exists, skipping download.")
            set(SHOULD_DOWNLOAD FALSE)
        endif()
    else()
        message(STATUS "Cache does not exist, downloading.")
        set(SHOULD_DOWNLOAD TRUE)
        message(STATUS "Creating cache directory: ${TORCH_CACHE_DIR}")
        file(MAKE_DIRECTORY "${TORCH_CACHE_DIR}")
    endif()

    if(SHOULD_DOWNLOAD)
        message(STATUS "Downloading from ${TORCH_DISTRIBUTION}")
        file(DOWNLOAD "${TORCH_DISTRIBUTION}" "${TORCH_CACHE}" STATUS dlstatus)
        list(GET dlstatus 0 status_code)
        if(status_code)
            message(FATAL_ERROR "${dlstatus} (${url})")
        endif()
        message(STATUS "Downloaded ${TORCH_DISTRIBUTION} to ${TORCH_CACHE}")

        if(EXISTS ${TORCH_DIR})
            message(STATUS "Removing existing directory: ${TORCH_DIR}")
            file(REMOVE_RECURSE "${TORCH_DIR}")
            file(MAKE_DIRECTORY "${TORCH_DIR}")
        endif()
        message(STATUS "Unpacking ${TORCH_CACHE} to ${TORCH_DIR}")

        cmake_print_variables(TORCH_DIR)

        file(MAKE_DIRECTORY "${TORCH_CACHE_DIR}/tmp")
        file(ARCHIVE_EXTRACT 
                INPUT ${TORCH_CACHE} 
                DESTINATION "${TORCH_CACHE_DIR}/tmp"
        )
        file(RENAME "${TORCH_CACHE_DIR}/tmp/libtorch" "${TORCH_DIR}")

        if(NOT EXISTS "${TORCH_CACHE_DIR}/tmp")
            message(FATAL_ERROR "Failed to extract ${TORCH_CACHE} to ${TORCH_DIR}")
        else()
            message(STATUS "Extracted ${TORCH_CACHE} to ${TORCH_DIR}")
            file(REMOVE_RECURSE "${TORCH_CACHE_DIR}/tmp")
        endif()
        
    endif()

endfunction()

# dlcache("${TORCH_DISTRIBUTION}" OUT url)

# message(STATUS "file loc >>>>> ${url}")