get_filename_component(GGML_HRX_LLAMA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_HRX ON CACHE BOOL "" FORCE)
set(GGML_HRX_STATIC_RUNTIME ON CACHE BOOL "" FORCE)
set(IREE_HAL_AMDGPU_LIBHSA_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

# Apply static libhsa settings because HRX targets exist before this file is included.
if(NOT TARGET iree::third_party::hsa_runtime)
    message(FATAL_ERROR
        "GGML_HRX_STATIC_RUNTIME requires an HSA runtime package")
endif()
if(WIN32 AND NOT TARGET hsa-runtime64::hsa-runtime64_static)
    message(FATAL_ERROR
        "GGML_HRX_STATIC_RUNTIME requires a static Windows HSA runtime package")
endif()
foreach(libhsa_target
        iree_hal_drivers_amdgpu_util_libhsa
        iree_hal_drivers_amdgpu_util_libhsa.objects)
    if(NOT TARGET "${libhsa_target}")
        message(FATAL_ERROR
            "GGML_HRX_STATIC_RUNTIME requires the HRX AMDGPU libhsa targets")
    endif()
    target_compile_definitions("${libhsa_target}"
        PUBLIC IREE_HAL_AMDGPU_LIBHSA_STATIC=1)
endforeach()
target_link_libraries(iree_hal_drivers_amdgpu_util_libhsa
    PRIVATE iree_rocm_hsa_runtime)

if(WIN32)
    set(IREE_ROCM_HSA_RUNTIME_ELF_LIBRARY "" CACHE FILEPATH
        "Static oclelf library required by the Windows HSA runtime")
    if(NOT EXISTS "${IREE_ROCM_HSA_RUNTIME_ELF_LIBRARY}")
        message(FATAL_ERROR
            "GGML_HRX_STATIC_RUNTIME requires oclelf.lib from the matching HSA runtime build")
    endif()
    target_link_libraries(iree_rocm_hsa_runtime
        INTERFACE "${IREE_ROCM_HSA_RUNTIME_ELF_LIBRARY}")

    # Apply TheRock's static MSVC runtime to existing HRX targets and subsequent llama.cpp targets.
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded CACHE STRING "" FORCE)

    function(ggml_hrx_set_static_msvc_runtime directory)
        get_property(directory_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS directory_targets)
            get_target_property(target_imported "${target}" IMPORTED)
            if(NOT target_imported)
                set_property(TARGET "${target}" PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
            endif()
        endforeach()

        get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(subdirectory IN LISTS subdirectories)
            ggml_hrx_set_static_msvc_runtime("${subdirectory}")
        endforeach()
    endfunction()
    ggml_hrx_set_static_msvc_runtime("${CMAKE_SOURCE_DIR}")
endif()

add_subdirectory("${GGML_HRX_LLAMA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama-cpp")
