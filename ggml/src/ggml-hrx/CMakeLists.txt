set(HRX_SOURCE_DIR "" CACHE PATH "Optional HRX source tree to build instead of using installed hrx and loomc packages")

if(HRX_SOURCE_DIR)
    include(ExternalProject)
    include(GNUInstallDirs)

    get_filename_component(HRX_SOURCE_DIR "${HRX_SOURCE_DIR}" ABSOLUTE)
    if(NOT EXISTS "${HRX_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "HRX_SOURCE_DIR does not contain a CMakeLists.txt: ${HRX_SOURCE_DIR}")
    endif()

    set(GGML_HRX_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/hrx")
    set(GGML_HRX_BUILD_DIR "${GGML_HRX_PREFIX}/src/ggml-hrx-deps-build")
    set(GGML_HRX_LIB "${GGML_HRX_BUILD_DIR}/libhrx/src/libhrx/${CMAKE_SHARED_LIBRARY_PREFIX}hrx${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(GGML_LOOMC_LIB "${GGML_HRX_BUILD_DIR}/loom/binding/c/${CMAKE_SHARED_LIBRARY_PREFIX}loomc${CMAKE_SHARED_LIBRARY_SUFFIX}")

    set(GGML_HRX_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DIREE_BUILD_TESTS=OFF
        -DIREE_BUILD_BENCHMARKS=OFF
        -DIREE_HAL_DRIVER_DEFAULTS=OFF
        -DIREE_HAL_DRIVER_AMDGPU=ON
        -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
        -DIREE_HAL_DRIVER_LOCAL_TASK=ON
        -DIREE_HAL_DRIVER_NULL=ON
        -DLIBHRX_BUILD_CTS=OFF
    )
    if(IREE_ROCM_PATH)
        list(APPEND GGML_HRX_CMAKE_ARGS -DIREE_ROCM_PATH=${IREE_ROCM_PATH})
    endif()
    if(FETCHCONTENT_BASE_DIR)
        list(APPEND GGML_HRX_CMAKE_ARGS -DFETCHCONTENT_BASE_DIR=${FETCHCONTENT_BASE_DIR})
    endif()

    ExternalProject_Add(ggml-hrx-deps
        SOURCE_DIR "${HRX_SOURCE_DIR}"
        PREFIX "${GGML_HRX_PREFIX}"
        CMAKE_ARGS ${GGML_HRX_CMAKE_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target hrx loomc_shared --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${GGML_HRX_LIB}" "${GGML_LOOMC_LIB}"
        UPDATE_COMMAND ""
    )

    add_library(hrx::hrx SHARED IMPORTED GLOBAL)
    set_target_properties(hrx::hrx PROPERTIES
        IMPORTED_LOCATION "${GGML_HRX_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${HRX_SOURCE_DIR}/libhrx/include")
    add_dependencies(hrx::hrx ggml-hrx-deps)

    add_library(loomc::loomc SHARED IMPORTED GLOBAL)
    set_target_properties(loomc::loomc PROPERTIES
        IMPORTED_LOCATION "${GGML_LOOMC_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${HRX_SOURCE_DIR}/loom/binding/c/include"
        INTERFACE_COMPILE_DEFINITIONS LOOMC_USING_SHARED_LIBRARY)
    add_dependencies(loomc::loomc ggml-hrx-deps)
else()
    find_package(hrx CONFIG REQUIRED)
    find_package(loomc CONFIG REQUIRED)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_QWEN_KERNEL_CORPUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/qwen_moe")
set(GGML_HRX_QWEN_KERNEL_CORPUS_MANIFEST "${GGML_HRX_QWEN_KERNEL_CORPUS_DIR}/manifest.json")
set(GGML_HRX_KERNEL_CORPUS_SOURCES_INC "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus-sources.inc")
set(GGML_HRX_KERNEL_CORPUS_QWEN_INC "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus-qwen.inc")
set(GGML_HRX_KERNEL_CORPUS_CATALOG_INC "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus-catalog.inc")
set(GGML_HRX_KERNEL_CORPUS_DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus.d")

add_custom_command(
    OUTPUT
        "${GGML_HRX_KERNEL_CORPUS_SOURCES_INC}"
        "${GGML_HRX_KERNEL_CORPUS_QWEN_INC}"
        "${GGML_HRX_KERNEL_CORPUS_CATALOG_INC}"
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_corpus.py"
        --source-output "${GGML_HRX_KERNEL_CORPUS_SOURCES_INC}"
        --corpus-output "${GGML_HRX_KERNEL_CORPUS_QWEN_INC}"
        --catalog-output "${GGML_HRX_KERNEL_CORPUS_CATALOG_INC}"
        --manifest "${GGML_HRX_QWEN_KERNEL_CORPUS_MANIFEST}"
        --corpus-dir "${GGML_HRX_QWEN_KERNEL_CORPUS_DIR}"
        --depfile "${GGML_HRX_KERNEL_CORPUS_DEPFILE}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_corpus.py"
        "${GGML_HRX_QWEN_KERNEL_CORPUS_MANIFEST}"
    DEPFILE "${GGML_HRX_KERNEL_CORPUS_DEPFILE}"
    VERBATIM
)

option(GGML_HRX_BUNDLE_RUNTIME_LIBS "Bundle HRX/ROCm runtime libraries next to the HRX backend" OFF)
set(GGML_HRX_BUNDLE_LIBRARY_DIRS "" CACHE STRING "Library directories to scan when GGML_HRX_BUNDLE_RUNTIME_LIBS=ON")

add_library(ggml-hrx-graph STATIC
    graph/fusion-search.cpp
    graph/fusion-search.h
    graph/graph-index.cpp
    graph/graph-index.h
    graph/command-program.cpp
    graph/command-program.h
    graph/graph-ir.cpp
    graph/graph-ir.h
    graph/reactive-plan.cpp
    graph/reactive-plan.h
    graph/resource-access.h
    graph/routed-transformer.cpp
    graph/routed-transformer.h
    graph/routed-transformer-bindings.cpp
    graph/routed-transformer-bindings.h
    graph/routed-transformer-program.cpp
    graph/routed-transformer-program.h
    graph/schedule.cpp
    graph/schedule.h
    kernel-corpus-json.cpp
    kernel-corpus-json.h
    kernel-corpus-catalog-verify.h
    kernel-corpus-catalog.h
    kernel-corpus.cpp
    kernel-corpus.h
    "${GGML_HRX_KERNEL_CORPUS_SOURCES_INC}"
    "${GGML_HRX_KERNEL_CORPUS_QWEN_INC}"
    "${GGML_HRX_KERNEL_CORPUS_CATALOG_INC}"
)
target_include_directories(ggml-hrx-graph PUBLIC . graph PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" .. ../../include ../../../vendor)
target_compile_features(ggml-hrx-graph PRIVATE cxx_std_17)
target_link_libraries(ggml-hrx-graph PRIVATE ggml-base)
set_target_properties(ggml-hrx-graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml-hrx-transfer STATIC
    hrx-interop-utils.h
    transfer-manager.cpp
    transfer-manager.h
    weight-residency.cpp
    weight-residency.h
)
target_include_directories(ggml-hrx-transfer PUBLIC .)
target_compile_features(ggml-hrx-transfer PRIVATE cxx_std_17)
target_link_libraries(ggml-hrx-transfer PUBLIC hrx::hrx)
set_target_properties(ggml-hrx-transfer PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml-hrx-executable STATIC
    executable-program.cpp
    executable-program.h
    loom-jit.cpp
    loom-jit.h
)
target_include_directories(ggml-hrx-executable PUBLIC . graph PRIVATE loom-jit .. ../../include ../../../vendor)
target_compile_features(ggml-hrx-executable PRIVATE cxx_std_17)
target_link_libraries(ggml-hrx-executable PUBLIC ggml-hrx-graph ggml-hrx-transfer hrx::hrx PRIVATE loomc::loomc)
set_target_properties(ggml-hrx-executable PROPERTIES POSITION_INDEPENDENT_CODE ON)

ggml_add_backend_library(ggml-hrx
    ggml-hrx.cpp
)
target_link_libraries(ggml-hrx PRIVATE hrx::hrx loomc::loomc)
target_link_libraries(ggml-hrx PRIVATE ggml-hrx-executable)
target_compile_definitions(ggml-hrx PRIVATE GGML_USE_HRX)

if (GGML_HRX_BUNDLE_RUNTIME_LIBS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleRuntime.cmake")
    ggml_hrx_bundle_runtime(ggml-hrx)
endif()

add_executable(ggml-hrx-compile-kernel
    tools/compile-kernel.cpp
    tools/tool-utils.h
    loom-jit.cpp
)
target_link_libraries(ggml-hrx-compile-kernel PRIVATE hrx::hrx loomc::loomc)
target_include_directories(ggml-hrx-compile-kernel PRIVATE .)
target_compile_features(ggml-hrx-compile-kernel PRIVATE cxx_std_17)

add_executable(ggml-hrx-record-command-buffer-smoke tools/record-command-buffer-smoke.cpp)
target_link_libraries(ggml-hrx-record-command-buffer-smoke PRIVATE hrx::hrx)
target_compile_features(ggml-hrx-record-command-buffer-smoke PRIVATE cxx_std_17)

add_executable(ggml-hrx-dump-program tools/dump-program.cpp tools/tool-utils.h)
target_link_libraries(ggml-hrx-dump-program PRIVATE ggml-hrx-graph hrx::hrx)
target_include_directories(ggml-hrx-dump-program PRIVATE . graph .. ../../include ../../../vendor)
target_compile_features(ggml-hrx-dump-program PRIVATE cxx_std_17)

add_executable(ggml-hrx-record-program tools/record-program.cpp tools/tool-utils.h)
target_link_libraries(ggml-hrx-record-program PRIVATE ggml-hrx-executable hrx::hrx)
target_include_directories(ggml-hrx-record-program PRIVATE . graph .. ../../include ../../../vendor)
target_compile_features(ggml-hrx-record-program PRIVATE cxx_std_17)
