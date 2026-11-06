project(XenosRecomp)

option(XENOS_RECOMP_BUILD_TOOL "Build the XenosRecomp command line tool (needs the zstd, xxHash, smol-v and DXC submodules)" ON)

if (WIN32)
    option(XENOS_RECOMP_DXIL "Generate DXIL shader cache" ON)
endif()

# The translator as a library: shader_recompiler.cpp and nothing else, so a
# program can embed it without zstd, xxHash, smol-v or DXC. Needs fmt::fmt.
add_library(XenosRecompLib STATIC
    be.h
    constant_table.h
    container_scan.h
    shader.h
    shader_code.h
    shader_recompiler.cpp
    shader_recompiler.h)

target_compile_features(XenosRecompLib PUBLIC cxx_std_17)

# Programs include the headers as "XenosRecomp/<name>.h".
target_include_directories(XenosRecompLib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_include_directories(XenosRecompLib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(XenosRecompLib PRIVATE fmt::fmt)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(XenosRecompLib PRIVATE -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions)
endif()

if (WIN32)
    target_compile_definitions(XenosRecompLib PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if (NOT XENOS_RECOMP_BUILD_TOOL)
    return()
endif()

set(SMOLV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/smol-v/source")

add_executable(XenosRecomp 
    constant_table.h
    container_scan.h
    dxc_compiler.cpp
    dxc_compiler.h
    main.cpp
    pch.h
    shader.h
    shader_code.h
    shader_recompiler.h
    "${SMOLV_SOURCE_DIR}/smolv.cpp")

target_link_libraries(XenosRecomp PRIVATE
    XenosRecompLib
    Microsoft::DirectXShaderCompiler
    xxHash::xxhash
    libzstd_static
    fmt::fmt)

target_include_directories(XenosRecomp PRIVATE ${SMOLV_SOURCE_DIR})

target_precompile_headers(XenosRecomp PRIVATE pch.h)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(XenosRecomp PRIVATE -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions)

    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if(LIBCPP)
        # Allows using std::execution
        target_compile_options(XenosRecomp PRIVATE -fexperimental-library)
    endif()
endif()

if (WIN32)
    target_compile_definitions(XenosRecomp PRIVATE _CRT_SECURE_NO_WARNINGS)
    add_custom_command(TARGET XenosRecomp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:XenosRecomp> $<TARGET_FILE_DIR:XenosRecomp>
        COMMAND_EXPAND_LISTS
    )
endif()

if (XENOS_RECOMP_DXIL)
    target_compile_definitions(XenosRecomp PRIVATE XENOS_RECOMP_DXIL)
    target_link_libraries(XenosRecomp PRIVATE Microsoft::DXIL)
endif()
