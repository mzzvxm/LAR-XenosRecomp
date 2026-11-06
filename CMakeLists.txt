cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)

project("XenosRecomp-ALL")

option(XENOS_RECOMP_BUILD_TOOL "Build the XenosRecomp command line tool (needs the zstd, xxHash, smol-v and DXC submodules)" ON)

set(XENOS_RECOMP_THIRDPARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

add_subdirectory(${XENOS_RECOMP_THIRDPARTY_ROOT})
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/XenosRecomp")
