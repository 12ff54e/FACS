cmake_minimum_required(VERSION 3.15)
project(
    FACS
    VERSION 0.2
    HOMEPAGE_URL "https://github.com/12ff54e/FACS.git"
    LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to debug.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build type = " ${CMAKE_BUILD_TYPE})

# Check compile flags availability
include(CheckCXXCompilerFlag)

set(asan_flag -fsanitize=address)
set(CMAKE_REQUIRED_LIBRARIES ${asan_flag}) # This is also a linker flag
check_cxx_compiler_flag(${asan_flag} compiler_support_asan)
if(${compiler_support_asan})
    add_compile_options($<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${asan_flag}>)
    add_link_options($<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${asan_flag}>)
endif()

# Add options according to compiler.
if(MSVC)
    set(CMAKE_CXX_FLAGS_INIT "/W4 /EHsc")
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
        set(CMAKE_CXX_FLAGS_INIT "-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-exit-time-destructors")
    else()
        set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra")
    endif()
endif()

add_compile_definitions(INTP_CELL_LAYOUT)

add_executable(facs src/main.cpp src/gFileRawData.cpp)
target_include_directories(facs PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_executable(index EXCLUDE_FROM_ALL)
set_target_properties(index PROPERTIES SUFFIX .html)
target_sources(index PRIVATE src/main src/gFileRawData.cpp)
target_include_directories(index PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# specify html template
string(JOIN "" shell_file ${CMAKE_CURRENT_SOURCE_DIR}/src/index_template.html)
target_link_options(index PUBLIC --shell-file ${shell_file})
set_property(TARGET index PROPERTY LINK_DEPENDS ${shell_file})
# allow exceptions
target_compile_options(index PUBLIC -fwasm-exceptions)
target_link_options(index PUBLIC -fwasm-exceptions)
# Don't run main function at startup
target_link_options(index PUBLIC -sINVOKE_RUN=0 -sEXPORTED_FUNCTIONS=_main)
# memory settings
target_link_options(index PUBLIC -sALLOW_MEMORY_GROWTH=1)
# wasm worker
target_compile_options(index PUBLIC -sWASM_WORKERS=1)
target_link_options(index PUBLIC -sWASM_WORKERS=1)
