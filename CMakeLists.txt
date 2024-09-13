cmake_minimum_required(VERSION 3.12)
project(autodiff)

set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "No build type selected, default to Debug" FORCE)
endif()
string (ASCII 27 ESC)
set(COLOR_GREEN "${ESC}[1;32m")
set(COLOR_RED "${ESC}[1;31m")
set(COLOR_RESET "${ESC}[0m")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message ("-- ${COLOR_RED}Debug mode${COLOR_RESET}")
    add_compile_options(-DDEBUG -g)
    add_compile_options(-fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
    add_link_options(-fno-omit-frame-pointer -fsanitize=undefined,address,leak,null,bounds)
else()
    message ("-- ${COLOR_GREEN}Release mode${COLOR_RESET}")
    add_compile_options(-DRELEASE -O3)
endif()

# Set the output directory for the binary library
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Set the source directory
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

set(AUTODIFF_SOURCE ${CMAKE_SOURCE_DIR}/src/autodiff.cpp)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Create the binary library
add_library(autodiff STATIC ${AUTODIFF_SOURCE})
# add_library(autodiff_dyn SHARED ${AUTODIFF_SOURCE})
# set_target_properties(autodiff_dyn PROPERTIES OUTPUT_NAME "autodiff") 

# Set the output directory for the binary executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add the main executable target
add_executable(main ${SOURCE_DIR}/main.cpp)

# Link the autodiff library to the main executable
target_link_libraries(main PRIVATE autodiff)