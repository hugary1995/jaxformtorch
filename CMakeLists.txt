CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(JAXFORMTORCH LANGUAGES CXX)

# Let's be very careful
add_compile_options(-Wall -Wextra -pedantic)

# For debugging
option(DEBUG "include debugging information" OFF)

if(DEBUG)
  message(STATUS "Configure to include debugging information")
  add_compile_options(-g -O0)
endif()

# Enable testing
enable_testing()

# c++ 17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# torch
find_package(Torch REQUIRED)

# Catch2, for testing
find_package(Catch2 REQUIRED)

# MKL
message(STATUS "MKL_ROOT=${MKL_ROOT}")
LINK_DIRECTORIES(${MKL_ROOT}/lib)

# tests
add_subdirectory(tests)
