cmake_minimum_required(VERSION 3.5)
project(avx2_sort_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -mavx2")

add_executable(avx2_sort_demo main.cpp)