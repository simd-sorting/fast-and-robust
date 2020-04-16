cmake_minimum_required(VERSION 3.5)
project(sort_128_int_no_transpose)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -mavx2")

add_executable(sort_128_int_no_transpose sort_128_int_no_transpose.cpp)
