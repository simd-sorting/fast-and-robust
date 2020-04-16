cmake_minimum_required(VERSION 3.5)
project(sort_16_int)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -mavx2")

add_executable(sort_16_int sort_16_int.cpp)
