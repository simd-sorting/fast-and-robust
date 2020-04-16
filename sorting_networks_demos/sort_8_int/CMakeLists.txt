cmake_minimum_required(VERSION 3.5)
project(sort_8_int)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -mavx2")

add_executable(sort_8_int sort_8_int.cpp)
