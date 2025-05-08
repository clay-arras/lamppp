cmake_minimum_required(VERSION 3.10)

project(test_playground LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_playground test_playground.cpp)
