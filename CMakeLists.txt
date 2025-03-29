cmake_minimum_required(VERSION 3.10)
project(autograd_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_library(autograd_core
    autograd/engine.cpp
    autograd/nn.cpp
    autograd/util.cpp
)

target_include_directories(autograd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(test_engine test/test_engine.cpp)
target_link_libraries(test_engine autograd_core)

add_executable(test_nn test/test_nn.cpp)
target_link_libraries(test_nn autograd_core)

add_executable(test_gen test/test_gen.cpp)
target_link_libraries(test_gen autograd_core)

add_executable(mnist autograd/mnist.cpp)
target_link_libraries(mnist autograd_core)