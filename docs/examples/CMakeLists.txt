project(examples LANGUAGES CXX)

# EXECUTABLES ---------------------------------
add_executable(mnist mnist.cpp)
target_link_libraries(mnist PRIVATE lamppp)