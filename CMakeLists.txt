project(DE)
include_directories(include)
# We heavily use C++11, C++11 support is required
set (CMAKE_CXX_STANDARD 11)
add_executable(de main.cpp)