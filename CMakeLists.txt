#setting minimum required verstion of cmake to 3.1
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Neural_Network LANGUAGES CXX)

#
set(CMAKE_CXX_STANDERD 20)
set(CMAKE_CXX_STANDERD_REQUIRED True)

#find librarys
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

#include librarys 
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

#add the executable
add_executable(convert_images app/convert_images.cpp)

#link librarys 
target_link_libraries(convert_images ${OpenCV_LIBS} Eigen3::Eigen)
