#setting minimum required verstion of cmake to 3.1
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Neural_Network LANGUAGES CXX)

#
set(CMAKE_CXX_STANDERD 20)
set(CMAKE_CXX_STANDERD_REQUIRED True)

#find librarys
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

#adding librarys 
add_library(lib_num_recognizer "app/num_recognizer.cpp" "include/num_recognizer.hpp")

#setting linker languages for library to c++
set_target_properties(lib_num_recognizer PROPERTIES LINKER_LANGUAGE CXX)

#include librarys 
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

#specifying the target include dir for the librarys 
target_include_directories(lib_num_recognizer PUBLIC "${CMAKE_SOURCE_DIR}/include/")

#add the executable
add_executable(convert_images app/convert_images.cpp)
add_executable(num_recognizer app/num_recognizer.cpp include/num_recognizer.hpp)

#linking the executables to the librarys
target_link_libraries(num_recognizer lib_num_recognizer Eigen3::Eigen)
target_link_libraries(convert_images ${OpenCV_LIBS} Eigen3::Eigen)
