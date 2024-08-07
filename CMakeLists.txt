#setting minimum required verstion of cmake to 3.1
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Neural_Network LANGUAGES CXX)

#
set(CMAKE_CXX_STANDERD 20)
set(CMAKE_CXX_STANDERD_REQUIRED True)

#find librarys
find_package(OpenCV REQUIRED)

#include librarys 
include_directories(${CURL_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

#add the executable
add_executable(convert_images app/convert_images.cpp)

#link librarys 
target_link_libraries(convert_images ${CURL_LIBRARIES} ${ZLIB_LIBRARIES} ${OpenCV_LIBS})
