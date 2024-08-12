#setting minimum required verstion of cmake to 3.1
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(Neural_Network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find libraries
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Adding libraries 
add_library(lib_num_recognizer  "include/num_recognizer.hpp")
add_library(lib_ann_trainer "include/ann_trainer.hpp")

# Setting linker languages for library to c++
set_target_properties(lib_num_recognizer PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(lib_ann_trainer PROPERTIES LINKER_LANGUAGE CXX)

# Include these directories
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# Specifying the target include dir for the libraries
target_include_directories(lib_num_recognizer PUBLIC "${CMAKE_SOURCE_DIR}/include/")
target_include_directories(lib_ann_trainer PUBLIC "${CMAKE_SOURCE_DIR}/include/")

# Add the executable
add_executable(convert_images app/convert_images.cpp)
add_executable(num_recognizer app/num_recognizer.cpp)
add_executable(ann_trainer app/ann_trainer.cpp)

# Linking the executables to the libraries
target_link_libraries(num_recognizer lib_num_recognizer ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(convert_images ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(ann_trainer lib_num_recognizer lib_ann_trainer ${OpenCV_LIBS} Eigen3::Eigen)

file(COPY ${CMAKE_SOURCE_DIR}/lib DESTINATION ${CMAKE_BINARY_DIR})

# Installation rules
install(TARGETS convert_images num_recognizer ann_trainer RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_BINARY_DIR}/lib/images DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/lib/weights.bin ${CMAKE_BINARY_DIR}/lib/biases.bin DESTINATION lib)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/lib/demo DESTINATION bin)
