cmake_minimum_required(VERSION 2.8)
project(main)
add_executable(main src/main.cpp)

add_definitions(-std=c++11)

set(eigen_DIR "/usr/local/Cellar/eigen/3.3.7/share/eigen3/cmake")


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH 
${PROJECT_SOURCE_DIR}/bin 
)
set(LIBRARY_OUTPUT_PATH
    ${PROJECT_SOURCE_DIR}/lib
)                                        
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
link_directories(
    ${OpenCV_LIBRARY_DIRS}
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

