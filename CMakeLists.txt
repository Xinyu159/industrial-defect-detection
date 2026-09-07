cmake_minimum_required(VERSION 3.16)
project(industrial_defect_detection VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(defect_detector
    src/main.cpp
    src/preprocess.cpp
    src/segmentation.cpp
    src/visualize.cpp
)

target_include_directories(defect_detector PRIVATE include)
target_link_libraries(defect_detector PRIVATE ${OpenCV_LIBS})
