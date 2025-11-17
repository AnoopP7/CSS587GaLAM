cmake_minimum_required(VERSION 3.15)
project(Galam VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
set(SOURCES galam.cpp main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        opencv_core
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        opencv_features2d
        opencv_calib3d
)