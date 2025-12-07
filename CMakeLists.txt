cmake_minimum_required(VERSION 3.15)

project(Galam
    VERSION 1.0
    LANGUAGES CXX
)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Common sources (include match_test.cpp so MatchTest symbols are available)
set(GALAM_SOURCES
    galam.cpp
    match_test.cpp
    main.cpp
)

# Main executable
add_executable(${PROJECT_NAME}
    ${GALAM_SOURCES}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

# Use OpenCV's provided libs list (more robust than listing each one)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${OpenCV_LIBS}
)

# -------------------------------------------------------------------
# Test executable
#   For now this uses the same sources; if you later add a separate
#   main (e.g., test_main.cpp), just change the source list here.
# -------------------------------------------------------------------
add_executable(test_galam
    ${GALAM_SOURCES}
)

target_include_directories(test_galam
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_galam
    PRIVATE
        ${OpenCV_LIBS}
)
