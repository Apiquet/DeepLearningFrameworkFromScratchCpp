cmake_minimum_required(VERSION 3.0.0)
project(TestDeepLearningFramework VERSION 0.1.0)

include(CTest)
enable_testing()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/Activations
    ${PROJECT_SOURCE_DIR}/include/Data
    ${PROJECT_SOURCE_DIR}/include/Layers
    ${PROJECT_SOURCE_DIR}/include/Losses
    ${PROJECT_SOURCE_DIR}/include/Metrics
    ${PROJECT_SOURCE_DIR}/include/Module
    ${PROJECT_SOURCE_DIR}/include/Sequential
    ${PROJECT_SOURCE_DIR}/include/Trainers
    ${PROJECT_SOURCE_DIR}/src/Trainers
)

# source files
file(GLOB SOURCES "src/*/*.cpp" tests/main.cpp)

if(BUILD_TESTING)
  add_subdirectory(tests/unitTests)
endif()

add_executable(TestDeepLearningFramework ${SOURCES})
