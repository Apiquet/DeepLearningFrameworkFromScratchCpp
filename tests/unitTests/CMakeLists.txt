# source files
include_directories(
    include
)

# source files
file(GLOB SOURCES "../../src/*/*.cpp" "src/*.cpp" runAllTests.cpp)

add_executable(UnitTests ${SOURCES})
