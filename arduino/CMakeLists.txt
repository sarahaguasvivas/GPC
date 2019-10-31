#name of the project
PROJECT (GPC)

# Minimum version of cmake required
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# Add other flags to the compiler
SET (CMAKE_CXX_FLAGS "-g")

# Use C++11
SET (CMAKE_CXX_STANDARD 11)

# Version Number
SET (GPC_VERSION_MAJOR 0)
SET (GPC_VERSION_MINOR 0)

# Bring the headers and source files into the project
INCLUDE_DIRECTORIES (src)
INCLUDE_DIRECTORIES (include)

FILE (GLOB CORE_LIB src/*.cpp)

# Build libraries
ADD_LIBRARY (GPC_core SHARED ${CORE_LIB})

# Add the executable
ADD_EXECUTABLE (test test/test.cpp)

TARGET_LINK_LIBRARIES (test GPC_core)
