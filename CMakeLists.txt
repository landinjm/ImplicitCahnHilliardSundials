cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME implicit_cahn_hilliard)
project(${PROJECT_NAME})

# Global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the build types
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(
    CMAKE_BUILD_TYPE
    "DebugRelease"
    CACHE STRING
    "Choose the type of build, options are: Debug, Release and DebugRelease."
    FORCE
  )
endif()

# Find deal.II
find_package(
  deal.II
  9.7.0
  HINTS
    ${deal.II_DIR}
    ${DEAL_II_DIR}
    $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not find a valid deal.II installation.")
endif()
if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_LAPACK)
  message(FATAL_ERROR "Could not find a valid deal.II installation.")
endif()

# deal.II cached variables
deal_ii_initialize_cached_variables()

# The source and header files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.hpp")

# Add the main file
list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# Add executable and link libraries
add_executable(
  ${PROJECT_NAME}_debug
  ${SOURCES}
  ${HEADERS}
)
set_property(
  TARGET
    ${PROJECT_NAME}_debug
  PROPERTY
    OUTPUT_NAME
      main-debug
)
target_include_directories(
  ${PROJECT_NAME}_debug
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_flags(${PROJECT_NAME}_debug PRIVATE
	"$<COMPILE_LANGUAGE:CXX>" "${DEAL_II_WARNING_FLAGS} ${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_DEBUG}"
)
target_link_flags(${PROJECT_NAME}_debug PRIVATE
    "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_DEBUG}"
)
target_link_libraries(${PROJECT_NAME}_debug PRIVATE dealii::dealii_debug)

add_executable(
  ${PROJECT_NAME}_release
  ${SOURCES}
  ${HEADERS}
)
set_property(
  TARGET
    ${PROJECT_NAME}_release
  PROPERTY
    OUTPUT_NAME
      main-release
)
target_include_directories(
  ${PROJECT_NAME}_release
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_flags(${PROJECT_NAME}_release PRIVATE
	"$<COMPILE_LANGUAGE:CXX>" "${DEAL_II_WARNING_FLAGS} ${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_RELEASE}"
)
target_link_flags(${PROJECT_NAME}_release PRIVATE
	"${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_RELEASE}"
)
target_link_libraries(${PROJECT_NAME}_release PRIVATE dealii::dealii_release)
