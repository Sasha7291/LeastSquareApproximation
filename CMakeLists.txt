cmake_minimum_required(VERSION 3.16)

set(LEAST_SQUARE_APPROXIMATION_LIB_NAME lsa)

project(${LEAST_SQUARE_APPROXIMATION_LIB_NAME} LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Core)

set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SOURCES ${SRC_ROOT}/*.cpp)
file(GLOB HEADERS ${INCLUDE_ROOT}/*.h ${INCLUDE_ROOT}/*.hpp)

add_library(${PROJECT_NAME} STATIC ${HEADERS} ${SOURCES})
add_subdirectory(lib/dynamic_matrix/)
add_subdirectory(lib/psr/)
target_link_libraries(${PROJECT_NAME} PUBLIC
  dynamic_matrix
  psr
)

if (${Qt6_FOUND})
  target_compile_definitions(${PROJECT_NAME} PRIVATE QT)
  target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
  ${INCLUDE_ROOT}
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
  ${INCLUDE_ROOT}
  lib/dynamic_matrix/include
  lib/psr/include
)
