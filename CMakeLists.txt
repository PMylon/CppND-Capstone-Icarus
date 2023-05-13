cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_COMPILER g++-9)
set(CXX_FLAGS "-Wall -g -Og")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)

project(Icarus)

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/thirdparty/onnxruntime/lib/libonnxruntime.so)
target_include_directories(onnxruntime INTERFACE thirdparty/onnxruntime/include)

find_package(OpenCV 4 REQUIRED)

add_executable(Icarus src/main.cpp src/image_provider.cpp src/image_preprocessor.cpp src/model_handler.cpp src/runtime.cpp)
set_property(TARGET Icarus PROPERTY CXX_STANDARD 17)
target_include_directories(Icarus PRIVATE ${OpenCV_INCLUDE_DIRS} src)
target_link_libraries(Icarus onnxruntime ${OpenCV_LIBS} pthread)
