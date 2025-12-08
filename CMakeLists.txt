cmake_minimum_required(VERSION 3.5)
project(DPC-ANN LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# -------------------------- Includes ------------------------------------------

include_directories("src")
include_directories("ParlayANN/parlaylib/include")
include_directories("ParlayANN")
include_directories("apps")

# -------------------------- Options ------------------------------------------

set(COMPILE_OPTIONS
    $<$<CONFIG:Debug>:-std=c++17 -DPARLAY_SEQUENTIAL -mcx16 -pthread -march=native -g -O0 -DDEBUG -fPIC>
    $<$<CONFIG:RelWithDebInfo>:-std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG -fPIC -g>
    $<$<CONFIG:Release>:-std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG -fPIC>
)

# --------------------- Create Python Library --------------------------------------

set(CC_SOURCES
    python_bindings/python_bindings.cpp
)

pybind11_add_module(window_ann ${CC_SOURCES})
target_compile_options(window_ann PRIVATE ${COMPILE_OPTIONS})

# --------------------- Create Wrapper Executables --------------------------------------

# Data converter
add_executable(fvecs_to_bin apps/fvecs_to_bin.cpp)
target_compile_options(fvecs_to_bin PRIVATE ${COMPILE_OPTIONS})

# Build wrapper
add_executable(build_wrapper apps/build_wrapper.cpp)
target_compile_options(build_wrapper PRIVATE ${COMPILE_OPTIONS})
target_compile_definitions(build_wrapper PRIVATE NO_PYTHON_BINDINGS)
target_link_libraries(build_wrapper OpenMP::OpenMP_CXX)

# Search wrapper  
add_executable(search_wrapper apps/search_wrapper.cpp)
target_compile_options(search_wrapper PRIVATE ${COMPILE_OPTIONS})
target_compile_definitions(search_wrapper PRIVATE NO_PYTHON_BINDINGS)
target_link_libraries(search_wrapper OpenMP::OpenMP_CXX)
