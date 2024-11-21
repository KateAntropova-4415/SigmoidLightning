include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG master
)

FetchContent_MakeAvailable(pybind11)

cmake_minimum_required(VERSION 3.12)
project(sigmoid_example)

# Укажите версию Python и путь к pybind11
find_package(Python3 REQUIRED)

# Укажите исходные файлы
set(SOURCES src/sigmoid.cpp bindings/bindings.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Создайте модуль для Python
pybind11_add_module(sigmoid_core ${SOURCES})
