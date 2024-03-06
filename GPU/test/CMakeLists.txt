cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Num1 LANGUAGES CXX CUDA)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_80,code=sm_80")

add_executable(tmptest Num1.cu)



